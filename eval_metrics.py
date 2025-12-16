#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取训练好的 generator + df_used.pkl，
在一小部分 test 数据上计算 FID 和 LPIPS。
"""

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

DATA_PATH = "./data/"
MARKER_PATH = "./markers_official/"
models_dir = "./models"

def build_generator(template_shape=(128, 128, 1), channels=1, debug=False):
    from tensorflow.keras.layers import (Input, Dropout, Concatenate, MaxPool2D,
                                         BatchNormalization, LeakyReLU,
                                         UpSampling2D, Conv2D)
    from tensorflow.keras.models import Model
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding="same",
                   kernel_initializer=initializer)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        d = MaxPool2D(padding="same")(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, apply_dropout=False):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding="same",
                   activation="relu", kernel_initializer=initializer)(u)
        if apply_dropout:
            u = Dropout(0.5)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    d0 = Input(shape=template_shape, name="input_template")
    d1 = conv2d(d0, 32, bn=False)
    d2 = conv2d(d1, 64)
    d3 = conv2d(d2, 128)
    d4 = conv2d(d3, 256)
    d5 = conv2d(d4, 512)

    u5 = deconv2d(d5, d4, 256, apply_dropout=True)
    u6 = deconv2d(u5, d3, 128)
    u7 = UpSampling2D(size=2)(u6)
    out_img = Conv2D(channels, kernel_size=4, strides=1, padding="same",
                     activation="tanh")(u7)

    model = Model(d0, out_img, name="generator")
    if debug:
        model.summary()
    return model

def to_torch_for_fid(imgs_np):
    """
    imgs_np: (N, H, W, 1), float32 in [-1, 1]
    -> (N, 3, 299, 299), float32 in [0, 1]
    """
    x = torch.from_numpy(imgs_np)  # N,H,W,1
    x = (x + 1.0) / 2.0            # [-1,1] -> [0,1]
    x = x.permute(0, 3, 1, 2)      # -> N,1,H,W
    x = x.repeat(1, 3, 1, 2)       # -> N,3,H,W
    x = F.interpolate(x, size=(299, 299),
                      mode="bilinear", align_corners=False)
    return x

def to_torch_for_lpips(imgs_np):
    """
    imgs_np: (N, H, W, 1), float32 in [-1, 1]
    -> (N, 3, 64, 64), float32 in [-1, 1]
    """
    x = torch.from_numpy(imgs_np)  # N,H,W,1
    x = x.permute(0, 3, 1, 2)      # -> N,1,H,W
    x = x.repeat(1, 3, 1, 2)       # -> N,3,H,W
    x = F.interpolate(x, size=(64, 64),
                      mode="bilinear", align_corners=False)
    return x

def main():
    # 1. 读取训练时使用的 df
    df_path = os.path.join(models_dir, "df_used.pkl")
    if not os.path.exists(df_path):
        raise RuntimeError(f"找不到 {df_path}，请先运行 train_cgan.py")
    df = pd.read_pickle(df_path)
    print("df_loaded, 总样本数:", len(df))

    # 2. 构建 generator 并加载权重
    temp_res = (128, 128, 1)
    cap_res = (32, 32, 1)
    generator = build_generator(template_shape=temp_res, channels=1, debug=False)

    weights_path = os.path.join(models_dir, "generator_weights_latest.h5")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"找不到 {weights_path}，请先运行 train_cgan.py")
    generator.load_weights(weights_path)
    print("已加载生成器权重:", weights_path)

    # 3. 从 test 中抽一个评估子集
    df_test = df[df.Datatype == "test"].reset_index(drop=True)
    print("Test 样本数:", len(df_test))

    N_EVAL = 2000
    if len(df_test) > N_EVAL:
        df_eval = df_test.sample(N_EVAL, random_state=1).reset_index(drop=True)
    else:
        df_eval = df_test
    print("评估子集样本数:", len(df_eval))

    # 4. 生成 real/fake
    real_list = []
    fake_list = []

    for idx, row in df_eval.iterrows():
        blob = row.Blob           # (32,32)
        marker = row.Marker       # (128,128)

        blob = blob.reshape(1, 32, 32, 1).astype("float32") / 127.5 - 1.0
        marker = marker.reshape(1, 128, 128, 1).astype("float32") / 127.5 - 1.0

        gen_blob = generator(marker, training=False).numpy()  # (1,32,32,1)

        real_list.append(blob)
        fake_list.append(gen_blob)

    real_np = np.concatenate(real_list, axis=0)
    fake_np = np.concatenate(fake_list, axis=0)
    print("real_np 形状:", real_np.shape, "范围:", real_np.min(), real_np.max())
    print("fake_np 形状:", fake_np.shape, "范围:", fake_np.min(), fake_np.max())

    # 5. 计算 FID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    batch_for_fid = 64
    N = real_np.shape[0]

    for i in range(0, N, batch_for_fid):
        real_batch = real_np[i:i+batch_for_fid]
        fake_batch = fake_np[i:i+batch_for_fid]

        real_t = to_torch_for_fid(real_batch).to(device)
        fake_t = to_torch_for_fid(fake_batch).to(device)

        fid.update(real_t, real=True)
        fid.update(fake_t, real=False)

    fid_score = fid.compute().item()
    print("===== FID（评估子集）=====", fid_score)

    # 6. 计算 LPIPS
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=False  # 输入范围 [-1,1]
    ).to(device)

    batch_for_lpips = 32
    lpips_vals = []

    for i in range(0, N, batch_for_lpips):
        real_batch = real_np[i:i+batch_for_lpips]
        fake_batch = fake_np[i:i+batch_for_lpips]

        real_t = to_torch_for_lpips(real_batch).to(device)
        fake_t = to_torch_for_lpips(fake_batch).to(device)

        val = lpips_metric(fake_t, real_t)
        lpips_vals.append(val.detach().cpu())

    lpips_score = torch.mean(torch.stack(lpips_vals)).item()
    print("===== LPIPS（alex, 评估子集）=====", lpips_score)


if __name__ == "__main__":
    main()
