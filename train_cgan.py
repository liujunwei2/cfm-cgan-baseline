#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 cGAN：模板(Marker 128x128) -> Blob(32x32)，并保存生成器权重和 df。
"""

import os
import sys
import time
import datetime
import gc
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dropout, Concatenate, MaxPool2D,
                                     BatchNormalization, LeakyReLU,
                                     UpSampling2D, Conv2D)
from tensorflow.keras.models import Model

# ---------------- GPU 配置（没有 GPU 也没关系，会用 CPU） ----------------
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU 配置完成，使用:", physical_devices[0])
    except Exception as e:
        print("GPU 配置失败，改用 CPU:", e)
else:
    print("未检测到 GPU，使用 CPU。")

# ---------------- 本项目自带模块 ----------------
import simulate
import process  # 暂时没用到，但保留方便以后用

# ---------------- 路径 ----------------
DATA_PATH = "./data/"
MARKER_PATH = "./markers_official/"

checkpoint_dir = "./training_checkpoints"
models_dir = "./models"
logs_dir = "./logs"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ---------------- DataGenerator ----------------
class DataGenerator:
    """
    每次返回一批 (模板 Marker, 真实 Blob)：
    - imgs_B: 模板图 (128x128x1)，作为生成器输入
    - imgs_A: Blob 图 (32x32x1)，作为生成器目标 / 判别器输入
    """
    def __init__(self, df, batch_size=64, shuffle=True,
                 cap_res=(32, 32, 1), temp_res=(128, 128, 1)):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.df = df.reset_index(drop=True)
        self.cap_res = cap_res
        self.temp_res = temp_res

        self.n_samples = len(self.df)
        self.n_batches = self.n_samples // batch_size
        self.actual_samples = self.n_batches * batch_size

        print(f"DataGenerator: 共 {self.n_samples} 条 -> 实际使用 {self.actual_samples} 条 ({self.n_batches} 个 batch)")

    def __call__(self):
        indexes = np.arange(self.actual_samples)
        if self.shuffle:
            np.random.shuffle(indexes)

        for i in range(self.n_batches):
            batch_start = i * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_indexes = indexes[batch_start:batch_end]

            imgs_A, imgs_B = [], []

            for idx in batch_indexes:
                row = self.df.iloc[idx]
                img_A = row.Blob    # Blob: 32x32
                img_B = row.Marker  # Marker: 128x128

                # 数据增强：随机移位（训练用，eval 可关）
                if self.shuffle:
                    for shift_axis in [0, 1]:
                        shift_amount = np.random.randint(-1, 2)
                        img_A = np.roll(img_A, shift_amount, axis=shift_axis)
                        img_B = np.roll(img_B, shift_amount * 4, axis=shift_axis)

                img_A = img_A.reshape(self.cap_res)
                img_B = img_B.reshape(self.temp_res)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A, dtype=np.float32) / 127.5 - 1.0
            imgs_B = np.array(imgs_B, dtype=np.float32) / 127.5 - 1.0

            yield imgs_B, imgs_A

# ---------------- Generator / Discriminator ----------------
def build_generator(template_shape=(128, 128, 1), channels=1, debug=True):
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


def build_discriminator(cap_shape=(32, 32, 1), template_shape=(128, 128, 1), debug=True):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same",
                   kernel_initializer=initializer)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def conv2d(layer_input, filters, f_size=4, strides=1, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding="same",
                   kernel_initializer=initializer)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=cap_shape, name="input_cap")
    img_B = Input(shape=template_shape, name="input_template")

    db1 = conv2d(img_B, 16, strides=2)
    db2 = conv2d(db1, 32, strides=2)
    da1 = conv2d(img_A, 32, strides=1)

    combined = Concatenate(axis=-1)([db2, da1])
    d1 = d_layer(combined, 64, bn=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d3)

    model = Model([img_B, img_A], validity, name="discriminator")
    if debug:
        model.summary()
    return model

# ---------------- GAN 损失 ----------------
LAMBDA = 100
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total = gan_loss + LAMBDA * l1_loss
    return total, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss

@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer,
               input_image, target):
    input_image = tf.cast(input_image, tf.float32)
    target = tf.cast(target, tf.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_grads = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

# ---------------- 主流程 ----------------
def main():
    print("开始加载数据...")

    files = [
        "16h5_training.pkl",
        "16h5_validation.pkl",
        "36h11_training.pkl",
        "36h11_validation.pkl",
    ]
    lst = []
    for f in files:
        path = os.path.join(DATA_PATH, f)
        print("读取:", path)
        lst.append(pd.read_pickle(path))
    df = pd.concat(lst).reset_index(drop=True)
    del lst
    gc.collect()
    print("总样本数:", len(df))

    # 为了省显存和时间，可以只用一个子集（你可以调整 MAX_SAMPLES）
    MAX_SAMPLES = 20000
    if len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=0).reset_index(drop=True)
        print("截取子集:", len(df))

    # 加载 Marker 图
    print("加载 Marker 图像（可能稍微有点慢）...")
    tqdm.pandas()
    df["Marker"] = df.progress_apply(
        lambda e: simulate.load_tag(e.Size, e.MarkerType, e.Id, e.Angle,
                                    MARKER_PATH=MARKER_PATH),
        axis=1,
    )
    df.reset_index(drop=True, inplace=True)
    print("Marker 生成完成。")

    # 划分 train / test
    df["Datatype"] = ""
    np.random.seed(420)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    train_ratio = 0.7
    train_len = int(np.ceil(len(indices) * train_ratio))
    train_ids = indices[:train_len]
    test_ids = indices[train_len:]
    df.loc[train_ids, "Datatype"] = "train"
    df.loc[test_ids, "Datatype"] = "test"
    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # 数据集
    batch_size = 64
    cap_res = (32, 32, 1)
    temp_res = (128, 128, 1)

    gen_train = DataGenerator(df[df.Datatype == "train"],
                              batch_size=batch_size,
                              cap_res=cap_res, temp_res=temp_res,
                              shuffle=True)
    train_dataset = tf.data.Dataset.from_generator(
        gen_train,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 32, 32, 1), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    # 模型
    generator = build_generator(template_shape=temp_res)
    discriminator = build_discriminator(cap_shape=cap_res, template_shape=temp_res)

    gen_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    steps_per_epoch = gen_train.n_batches
    print("每个 epoch 步数:", steps_per_epoch)

    # 为了先跑通，epoch 不要太大
    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        start = time.time()
        train_iter = iter(train_dataset)

        avg_gen_loss = []
        avg_disc_loss = []

        for step in range(steps_per_epoch):
            try:
                inp, targ = next(train_iter)
            except StopIteration:
                print("  训练数据用完")
                break

            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
                generator, discriminator, gen_opt, disc_opt,
                inp, targ
            )
            avg_gen_loss.append(float(gen_total_loss))
            avg_disc_loss.append(float(disc_loss))

            if step % 50 == 0:
                print(f"  step {step}/{steps_per_epoch}  Gen: {float(gen_total_loss):.4f}  Disc: {float(disc_loss):.4f}")

        if avg_gen_loss:
            print("  Epoch 平均 Generator loss:", np.mean(avg_gen_loss))
            print("  Epoch 平均 Discriminator loss:", np.mean(avg_disc_loss))

        elapsed = time.time() - start
        print(f"Epoch {epoch+1} 用时 {elapsed:.1f} 秒")

    # 训练完保存生成器权重
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "generator_weights_latest.h5")
    generator.save_weights(weights_path)
    print("生成器权重已保存:", weights_path)

    # 同时把 df 存一下，方便评估脚本直接用
    df_path = os.path.join(models_dir, "df_used.pkl")
    df.to_pickle(df_path)
    print("训练用的 df 已保存:", df_path)


if __name__ == "__main__":
    main()
