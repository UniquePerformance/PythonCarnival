#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   nox.py
@Time    :   2020/02/22 23:10:14
@Author  :   MegaWatt 
@Version :   1.0
@Contact :   icesharp@gmail.com
@License :   (C)Copyright 2019-2020
@Desc    :   None
"""

# here put the import lib
import os

import openpyxl
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from openpyxl import Workbook, load_workbook
from bing.bing import debug_info
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.keras.backend.clear_session()  # For easy reset of notebook state.

data_file = "data.xlsx"
current_dir = os.path.abspath(os.path.dirname(__file__)) + "/"

dat_df = pd.read_excel(current_dir + data_file, sheet_name="Sheet1")
dat_df = dat_df.iloc[:, 0:32]
dataset = dat_df

intput_dat = dat_df.iloc[:, 0:31]
output_dat = dat_df.iloc[:, 31:32]

for items in intput_dat.isna().sum().index:
    if intput_dat.isna().sum()[items] == 0:
        pass
    else:
        print(f"index:{items},Value:{intput_dat.isna().sum()[items]}")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
debug_info("sns plot")
sns.pairplot(
    train_dataset[["fuhe", "coal31", "coal32", "coal33", "coal34", "coal35", "O2", "ZFL", "NOX"]],
    diag_kind="kde",
    # hue="NOX",
)
# plt.show()
debug_info("save plot")
plt.savefig("image02.png")  # 保存图片

print(intput_dat.shape)
model = tf.keras.Sequential(
    [
        # 向模型添加一个64单元的密集连接层：
        tf.keras.layers.Dense(3000, activation="relu", input_shape=(31,)),
        # 加上另一个:
        tf.keras.layers.Dense(3000, activation="relu"),
        tf.keras.layers.Dense(3000, activation="relu"),
        # 添加具有10个输出单位的softmax层:
        tf.keras.layers.Dense(1, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"]  # 均方误差
)  # 平均绝对误差

model.fit(intput_dat, output_dat, epochs=100, batch_size=32)
