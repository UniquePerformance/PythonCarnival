#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-29 02:33:03 -0800
@LastEditTime: 2020-02-29 02:33:03 -0800
@FilePath: /play/plot.py
@Description: file content
"""
from bing import PLOT
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# bing.bing.debug_info("gello")
PLOT.set_font("./fonts/STFANGSO.TTF")

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel="time (s)", ylabel="voltage (mV)", title="这样的漏洞一个毁全盘")
ax.grid()

fig.savefig("test.png")
plt.show()
