#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-29 01:41:09 -0800
@LastEditTime: 2020-02-29 01:41:09 -0800
@FilePath: /bing/bing/plot.py
@Description: file content
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import os
import numpy as np

if not plt.isinteractive():
    print(
        "\nMatplotlib interactive mode is currently OFF. It is "
        "recommended to use a suitable matplotlib backend and turn it "
        "on by calling matplotlib.pyplot.ion()\n"
    )


class PLOT(object):
    def __init__(self, name, age):
        self._CHN_FONT_ = None
        self._FONT_PROP_ = None
        self.font = None


# set_font("./fonts/STFANGSO.TTF")  # for displaying Chinese characters in plots
