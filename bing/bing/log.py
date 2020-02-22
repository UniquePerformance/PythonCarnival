#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-22 05:32:39 -0800
@LastEditTime: 2020-02-22 05:32:39 -0800
@FilePath: /bing/bing/log.py
@Description: file content
"""
import click
from prompt_toolkit import print_formatted_text, prompt, HTML
from prompt_toolkit.styles import Style
from datetime import datetime


def log(text=None, type="default"):

    style = Style.from_dict({"red": "#FF0000", "salmon": "#FA8072", "yellow": "#FFFF00"})

    if type == "default":
        return print_formatted_text(text, style=style)
    elif type == "Warning":
        content = f"<red>{text}</red>"
        return print_formatted_text(HTML(content), style=style)
