#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-29 04:50:57 -0800
@LastEditTime: 2020-02-29 04:50:57 -0800
@FilePath: /bing/bing/log/log.py
@Description: file content
"""
from datetime import datetime
import click


def info_header():
    now = datetime.now().strftime("%H:%M:%S")
    return click.style(f"[{now}] INFO - ", bold=True, fg="green")


class MSG(object):
    def __init__(self, text, color):
        self.text = None
        self.color = "bright_blue"

    def info(self):
        return click.echo(
            info_header() + "  " + click.style(str(self.text), bold=False, fg=self.color)
        )
