#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-22 04:49:26 -0800
@LastEditTime: 2020-02-22 04:49:26 -0800
@FilePath: /bing/xlsx/xlsx.py
@Description: file content
"""
from datetime import datetime

import click
import openpyxl
import xlsxwriter
from openpyxl.utils.cell import column_index_from_string, coordinate_from_string


def info_header():
    now = datetime.now().strftime("%H:%M:%S")
    return click.style(f"[{now}] INFO - ", bold=True, fg="green")


def debug_info(text=None, color="bright_blue"):
    return click.echo(info_header() + "  " + click.style(str(text), bold=False, fg=color))


def get_xy(cor):
    """
    cor:"A1"

    return: col,row,
    """
    xy = coordinate_from_string(cor)
    col = column_index_from_string(xy[0])
    return col, xy[1]


def save_to_excel(df=None, path=None, sheet_name="default"):

    while True:
        try:
            df.to_excel(path, sheet_name=sheet_name)
        except xlsxwriter.exceptions.FileCreateError as e:
            # For Python 3 use input() instead of raw_input().
            decision = input(
                "Exception caught in to_excel(): %s\n"
                "Please close the file if it is open in Excel.\n"
                "Try to write file again? [Y/n]: " % e
            )
            if decision != "n":
                continue
        break
