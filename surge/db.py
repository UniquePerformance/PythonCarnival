#!/usr/bin/env python
# coding=utf-8
"""
@Author: Bing@HOT.AS.ICE
@LastEditors: Bing@HOT.AS.ICE
@Date: 2020-02-29 04:47:25 -0800
@LastEditTime: 2020-02-29 04:47:25 -0800
@FilePath: /surge/db.py
@Description: file content
"""
import sqlite3, os
from bing.log import MSG
from bing.db import db

msg = MSG(color="bright_blue", bold=False)

script_path = os.path.dirname(os.path.abspath(__file__))
db_file = f"{script_path}/data/Database.db"
# msg.warn(db_file)
paper = db(path=db_file, auto_commit=False, reconnects=5, auto_connect=True)
"""
create table of saving report monthly 
"""
paper.create_table(
    "dhfd",
    ["fdl", "gdl", "grl", "fdmh", "gdmh", "zhcydl", "fdcydl", "fhl", "fdsh"],
    ["REAL", "REAL", "REAL"],
)
# Commit changes
paper.commit()
# Rollback
paper.rollback()
# Close the connection
paper.close()
