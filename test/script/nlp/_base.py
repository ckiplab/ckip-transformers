#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = "Mu Yang <http://muyang.pro>"
__copyright__ = "2023 CKIP Lab"
__license__ = "GPL-3.0"

from ckip_transformers.nlp import *

################################################################################################################################
# ckiplab/albert-tiny-chinese-#

# fmt: off
text = [
    '中文字耶，啊哈哈哈。',
    '「完蛋了！」畢卡索 他想',
    '趙、錢、孫、李',
]

ws = [
    [ '中文字', '耶', '，', '啊哈', '哈哈', '。' ],
    [ '「', '完蛋', '了', '！', '」', '畢卡索', ' ', '他', '想' ],
    [ '趙', '、', '錢', '、', '孫', '、', '李' ]
]

pos = [
    [ 'Na', 'T', 'COMMACATEGORY', 'I', 'D', 'PERIODCATEGORY' ],
    [ 'PARENTHESISCATEGORY', 'VH', 'T', 'EXCLAMATIONCATEGORY', 'PARENTHESISCATEGORY', 'Nb', 'WHITESPACE', 'Nh', 'VE' ],
    [ 'Nb', 'PAUSECATEGORY', 'Nb', 'PAUSECATEGORY', 'Nb', 'PAUSECATEGORY', 'Nb' ],
]

ner = [
    [],
    [ ( '畢卡索', 'PERSON', (6, 9), ) ],
    [ ('趙', 'PERSON', (0, 1), ),  ('錢', 'PERSON', (2, 3), ),  ('孫', 'PERSON', (4, 5), ),  ('李', 'PERSON', (6, 7), ) ],
]
# fmt: on
