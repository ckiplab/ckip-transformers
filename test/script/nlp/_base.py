#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mu Yang <http://muyang.pro>'
__copyright__ = '2020 CKIP Lab'
__license__ = 'GPL-3.0'

from ckip_transformers.nlp import *

################################################################################################################################

text = [
    '中文字耶，啊哈哈哈。',
    '「完蛋了！」 畢卡索他想',
]
ws = [
    [ '中文字', '耶', '，', '啊', '哈', '哈哈', '。', ],
    [ '「', '完蛋', '了', '！', '」', ' ', '畢卡索', '他', '想', ],
]
pos = [
    [ 'Na', 'T', 'COMMACATEGORY', 'I', 'D', 'D', 'PERIODCATEGORY', ],
    [ 'PARENTHESISCATEGORY', 'VH', 'T', 'EXCLAMATIONCATEGORY', 'PARENTHESISCATEGORY', 'WHITESPACE', 'Nb', 'Nh', 'VE', ],
]
ner = [
    [ ( '中文字', 'LANGUAGE', (0, 3), ), ],
    [ ( '畢卡索', 'PERSON', (7, 10), ), ],
]
