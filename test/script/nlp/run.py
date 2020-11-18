#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mu Yang <http://muyang.pro>'
__copyright__ = '2020 CKIP Lab'
__license__ = 'GPL-3.0'

from _base import *

################################################################################################################################

def test_word_segmenter():
    nlp = CkipWordSegmenter()
    output_ws = nlp(text)
    assert output_ws == ws

################################################################################################################################

def test_pos_tagger():
    nlp = CkipPosTagger()
    output_pos = nlp(ws)
    assert output_pos == pos

################################################################################################################################

def test_ner_chunker():
    nlp = CkipNerChunker()
    output_ner = nlp(text)
    output_ner = [ [ tuple(entity) for entity in sent ] for sent in output_ner ]
    assert output_ner == ner
