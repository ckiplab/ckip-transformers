#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mu Yang <http://muyang.pro>'
__copyright__ = '2020 CKIP Lab'
__license__ = 'GPL-3.0'

from _base import *

################################################################################################################################

def test_word_segmenter():
    nlp = CkipWordSegmenter(level=1)
    output_ws = nlp(text, show_progress=False)
    assert output_ws == ws

################################################################################################################################

def test_pos_tagger():
    nlp = CkipPosTagger(level=1)
    output_pos = nlp(ws, show_progress=False)
    assert output_pos == pos

################################################################################################################################

def test_ner_chunker():
    nlp = CkipNerChunker(level=1)
    output_ner = nlp(text, show_progress=False)
    output_ner = [ [ tuple(entity) for entity in sent ] for sent in output_ner ]
    assert output_ner == ner
