#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This module implements the CKIP Transformers NLP drivers.
"""

__author__ = 'Mu Yang <http://muyang.pro>'
__copyright__ = '2020 CKIP Lab'
__license__ = 'GPL-3.0'

from typing import (
    List,
    Optional,
)

import numpy as np

from .util import (
    CkipTokenClassification,
    NerToken,
)

################################################################################################################################

class CkipWordSegmenter(CkipTokenClassification):
    """The word segmentation driver.

        Parameters
        ----------
            model_name : ``str``, *optional*, defaults to ``'ckiplab/bert-base-chinese-ws'``
                The pretrained model name.
            tokenizer_name : ``str``, *optional*, defaults to **model_name**
                The pretrained tokenizer name.
    """

    def __init__(self,
        model_name: Optional[str] = 'ckiplab/bert-base-chinese-ws',
        tokenizer_name: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name)

    def __call__(self,
        input_text: List[str],
        *,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).

        Returns
        -------
            ``List[List[NerToken]]``
                A list of list of words (``str``).
        """

        # Call model
        (
            loss,
            index_map,
        ) = super().__call__(input_text, max_length=max_length)

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            word = ''
            for input_char, loss_index in zip(*sent_data):
                if loss_index is None:
                    if word:
                        output_sent.append(word)
                    output_sent.append(input_char)
                    word = ''
                else:
                    loss_b, loss_i = loss[loss_index]

                    if loss_b > loss_i:
                        if word:
                            output_sent.append(word)
                        word = input_char
                    else:
                        word += input_char

            if word:
                output_sent.append(word)
            output_text.append(output_sent)

        return output_text

################################################################################################################################

class CkipPosTagger(CkipTokenClassification):
    """The part-of-speech tagging driver.

        Parameters
        ----------
            model_name : ``str``, *optional*, defaults to ``'ckiplab/bert-base-chinese-pos'``
                The pretrained model name.
            tokenizer_name : ``str``, *optional*, defaults to **model_name**
                The pretrained tokenizer name.
    """

    def __init__(self,
        model_name: Optional[str] = 'ckiplab/bert-base-chinese-pos',
        tokenizer_name: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name)

    def __call__(self,
        input_text: List[List[str]],
        *,
        max_length: Optional[int] = None,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[List[str]]``
                The input sentences. Each sentence is a list of strings (words).
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).

        Returns
        -------
            ``List[List[NerToken]]``
                A list of list of POS tags (``str``).
        """

        # Call model
        (
            loss,
            index_map,
        ) = super().__call__(input_text, max_length=max_length)

        # Get labels
        id2label = self.model.config.id2label

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            for _, loss_index in zip(*sent_data):
                if loss_index is None:
                    label = 'WHITESPACE'
                else:
                    label = id2label[np.argmax(loss[loss_index])]
                output_sent.append(label)
            output_text.append(output_sent)

        return output_text

################################################################################################################################

class CkipNerChunker(CkipTokenClassification):
    """The named-entity recognition driver.

        Parameters
        ----------
            model_name : ``str``, *optional*, defaults to ``'ckiplab/bert-base-chinese-ner'``
                The pretrained model name.
            tokenizer_name : ``str``, *optional*, defaults to **model_name**
                The pretrained tokenizer name.
    """

    def __init__(self,
        model_name: Optional[str] = 'ckiplab/bert-base-chinese-ner',
        tokenizer_name: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name)

    def __call__(self,
        input_text: List[str],
        *,
        max_length: Optional[int] = None,
    ) -> List[List[NerToken]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).

        Returns
        -------
            ``List[List[NerToken]]``
                A list of list of entities (:class:`~.util.NerToken`).
        """

        # Call model
        (
            loss,
            index_map,
        ) = super().__call__(input_text, max_length=max_length)

        # Get labels
        id2label = self.model.config.id2label

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            entity_word = None
            entity_ner = None
            entity_idx0 = None
            for index_char, (input_char, loss_index,) in enumerate(zip(*sent_data)):
                if loss_index is None:
                    label = 'O'
                else:
                    label = id2label[np.argmax(loss[loss_index])]

                if label == 'O':
                    entity_ner = None
                    continue

                bioes, ner = label.split('-')

                if bioes == 'S':
                    output_sent.append(NerToken(
                        word = input_char,
                        ner  = ner,
                        idx  = (index_char, index_char+len(input_char),),
                    ))
                    entity_ner = None
                elif bioes == 'B':
                    entity_word = input_char
                    entity_ner = ner
                    entity_idx0 = index_char
                elif bioes == 'I':
                    if entity_ner == ner:
                        entity_word += input_char
                    else:
                        entity_ner = None
                elif bioes == 'E':
                    if entity_ner == ner:
                        entity_word += input_char
                    output_sent.append(NerToken(
                        word = entity_word,
                        ner  = entity_ner,
                        idx  = (entity_idx0, index_char+len(input_char),),
                    ))
                    entity_ner = None

            output_text.append(output_sent)

        return output_text
