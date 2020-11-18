#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This module implements the utilities for CKIP Transformers NLP drivers.
"""

__author__ = 'Mu Yang <http://muyang.pro>'
__copyright__ = '2020 CKIP Lab'
__license__ = 'GPL-3.0'

from typing import (
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch

from transformers import (
    AutoModelForTokenClassification,
    BatchEncoding,
    BertTokenizerFast,
)

################################################################################################################################

class CkipTokenClassification:
    """The base class for token classification task.

        Parameters
        ----------
            model_name : ``str``
                The pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ws'``).
            tokenizer_name : ``str``, *optional*, defaults to **model_name**
                The pretrained tokenizer name (e.g. ``'bert-base-chinese'``).
    """

    def __init__(self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
    ):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or model_name)

    def __call__(self,
        input_text: Union[List[str], List[List[str]]],
        *,
        max_length: Optional[int] = None,
    ):
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]`` or ``List[List[str]]``
                The input sentences. Each sentence is a string or a list of string.
            max_length : ``int``
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
        """

        model_max_length = self.tokenizer.model_max_length - 2  # Add [CLS] and [SEP]
        if max_length:
            assert max_length < model_max_length, \
                'Sequence length is longer than the maximum sequence length for this model ' \
               f'({max_length} > {model_max_length}).'
        else:
            max_length = model_max_length

        # Get worded input IDs
        input_ids_worded = [
            [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_word)) for input_word in input_sent
            ] for input_sent in input_text
        ]

        # Flatten input IDs
        (
            input_ids,
            index_map,
        ) = self._flatten_input_ids(
            input_ids_worded=input_ids_worded,
            max_length=max_length,
        )

        # Pad and segment input IDs
        (
            input_ids,
            attention_mask,
        ) = self._pad_input_ids(
            input_ids=input_ids,
        )

        # Convert to tensors
        batch = BatchEncoding(
            data=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
            tensor_type='pt',
        )

        # Call Model
        with torch.no_grad():
            (
                loss,
            ) = self.model(**batch)
            loss = loss.cpu().numpy()[:, 1:, :]  # Remove [CLS]

        return loss, index_map

    @staticmethod
    def _flatten_input_ids(*,
        input_ids_worded,
        max_length,
    ):
        input_ids = []
        index_map = []

        input_ids_sent = []
        index_map_sent = []

        for input_ids_worded_sent in input_ids_worded:
            for word_ids in input_ids_worded_sent:
                word_length = len(word_ids)

                if word_length == 0:
                    index_map_sent.append(None)
                    continue

                # Check if sentence segmentation is needed
                if len(input_ids_sent) + word_length > max_length:
                    input_ids.append(input_ids_sent)
                    input_ids_sent = []

                # Insert tokens
                index_map_sent.append((
                    len(input_ids),  # line index
                    len(input_ids_sent),   # token index
                ))
                input_ids_sent += word_ids

            # End of a sentence
            if input_ids_sent:
                input_ids.append(input_ids_sent)
                input_ids_sent = []
            index_map.append(index_map_sent)
            index_map_sent = []

        return input_ids, index_map

    def _pad_input_ids(self, *,
        input_ids,
    ):
        max_length = max(map(len, input_ids))

        padded_input_ids = []
        attention_mask = []
        for input_ids_sent in input_ids:
            token_count = len(input_ids_sent)
            pad_count = max_length - token_count
            padded_input_ids.append(
                [self.tokenizer.cls_token_id] +
                input_ids_sent +
                [self.tokenizer.sep_token_id] +
                [self.tokenizer.pad_token_id] * pad_count
            )
            attention_mask.append(
                [1] * (token_count+2) +  # [CLS] & input & [SEP]
                [0] * pad_count          # [PAD]s
            )
        return padded_input_ids, attention_mask

################################################################################################################################

class NerToken(NamedTuple):
    """A named-entity recognition token."""
    word: str             #: ``str``, the token word.
    ner: str              #: ``str``, the NER-tag.
    idx: Tuple[int, int]  #: ``Tuple[int, int]``, the starting / ending index in the sentence.
