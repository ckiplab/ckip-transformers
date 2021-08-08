#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This module implements the CKIP Transformers NLP drivers.
"""

__author__ = "Mu Yang <http://muyang.pro>"
__copyright__ = "2020 CKIP Lab"
__license__ = "GPL-3.0"

from typing import (
    List,
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
        level : ``str`` *optional*, defaults to 3, must be 1—3
            The model level. The higher the level is, the more accurate and slower the model is.
        model_name : ``str`` *optional*, overwrites **level**
            The pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ws'``).
        device : ``int``, *optional*, defaults to -1,
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        1: "ckiplab/albert-tiny-chinese-ws",
        2: "ckiplab/albert-base-chinese-ws",
        3: "ckiplab/bert-base-chinese-ws",
    }

    def __init__(
        self,
        level: int = 3,
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name_from_level(level))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[str],
        *,
        use_delim: bool = False,
        **kwargs,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string.
            use_delim : ``bool``, *optional*, defaults to False
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is
                incompatible with multiprocessing.

        Returns
        -------
            ``List[List[str]]``
                A list of list of words (``str``).
        """

        # Call model
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            word = ""
            for input_char, logits_index in zip(*sent_data):
                if logits_index is None:
                    if word:
                        output_sent.append(word)
                    output_sent.append(input_char)
                    word = ""
                else:
                    logits_b, logits_i = logits[logits_index]

                    if logits_b > logits_i:
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
        level : ``str`` *optional*, defaults to 3, must be 1—3
            The model level. The higher the level is, the more accurate and slower the model is.
        model_name : ``str`` *optional*, overwrites **level**
            The pretrained model name (e.g. ``'ckiplab/bert-base-chinese-pos'``).
        device : ``int``, *optional*, defaults to -1,
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        1: "ckiplab/albert-tiny-chinese-pos",
        2: "ckiplab/albert-base-chinese-pos",
        3: "ckiplab/bert-base-chinese-pos",
    }

    def __init__(
        self,
        level: int = 3,
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name_from_level(level))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[List[str]],
        *,
        use_delim: bool = True,
        **kwargs,
    ) -> List[List[str]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[List[str]]``
                The input sentences. Each sentence is a list of strings (words).
            use_delim : ``bool``, *optional*, defaults to True
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is
                incompatible with multiprocessing.

        Returns
        -------
            ``List[List[str]]``
                A list of list of POS tags (``str``).
        """

        # Call model
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Get labels
        id2label = self.model.config.id2label

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            for input_char, logits_index in zip(*sent_data):
                if logits_index is None or input_char.isspace():
                    label = "WHITESPACE"
                else:
                    label = id2label[np.argmax(logits[logits_index])]
                output_sent.append(label)
            output_text.append(output_sent)

        return output_text


################################################################################################################################


class CkipNerChunker(CkipTokenClassification):
    """The named-entity recognition driver.

    Parameters
    ----------
        level : ``str`` *optional*, defaults to 3, must be 1—3
            The model level. The higher the level is, the more accurate and slower the model is.
        model_name : ``str`` *optional*, overwrites **level**
            The pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ner'``).
        device : ``int``, *optional*, defaults to -1,
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    _model_names = {
        1: "ckiplab/albert-tiny-chinese-ner",
        2: "ckiplab/albert-base-chinese-ner",
        3: "ckiplab/bert-base-chinese-ner",
    }

    def __init__(
        self,
        level: int = 3,
        **kwargs,
    ):
        model_name = kwargs.pop("model_name", self._get_model_name_from_level(level))
        super().__init__(model_name=model_name, **kwargs)

    def __call__(
        self,
        input_text: List[str],
        *,
        use_delim: bool = False,
        **kwargs,
    ) -> List[List[NerToken]]:
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]``
                The input sentences. Each sentence is a string or a list or string (words).
            use_delim : ``bool``, *optional*, defaults to False
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``int``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is
                incompatible with multiprocessing.

        Returns
        -------
            ``List[List[NerToken]]``
                A list of list of entities (:class:`~.util.NerToken`).
        """

        # Call model
        (
            logits,
            index_map,
        ) = super().__call__(input_text, use_delim=use_delim, **kwargs)

        # Get labels
        id2label = self.model.config.id2label

        # Post-process results
        output_text = []
        for sent_data in zip(input_text, index_map):
            output_sent = []
            entity_word = None
            entity_ner = None
            entity_idx0 = None
            for index_char, (
                input_char,
                logits_index,
            ) in enumerate(zip(*sent_data)):
                if logits_index is None:
                    label = "O"
                else:
                    label = id2label[np.argmax(logits[logits_index])]

                if label == "O":
                    entity_ner = None
                    continue

                bioes, ner = label.split("-")

                if bioes == "S":
                    output_sent.append(
                        NerToken(
                            word=input_char,
                            ner=ner,
                            idx=(
                                index_char,
                                index_char + len(input_char),
                            ),
                        )
                    )
                    entity_ner = None
                elif bioes == "B":
                    entity_word = input_char
                    entity_ner = ner
                    entity_idx0 = index_char
                elif bioes == "I":
                    if entity_ner == ner:
                        entity_word += input_char
                    else:
                        entity_ner = None
                elif bioes == "E":
                    if entity_ner == ner:
                        entity_word += input_char
                        output_sent.append(
                            NerToken(
                                word=entity_word,
                                ner=entity_ner,
                                idx=(
                                    entity_idx0,
                                    index_char + len(input_char),
                                ),
                            )
                        )
                    entity_ner = None

            output_text.append(output_sent)

        return output_text
