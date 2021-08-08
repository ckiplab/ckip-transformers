#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
This module implements the utilities for CKIP Transformers NLP drivers.
"""

__author__ = "Mu Yang <http://muyang.pro>"
__copyright__ = "2020 CKIP Lab"
__license__ = "GPL-3.0"


from abc import (
    ABCMeta,
    abstractmethod,
)

from typing import (
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from transformers import (
    AutoModelForTokenClassification,
    BatchEncoding,
    BertTokenizerFast,
)

################################################################################################################################


class CkipTokenClassification(metaclass=ABCMeta):
    """The base class for token classification task.

    Parameters
    ----------
        model_name : ``str``
            The pretrained model name (e.g. ``'ckiplab/bert-base-chinese-ws'``).
        tokenizer_name : ``str``, *optional*, defaults to **model_name**
            The pretrained tokenizer name (e.g. ``'bert-base-chinese'``).
        device : ``int``, *optional*, defaults to -1,
            Device ordinal for CPU/GPU supports.
            Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        *,
        device: int = -1,
    ):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or model_name)

        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")  # pylint: disable=no-member
        self.model.to(self.device)

    ########################################################################################################################

    @classmethod
    @abstractmethod
    def _model_names(cls):
        return NotImplemented  # pragma: no cover

    def _get_model_name_from_level(
        self,
        level: int,
    ):
        try:
            model_name = self._model_names[level]
        except KeyError as exc:
            raise KeyError(f"Invalid level {level}") from exc

        return model_name

    ########################################################################################################################

    def __call__(
        self,
        input_text: Union[List[str], List[List[str]]],
        *,
        use_delim: bool = False,
        delim_set: Optional[str] = "，,。：:；;！!？?",
        batch_size: int = 256,
        max_length: Optional[int] = None,
        show_progress: bool = True,
        pin_memory: bool = True,
    ):
        """Call the driver.

        Parameters
        ----------
            input_text : ``List[str]`` or ``List[List[str]]``
                The input sentences. Each sentence is a string or a list of string.
            use_delim : ``bool``, *optional*, defaults to False
                Segment sentence (internally) using ``delim_set``.
            delim_set : `str`, *optional*, defaults to ``'，,。：:；;！!？?'``
                Used for sentence segmentation if ``use_delim=True``.
            batch_size : ``int``, *optional*, defaults to 256
                The size of mini-batch.
            max_length : ``int``, *optional*
                The maximum length of the sentence,
                must not longer then the maximum sequence length for this model (i.e. ``tokenizer.model_max_length``).
            show_progress : ``bool``, *optional*, defaults to True
                Show progress bar.
            pin_memory : ``bool``, *optional*, defaults to True
                Pin memory in order to accelerate the speed of data transfer to the GPU. This option is
                incompatible with multiprocessing.
        """

        model_max_length = self.tokenizer.model_max_length - 2  # Add [CLS] and [SEP]
        if max_length:
            assert max_length < model_max_length, (
                "Sequence length is longer than the maximum sequence length for this model "
                f"({max_length} > {model_max_length})."
            )
        else:
            max_length = model_max_length

        # Apply delimiter cut
        delim_index = self._find_delim(
            input_text=input_text,
            use_delim=use_delim,
            delim_set=delim_set,
        )

        # Get worded input IDs
        if show_progress:
            input_text = tqdm(input_text, desc="Tokenization")

        input_ids_worded = [
            [self.tokenizer.convert_tokens_to_ids(list(input_word)) for input_word in input_sent] for input_sent in input_text
        ]

        # Flatten input IDs
        (input_ids, index_map,) = self._flatten_input_ids(
            input_ids_worded=input_ids_worded,
            max_length=max_length,
            delim_index=delim_index,
        )

        # Pad and segment input IDs
        (input_ids, attention_mask,) = self._pad_input_ids(
            input_ids=input_ids,
        )

        # Convert input format
        encoded_input = BatchEncoding(
            data=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
            tensor_type="pt",
        )

        # Create dataset
        dataset = TensorDataset(*encoded_input.values())
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )
        if show_progress:
            dataloader = tqdm(dataloader, desc="Inference")

        # Call Model
        logits = []
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(tensor.to(self.device) for tensor in batch)
                (batch_logits,) = self.model(**dict(zip(encoded_input.keys(), batch)), return_dict=False)
                batch_logits = batch_logits.cpu().numpy()[:, 1:, :]  # Remove [CLS]
                logits.append(batch_logits)

        # Call model
        logits = np.concatenate(logits, axis=0)

        return logits, index_map

    @staticmethod
    def _find_delim(
        *,
        input_text,
        use_delim,
        delim_set,
    ):
        if not use_delim:
            return set()

        delim_index = set()
        delim_set = set(delim_set)
        for sent_idx, input_sent in enumerate(input_text):
            for word_idx, input_word in enumerate(input_sent):
                if input_word in delim_set:
                    delim_index.add((sent_idx, word_idx))
        return delim_index

    @staticmethod
    def _flatten_input_ids(
        *,
        input_ids_worded,
        max_length,
        delim_index,
    ):
        input_ids = []
        index_map = []

        input_ids_sent = []
        index_map_sent = []

        for sent_idx, input_ids_worded_sent in enumerate(input_ids_worded):
            for word_idx, word_ids in enumerate(input_ids_worded_sent):
                word_length = len(word_ids)

                if word_length == 0:
                    index_map_sent.append(None)
                    continue

                # Check if sentence segmentation is needed
                if len(input_ids_sent) + word_length > max_length:
                    input_ids.append(input_ids_sent)
                    input_ids_sent = []

                # Insert tokens
                index_map_sent.append(
                    (
                        len(input_ids),  # line index
                        len(input_ids_sent),  # token index
                    )
                )
                input_ids_sent += word_ids

                if (sent_idx, word_idx) in delim_index:
                    input_ids.append(input_ids_sent)
                    input_ids_sent = []

            # End of a sentence
            if input_ids_sent:
                input_ids.append(input_ids_sent)
                input_ids_sent = []
            index_map.append(index_map_sent)
            index_map_sent = []

        return input_ids, index_map

    def _pad_input_ids(
        self,
        *,
        input_ids,
    ):
        max_length = max(map(len, input_ids))

        padded_input_ids = []
        attention_mask = []
        for input_ids_sent in input_ids:
            token_count = len(input_ids_sent)
            pad_count = max_length - token_count
            padded_input_ids.append(
                [self.tokenizer.cls_token_id]
                + input_ids_sent
                + [self.tokenizer.sep_token_id]
                + [self.tokenizer.pad_token_id] * pad_count
            )
            attention_mask.append([1] * (token_count + 2) + [0] * pad_count)  # [CLS] & input & [SEP]  # [PAD]s
        return padded_input_ids, attention_mask


################################################################################################################################


class NerToken(NamedTuple):
    """A named-entity recognition token."""

    word: str  #: ``str``, the token word.
    ner: str  #: ``str``, the NER-tag.
    idx: Tuple[int, int]  #: ``Tuple[int, int]``, the starting / ending index in the sentence.
