CKIP Transformers
-----------------

This open-source library implements CKIP Chinese NLP tools using transformers models.

* (WS) Word Segmentation
* (POS) Part-of-Speech Tagging
* (NER) Named Entity Recognition

Git
^^^

https://github.com/emfomy/ckip-transformers

|GitHub Version| |GitHub License| |GitHub Release| |GitHub Issues|

.. |GitHub Version| image:: https://img.shields.io/github/v/release/emfomy/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/emfomy/ckip-transformers/releases

.. |GitHub License| image:: https://img.shields.io/github/license/emfomy/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/emfomy/ckip-transformers/blob/master/LICENSE

.. |GitHub Release| image:: https://img.shields.io/github/release-date/emfomy/ckip-transformers.svg?cacheSeconds=3600

.. |GitHub Downloads| image:: https://img.shields.io/github/downloads/emfomy/ckip-transformers/total.svg?cacheSeconds=3600
   :target: https://github.com/emfomy/ckip-transformers/releases/latest

.. |GitHub Issues| image:: https://img.shields.io/github/issues/emfomy/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/emfomy/ckip-transformers/issues

.. |GitHub Forks| image:: https://img.shields.io/github/forks/emfomy/ckip-transformers.svg?style=social&label=Fork&cacheSeconds=3600

.. |GitHub Stars| image:: https://img.shields.io/github/stars/emfomy/ckip-transformers.svg?style=social&label=Star&cacheSeconds=3600

.. |GitHub Watchers| image:: https://img.shields.io/github/watchers/emfomy/ckip-transformers.svg?style=social&label=Watch&cacheSeconds=3600

PyPI
^^^^

https://pypi.org/project/ckip-transformers

|PyPI Version| |PyPI License| |PyPI Downloads| |PyPI Python| |PyPI Implementation| |PyPI Format| |PyPI Status|

.. |PyPI Version| image:: https://img.shields.io/pypi/v/ckip-transformers.svg?cacheSeconds=3600
   :target: https://pypi.org/project/ckip-transformers

.. |PyPI License| image:: https://img.shields.io/pypi/l/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/emfomy/ckip-transformers/blob/master/LICENSE

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/ckip-transformers.svg?cacheSeconds=3600
   :target: https://pypi.org/project/ckip-transformers#files

.. |PyPI Python| image:: https://img.shields.io/pypi/pyversions/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Implementation| image:: https://img.shields.io/pypi/implementation/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Format| image:: https://img.shields.io/pypi/format/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Status| image:: https://img.shields.io/pypi/status/ckip-transformers.svg?cacheSeconds=3600

Documentation
^^^^^^^^^^^^^

https://ckip-transformers.readthedocs.io/

|ReadTheDocs Home|

.. |ReadTheDocs Home| image:: https://img.shields.io/website/https/ckip-transformers.readthedocs.io.svg?cacheSeconds=3600&up_message=online&down_message=offline
   :target: https://ckip-transformers.readthedocs.io

Relative Demos / Packages
^^^^^^^^^^^^^^^^^^^^^^^^^

- `CkipTagger <https://github.com/ckiplab/ckiptagger>`_: An alternative Chinese NLP library with using BiLSTM.
- `CKIP CoreNLP Toolkit <https://github.com/ckiplab/ckipnlp>`_: A Chinese NLP library with more NLP tasks and utilities.

Contributers
^^^^^^^^^^^^

* `Mu Yang <https://muyang.pro>`__ at `CKIP <https://ckip.iis.sinica.edu.tw>`__ (Author & Maintainer)
* `Wei-Yun Ma <https://www.iis.sinica.edu.tw/pages/ma/>`__ at `CKIP <https://ckip.iis.sinica.edu.tw>`__ (Maintainer)

Performance
^^^^^^^^^^^

================================  =======  =========  ========
Tool                              WS (F1)  POS (Acc)  NER (F1)
================================  =======  =========  ========
Ckip Transformers (level 3†)      97.60%   95.67%     81.18%
Ckip Transformers (level 2†)      97.33%   95.30%     79.47%
Ckip Transformers (level 1†)      96.66%   94.48%     71.17%
CkipTagger‡                       97.33%   94.59%     77.87%
Monpa§                            92.58%   --         --
Jeiba‖                            81.18%   --         --
================================  =======  =========  ========

Installation
------------

``pip install -U ckip-transformers``

Requirements:

* `Python <https://www.python.org>`__ 3.6+
* `PyTorch <https://pytorch.org>`__ 1.1+
* `HuggingFace Transformers <https://huggingface.co/transformers/>`__ 3.5+

Usage
-----

See https://ckip-transformers.readthedocs.io/en/latest/_api/ckip_transformers.html for API details.

The complete script of this example is https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py.

1. Import module
^^^^^^^^^^^^^^^^

.. code-block:: python

   from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

2. Load models
^^^^^^^^^^^^^^

.. code-block:: python

   # Initialize drivers
   ws_driver  = CkipWordSegmenter()
   pos_driver = CkipPosTagger()
   ner_driver = CkipNerChunker()

3. Run pipeline
^^^^^^^^^^^^^^^

- The input for word segmentation and named-entity recognition must be a list of sentences.
- The input for part-of-speech tagging must be a list of list of words (the output of word segmentation).

.. code-block:: python

   # Input text
   text = [
      '傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。',
      '美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。',
   ]

   # Run pipeline
   ws  = ws_driver(text)
   pos = pos_driver(ws)
   ner = ner_driver(text)

4. Show results
^^^^^^^^^^^^^^^

.. code-block:: python

   # Pack word segmentation and part-of-speech results
   def pack_ws_pos_sentece(sentence_ws, sentence_pos):
      assert len(sentence_ws) == len(sentence_pos)
      res = []
      for word_ws, word_pos in zip(sentence_ws, sentence_pos):
         res.append(f'{word_ws}({word_pos})')
      return '\u3000'.join(res)

   # Show results
   for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
      print(sentence)
      print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
      for entity in sentence_ner:
         print(entity)
      print()

.. code-block:: text

   傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。
   傅達仁(Nb)　今(Nd)　將(D)　執行(VC)　安樂死(Na)　，(COMMACATEGORY)　卻(D)　突然(D)　爆出(VJ)　自己(Nh)　20(Neu)　年(Nf)　前(Ng)　遭(P)　緯來(Nb)　體育台(Na)　封殺(VC)　，(COMMACATEGORY)　他(Nh)　不(D)　懂(VK)　自己(Nh)　哪裡(Ncd)　得罪到(VC)　電視台(Nc)　。(PERIODCATEGORY)
   NerToken(word='傅達仁', ner='PERSON', idx=(0, 3))
   NerToken(word='20年', ner='DATE', idx=(18, 21))
   NerToken(word='緯來體育台', ner='ORG', idx=(23, 28))

   美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。
   美國(Nc)　參議院(Nc)　針對(P)　今天(Nd)　總統(Na)　布什(Nb)　所(D)　提名(VC)　的(DE)　勞工部長(Na)　趙小蘭(Nb)　展開(VC)　認可(VC)　聽證會(Na)　，(COMMACATEGORY)　預料(VE)　她(Nh)　將(D)　會(D)　很(Dfa)　順利(VH)　通過(VC)　參議院(Nc)　支持(VC)　，(COMMACATEGORY)　成為(VG)　該(Nes)　國(Nc)　有史以來(D)　第一(Neu)　位(Nf)　的(DE)　華裔(Na)　女性(Na)　內閣(Na)　成員(Na)　。(PERIODCATEGORY)
   NerToken(word='美國參議院', ner='ORG', idx=(0, 5))
   NerToken(word='今天', ner='LOC', idx=(7, 9))
   NerToken(word='布什', ner='PERSON', idx=(11, 13))
   NerToken(word='勞工部長', ner='ORG', idx=(17, 21))
   NerToken(word='趙小蘭', ner='PERSON', idx=(21, 24))
   NerToken(word='認可聽證會', ner='EVENT', idx=(26, 31))
   NerToken(word='參議院', ner='ORG', idx=(42, 45))
   NerToken(word='第一', ner='ORDINAL', idx=(56, 58))
   NerToken(word='華裔', ner='NORP', idx=(60, 62))

Pretrained Models
-----------------

One may also use our pretrained models with HuggingFace transformers library directly: https://huggingface.co/ckiplab/.

Pretrained Language Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

* `ALBERT Tiny <https://huggingface.co/ckiplab/albert-tiny-chinese>`_
* `ALBERT Base <https://huggingface.co/ckiplab/albert-base-chinese>`_
* `BERT Base <https://huggingface.co/ckiplab/bert-base-chinese>`_
* `GPT2 Base <https://huggingface.co/ckiplab/gpt2-base-chinese>`_

NLP Task Models
^^^^^^^^^^^^^^^

* `ALBERT Tiny — Word Segmentation <https://huggingface.co/ckiplab/albert-tiny-chinese-ws>`_
* `ALBERT Tiny — Part-of-Speech Tagging <https://huggingface.co/ckiplab/albert-tiny-chinese-pos>`_
* `ALBERT Tiny — Named-Entity Recognition <https://huggingface.co/ckiplab/albert-tiny-chinese-ner>`_

* `ALBERT Base — Word Segmentation <https://huggingface.co/ckiplab/albert-base-chinese-ws>`_
* `ALBERT Base — Part-of-Speech Tagging <https://huggingface.co/ckiplab/albert-base-chinese-pos>`_
* `ALBERT Base — Named-Entity Recognition <https://huggingface.co/ckiplab/albert-base-chinese-ner>`_

* `BERT Base — Word Segmentation <https://huggingface.co/ckiplab/bert-base-chinese-ws>`_
* `BERT Base — Part-of-Speech Tagging <https://huggingface.co/ckiplab/bert-base-chinese-pos>`_
* `BERT Base — Named-Entity Recognition <https://huggingface.co/ckiplab/bert-base-chinese-ner>`_

License
-------

|GPL-3.0|

Copyright (c) 2020 `CKIP Lab <https://ckip.iis.sinica.edu.tw>`__ under the `GPL-3.0 License <https://www.gnu.org/licenses/gpl-3.0.html>`__.

.. |GPL-3.0| image:: https://www.gnu.org/graphics/gplv3-with-text-136x68.png
   :target: https://www.gnu.org/licenses/gpl-3.0.html
