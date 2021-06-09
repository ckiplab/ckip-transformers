CKIP Transformers
-----------------

| This project provides traditional Chinese transformers models (including ALBERT, BERT, GPT2) and NLP tools (including word segmentation, part-of-speech tagging, named entity recognition).
| 這個專案提供了繁體中文的 transformers 模型（包含 ALBERT、BERT、GPT2）及自然語言處理工具（包含斷詞、詞性標記、實體辨識）。

Git
^^^

| https://github.com/ckiplab/ckip-transformers
| |GitHub Version| |GitHub License| |GitHub Release| |GitHub Issues|

.. |GitHub Version| image:: https://img.shields.io/github/v/release/ckiplab/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/ckiplab/ckip-transformers/releases

.. |GitHub License| image:: https://img.shields.io/github/license/ckiplab/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/ckiplab/ckip-transformers/blob/master/LICENSE

.. |GitHub Release| image:: https://img.shields.io/github/release-date/ckiplab/ckip-transformers.svg?cacheSeconds=3600

.. |GitHub Downloads| image:: https://img.shields.io/github/downloads/ckiplab/ckip-transformers/total.svg?cacheSeconds=3600
   :target: https://github.com/ckiplab/ckip-transformers/releases/latest

.. |GitHub Issues| image:: https://img.shields.io/github/issues/ckiplab/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/ckiplab/ckip-transformers/issues

.. |GitHub Forks| image:: https://img.shields.io/github/forks/ckiplab/ckip-transformers.svg?style=social&label=Fork&cacheSeconds=3600

.. |GitHub Stars| image:: https://img.shields.io/github/stars/ckiplab/ckip-transformers.svg?style=social&label=Star&cacheSeconds=3600

.. |GitHub Watchers| image:: https://img.shields.io/github/watchers/ckiplab/ckip-transformers.svg?style=social&label=Watch&cacheSeconds=3600

PyPI
^^^^

| https://pypi.org/project/ckip-transformers
| |PyPI Version| |PyPI License| |PyPI Downloads| |PyPI Python| |PyPI Implementation| |PyPI Format| |PyPI Status|

.. |PyPI Version| image:: https://img.shields.io/pypi/v/ckip-transformers.svg?cacheSeconds=3600
   :target: https://pypi.org/project/ckip-transformers

.. |PyPI License| image:: https://img.shields.io/pypi/l/ckip-transformers.svg?cacheSeconds=3600
   :target: https://github.com/ckiplab/ckip-transformers/blob/master/LICENSE

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/ckip-transformers.svg?cacheSeconds=3600
   :target: https://pypi.org/project/ckip-transformers#files

.. |PyPI Python| image:: https://img.shields.io/pypi/pyversions/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Implementation| image:: https://img.shields.io/pypi/implementation/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Format| image:: https://img.shields.io/pypi/format/ckip-transformers.svg?cacheSeconds=3600

.. |PyPI Status| image:: https://img.shields.io/pypi/status/ckip-transformers.svg?cacheSeconds=3600

Documentation
^^^^^^^^^^^^^

| https://ckip-transformers.readthedocs.io
| |ReadTheDocs Home|

.. |ReadTheDocs Home| image:: https://img.shields.io/website/https/ckip-transformers.readthedocs.io.svg?cacheSeconds=3600&up_message=online&down_message=offline
   :target: https://ckip-transformers.readthedocs.io

Demo
^^^^

| https://ckip.iis.sinica.edu.tw/service/transformers
| |Transformers Demo|

.. |Transformers Demo| image:: https://img.shields.io/website/https/ckip.iis.sinica.edu.tw/service/transformers.svg?cacheSeconds=3600&up_message=online&down_message=offline
   :target: https://ckip.iis.sinica.edu.tw/service/transformers

Contributers
^^^^^^^^^^^^

* `Mu Yang <https://muyang.pro>`__ at `CKIP <https://ckip.iis.sinica.edu.tw>`__ (Author & Maintainer).
* `Wei-Yun Ma <https://www.iis.sinica.edu.tw/pages/ma/>`__ at `CKIP <https://ckip.iis.sinica.edu.tw>`__ (Maintainer).

Related Packages
^^^^^^^^^^^^^^^^

- `CkipTagger <https://github.com/ckiplab/ckiptagger>`_: An alternative Chinese NLP library with using BiLSTM.
- `CKIP CoreNLP Toolkit <https://github.com/ckiplab/ckipnlp>`_: A Chinese NLP library with more NLP tasks and utilities.

Models
------

| You may also use our pretrained models with HuggingFace transformers library directly: https://huggingface.co/ckiplab/.
| 您可於 https://huggingface.co/ckiplab/ 下載預訓練的模型。

- Language Models
   * `ALBERT Tiny <https://huggingface.co/ckiplab/albert-tiny-chinese>`_: ``ckiplab/albert-tiny-chinese``
   * `ALBERT Base <https://huggingface.co/ckiplab/albert-base-chinese>`_: ``ckiplab/albert-base-chinese``
   * `BERT Base <https://huggingface.co/ckiplab/bert-base-chinese>`_: ``ckiplab/bert-base-chinese``
   * `GPT2 Base <https://huggingface.co/ckiplab/gpt2-base-chinese>`_: ``ckiplab/gpt2-base-chinese``

- NLP Task Models
   * `ALBERT Tiny — Word Segmentation <https://huggingface.co/ckiplab/albert-tiny-chinese-ws>`_: ``ckiplab/albert-tiny-chinese-ws``
   * `ALBERT Tiny — Part-of-Speech Tagging <https://huggingface.co/ckiplab/albert-tiny-chinese-pos>`_: ``ckiplab/albert-tiny-chinese-pos``
   * `ALBERT Tiny — Named-Entity Recognition <https://huggingface.co/ckiplab/albert-tiny-chinese-ner>`_: ``ckiplab/albert-tiny-chinese-ner``
   * `ALBERT Base — Word Segmentation <https://huggingface.co/ckiplab/albert-base-chinese-ws>`_: ``ckiplab/albert-base-chinese-ws``
   * `ALBERT Base — Part-of-Speech Tagging <https://huggingface.co/ckiplab/albert-base-chinese-pos>`_: ``ckiplab/albert-base-chinese-pos``
   * `ALBERT Base — Named-Entity Recognition <https://huggingface.co/ckiplab/albert-base-chinese-ner>`_: ``ckiplab/albert-base-chinese-ner``
   * `BERT Base — Word Segmentation <https://huggingface.co/ckiplab/bert-base-chinese-ws>`_: ``ckiplab/bert-base-chinese-ws``
   * `BERT Base — Part-of-Speech Tagging <https://huggingface.co/ckiplab/bert-base-chinese-pos>`_: ``ckiplab/bert-base-chinese-pos``
   * `BERT Base — Named-Entity Recognition <https://huggingface.co/ckiplab/bert-base-chinese-ner>`_: ``ckiplab/bert-base-chinese-ner``

Model Usage
^^^^^^^^^^^

| You may use our model directly from the HuggingFace's transformers library
| 您可直接透過 HuggingFace's transformers 套件使用我們的模型

.. code-block:: bash

   pip install -U transformers

| Please use BertTokenizerFast as tokenizer, and replace ``ckiplab/albert-tiny-chinese`` and ``ckiplab/albert-tiny-chinese-ws`` by any model you need in the following example.
| 請使用內建的 BertTokenizerFast，並將以下範例中的 ``ckiplab/albert-tiny-chinese`` 與 ``ckiplab/albert-tiny-chinese-ws`` 替換成任何您要使用的模型名稱。

.. code-block:: python

   from transformers import (
      BertTokenizerFast,
      AutoModelForMaskedLM,
      AutoModelForCausalLM,
      AutoModelForTokenClassification,
   )

   # masked language model (ALBERT, BERT)
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
   model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-tiny-chinese') # or other models above

   # casual language model (GPT2)
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
   model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese') # or other models above

   # nlp task model
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
   model = AutoModelForTokenClassification.from_pretrained('ckiplab/albert-tiny-chinese-ws') # or other models above

Model Fine-Tunning
^^^^^^^^^^^^^^^^^^

| To fine tunning our model on your own datasets, please refer to the following example from HuggingFace's transformers.
| 您可參考以下的範例去微調我們的模型於您自己的資料集。

- https://github.com/huggingface/transformers/tree/master/examples
- https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling
- https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification

| Remember to set ``--tokenizer_name bert-base-chinese`` in order to use Chinese tokenizer.
| 記得設置 ``--tokenizer_name bert-base-chinese`` 以正確的使用中文的 tokenizer。

.. code-block:: bash

   python run_mlm.py \
      --model_name_or_path ckiplab/albert-tiny-chinese \ # or other models above
      --tokenizer_name bert-base-chinese \
      ...

   python run_ner.py \
      --model_name_or_path ckiplab/albert-tiny-chinese-ws \ # or other models above
      --tokenizer_name bert-base-chinese \
      ...

Model Performance
^^^^^^^^^^^^^^^^^

| The following is a performance comparison between our model and other models.
| The results are tested on a traditional Chinese corpus.
| 以下是我們的模型與其他的模型之性能比較。
| 各個任務皆測試於繁體中文的測試集。

================================  ===========  ===========  ========  ==========  =========
Model                             #Parameters  Perplexity†  WS (F1)‡  POS (ACC)‡  NER (F1)‡
================================  ===========  ===========  ========  ==========  =========
ckiplab/albert-tiny-chinese         4M          4.80        96.66%    94.48%      71.17%
ckiplab/albert-base-chinese        10M          2.65        97.33%    95.30%      79.47%
ckiplab/bert-base-chinese         102M          1.88        97.60%    95.67%      81.18%
ckiplab/gpt2-base-chinese         102M         14.40        --        --          --
--------------------------------  -----------  -----------  --------  ----------  ---------

--------------------------------  -----------  -----------  --------  ----------  ---------
voidful/albert_chinese_tiny         4M         74.93        --        --          --
voidful/albert_chinese_base        10M         22.34        --        --          --
bert-base-chinese                 102M          2.53        --        --          --
================================  ===========  ===========  ========  ==========  =========

| † Perplexity; the smaller the better.
| † 混淆度；數字越小越好。
| ‡ WS: word segmentation; POS: part-of-speech; NER: named-entity recognition; the larger the better.
| ‡ WS: 斷詞；POS: 詞性標記；NER: 實體辨識；數字越大越好。

Training Corpus
^^^^^^^^^^^^^^^

| The language models are trained on the ZhWiki and CNA datasets; the WS and POS tasks are trained on the ASBC dataset; the NER tasks are trained on the OntoNotes dataset.
| 以上的語言模型訓練於 ZhWiki 與 CNA 資料集上；斷詞（WS）與詞性標記（POS）任務模型訓練於 ASBC 資料集上；實體辨識（NER）任務模型訓練於 OntoNotes 資料集上。

* ZhWiki: https://dumps.wikimedia.org/zhwiki/
   | Chinese Wikipedia text (20200801 dump), translated to Traditional using `OpenCC <https://github.com/BYVoid/OpenCC>`_.
   | 中文維基的文章（20200801 版本），利用 `OpenCC <https://github.com/BYVoid/OpenCC>`_ 翻譯成繁體中文。
* CNA: https://catalog.ldc.upenn.edu/LDC2011T13
   | Chinese Gigaword Fifth Edition — CNA (Central News Agency) part.
   | 中文 Gigaword 第五版 — CNA（中央社）的部分。
* ASBC: http://asbc.iis.sinica.edu.tw
   | Academia Sinica Balanced Corpus of Modern Chinese release 4.0.
   | 中央研究院漢語平衡語料庫第四版。
* OntoNotes: https://catalog.ldc.upenn.edu/LDC2013T19
   | OntoNotes release 5.0, Chinese part, translated to Traditional using `OpenCC <https://github.com/BYVoid/OpenCC>`_.
   | OntoNotes 第五版，中文部分，利用 `OpenCC <https://github.com/BYVoid/OpenCC>`_ 翻譯成繁體中文。

| Here is a summary of each corpus.
| 以下是各個資料集的一覽表。

================  ================  ================  ================  ================
Dataset           #Documents        #Lines            #Characters       Line Type
================  ================  ================  ================  ================
CNA               2,559,520         13,532,445        1,219,029,974     Paragraph
ZhWiki            1,106,783         5,918,975         495,446,829       Paragraph
ASBC              19,247            1,395,949         17,572,374        Clause
OntoNotes         1,911             48,067            1,568,491         Sentence
================  ================  ================  ================  ================

| Here is the dataset split used for language models.
| 以下是用於訓練語言模型的資料集切割。

================  ================  ================  ================
CNA+ZhWiki        #Documents        #Lines            #Characters
================  ================  ================  ================
Train             3,606,303         18,986,238        4,347,517,682
Dev               30,000            148,077           32,888,978
Test              30,000            151,241           35,216,818
================  ================  ================  ================

| Here is the dataset split used for word segmentation and part-of-speech tagging models.
| 以下是用於訓練斷詞及詞性標記模型的資料集切割。

================  ================  ================  ================  ================
ASBC              #Documents        #Lines            #Words            #Characters
================  ================  ================  ================  ================
Train             15,247            1,183,260         9,480,899         14,724,250
Dev               2,000             52,677            448,964           741,323
Test              2,000             160,012           1,315,129         2,106,799
================  ================  ================  ================  ================


| Here is the dataset split used for word segmentation and named entity recognition models.
| 以下是用於訓練實體辨識模型的資料集切割。

================  ================  ================  ================  ================
OntoNotes         #Documents        #Lines            #Characters       #Named-Entities
================  ================  ================  ================  ================
Train             1,511             43,362            1,367,658         68,947
Dev               200               2,304             93,535            7,186
Test              200               2,401             107,298           6,977
================  ================  ================  ================  ================

NLP Tools
---------

| The package also provide the following NLP tools.
| 我們的套件也提供了以下的自然語言處理工具。

* (WS) Word Segmentation 斷詞
* (POS) Part-of-Speech Tagging 詞性標記
* (NER) Named Entity Recognition 實體辨識

Installation
^^^^^^^^^^^^

``pip install -U ckip-transformers``

Requirements:

* `Python <https://www.python.org>`__ 3.6+
* `PyTorch <https://pytorch.org>`__ 1.5+
* `HuggingFace Transformers <https://huggingface.co/transformers/>`__ 3.5+

NLP Tools Usage
^^^^^^^^^^^^^^^

| See `here <../_api/ckip_transformers.html>`_ for API details.
| 詳細的 API 請參見 `此處 <../_api/ckip_transformers.html>`_ 。

| The complete script of this example is https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py.
| 以下的範例的完整檔案可參見 https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py 。

1. Import module
""""""""""""""""

.. code-block:: python

   from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

2. Load models
""""""""""""""

| We provide three levels (1–3) of drivers. Level 1 is the fastest, and level 3 (default) is the most accurate.
| 我們的工具分為三個等級（1—3）。等級一最快，等級三（預設值）最精準。

.. code-block:: python

   # Initialize drivers
   ws_driver = CkipWordSegmenter(level=3)
   pos_driver = CkipPosTagger(level=3)
   ner_driver = CkipNerChunker(level=3)

| One may also load their own checkpoints using our drivers.
| 也可以運用我們的工具於自己訓練的模型上。

.. code-block:: python

   # Initialize drivers with custom checkpoints
   ws_driver  = CkipWordSegmenter(model_name="path_to_your_model")
   pos_driver = CkipPosTagger(model_name="path_to_your_model")
   ner_driver = CkipNerChunker(model_name="path_to_your_model")

| To use GPU, one may specify device ID while initialize the drivers. Set to -1 (default) to disable GPU.
| 可於宣告斷詞等工具時指定 device 以使用 GPU，設為 -1 （預設值）代表不使用 GPU。

.. code-block:: python

   # Use CPU
   ws_driver = CkipWordSegmenter(device=-1)

   # Use GPU:0
   ws_driver = CkipWordSegmenter(device=0)

3. Run pipeline
"""""""""""""""

| The input for word segmentation and named-entity recognition must be a list of sentences.
| The input for part-of-speech tagging must be a list of list of words (the output of word segmentation).
| 斷詞與實體辨識的輸入必須是 list of sentences。
| 詞性標記的輸入必須是 list of list of words。

.. code-block:: python

   # Input text
   text = [
      "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
      "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
      "空白 也是可以的～",
   ]

   # Run pipeline
   ws  = ws_driver(text)
   pos = pos_driver(ws)
   ner = ner_driver(text)

| The POS driver will automatically segment the sentence internally using there characters ``'，,。：:；;！!？?'`` while running the model. (The output sentences will be concatenated back.) You may set ``delim_set`` to any characters you want.
| You may set ``use_delim=False`` to disable this feature, or set ``use_delim=True`` in WS and NER driver to enable this feature.
| 詞性標記工具會自動用 ``'，,。：:；;！!？?'`` 等字元在執行模型前切割句子（輸出的句子會自動接回）。可設定 ``delim_set`` 參數使用別的字元做切割。
| 另外可指定 ``use_delim=False`` 已停用此功能，或於斷詞、實體辨識時指定 ``use_delim=False`` 已啟用此功能。

.. code-block:: python

   # Enable sentence segmentation
   ws  = ws_driver(text, use_delim=True)
   ner = ner_driver(text, use_delim=True)

   # Disable sentence segmentation
   pos = pos_driver(ws, use_delim=False)

   # Use new line characters and tabs for sentence segmentation
   pos = pos_driver(ws, delim_set='\n\t')

| You may specify ``batch_size`` and ``max_length`` to better utilize you machine resources.
| 您亦可設置 ``batch_size`` 與 ``max_length`` 以更完美的利用您的機器資源。

.. code-block:: python

   # Sets the batch size and maximum sentence length
   ws = ws_driver(text, batch_size=256, max_length=512)

4. Show results
"""""""""""""""

.. code-block:: python

   # Pack word segmentation and part-of-speech results
   def pack_ws_pos_sentece(sentence_ws, sentence_pos):
      assert len(sentence_ws) == len(sentence_pos)
      res = []
      for word_ws, word_pos in zip(sentence_ws, sentence_pos):
         res.append(f"{word_ws}({word_pos})")
      return "\u3000".join(res)

   # Show results
   for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
      print(sentence)
      print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
      for entity in sentence_ner:
         print(entity)
      print()

.. code-block:: text

   傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。
   傅達仁(Nb)　今(Nd)　將(D)　執行(VC)　安樂死(Na)　，(COMMACATEGORY)　卻(D)　突然(D)　爆出(VJ)　自己(Nh)　20(Neu)　年(Nd)　前(Ng)　遭(P)　緯來(Nb)　體育台(Na)　封殺(VC)　，(COMMACATEGORY)　他(Nh)　不(D)　懂(VK)　自己(Nh)　哪裡(Ncd)　得罪到(VC)　電視台(Nc)　。(PERIODCATEGORY)
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

   空白 也是可以的～
   空白(VH)　 (WHITESPACE)　也(D)　是(SHI)　可以(VH)　的(T)　～(FW)

NLP Tools Performance
^^^^^^^^^^^^^^^^^^^^^

| The following is a performance comparison between our tool and other tools.
| 以下是我們的工具與其他的工具之性能比較。

CKIP Transformers v.s. Monpa & Jeiba
""""""""""""""""""""""""""""""""""""

=====  ========================  ===========  =============  ===============  ============
Level  Tool                        WS (F1)      POS (Acc)      WS+POS (F1)      NER (F1)
=====  ========================  ===========  =============  ===============  ============
3      CKIP BERT Base            **97.60%**   **95.67%**     **94.19%**       **81.18%**
2      CKIP ALBERT Base            97.33%       95.30%         93.52%           79.47%
1      CKIP ALBERT Tiny            96.66%       94.48%         92.25%           71.17%
-----  ------------------------  -----------  -------------  ---------------  ------------

-----  ------------------------  -----------  -------------  ---------------  ------------
--     Monpa†                      92.58%       --             83.88%           --
--     Jeiba                       81.18%       --             --               --
=====  ========================  ===========  =============  ===============  ============

| † Monpa provides only 3 types of tags in NER.
| † Monpa 的實體辨識僅提供三種標記而已。

CKIP Transformers v.s. CkipTagger
""""""""""""""""""""""""""""""""""""

| The following results are tested on a different dataset.†
| 以下實驗在另一個資料集測試。†

=====  ========================  ===========  =============  ===============  ============
Level  Tool                        WS (F1)      POS (Acc)      WS+POS (F1)      NER (F1)
=====  ========================  ===========  =============  ===============  ============
3      CKIP BERT Base            **97.84%**     96.46%       **94.91%**       **79.20%**
--     CkipTagger                  97.33%     **97.20%**       94.75%           77.87%
=====  ========================  ===========  =============  ===============  ============

| † Here we retrained/tested our BERT model using the same dataset with CkipTagger.
| † 我們重新訓練／測試我們的 BERT 模型於跟 CkipTagger 相同的資料集。

License
-------

|GPL-3.0|

Copyright (c) 2020 `CKIP Lab <https://ckip.iis.sinica.edu.tw>`__ under the `GPL-3.0 License <https://www.gnu.org/licenses/gpl-3.0.html>`__.

.. |GPL-3.0| image:: https://www.gnu.org/graphics/gplv3-with-text-136x68.png
   :target: https://www.gnu.org/licenses/gpl-3.0.html
