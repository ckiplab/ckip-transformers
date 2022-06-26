#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


def main():

    # Show version
    print(__version__)

    # Initialize drivers
    print("Initializing drivers ... WS")
    ws_driver = CkipWordSegmenter(model="bert-base")
    print("Initializing drivers ... POS")
    pos_driver = CkipPosTagger(model="bert-base")
    print("Initializing drivers ... NER")
    ner_driver = CkipNerChunker(model="bert-base")
    print("Initializing drivers ... done")
    print()

    # Input text
    text = [
        "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
        "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
        "空白 也是可以的～",
    ]

    # Run pipeline
    print("Running pipeline ... WS")
    ws = ws_driver(text)
    print("Running pipeline ... POS")
    pos = pos_driver(ws)
    print("Running pipeline ... NER")
    ner = ner_driver(text)
    print("Running pipeline ... done")
    print()

    # Show results
    for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
        print(sentence)
        print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
        for entity in sentence_ner:
            print(entity)
        print()


# Pack word segmentation and part-of-speech results
def pack_ws_pos_sentece(sentence_ws, sentence_pos):
    assert len(sentence_ws) == len(sentence_pos)
    res = []
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        res.append(f"{word_ws}({word_pos})")
    return "\u3000".join(res)


if __name__ == "__main__":
    main()
