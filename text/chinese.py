import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text import symbols

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {line.split("\t")[0]: line.strip().split("\t")[1] for line in
                        open(os.path.join(current_file_path, 'opencpop-strict.txt')).readlines()}



def replace_pu_symbol(ph):
    rep_map = {
        '：': ',',
        '；': ',',
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '\n': '.',
        "·": ",",
        '、': ",",
        '...': '…',
        '$': '.',
        '“': "'",
        '”': "'",
        '‘': "'",
        '’': "'",
        '（': "'",
        '）': "'",
        '(': "'",
        ')': "'",
        '《': "'",
        '》': "'",
        '【': "'",
        '】': "'",
        '[': "'",
        ']': "'",
        '—': "-",
        '～': "-",
        '~': "-",
        '「': "'",
        '」': "'",
    }
    if ph in symbols:
        return ph
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph not in symbols:
        ph = 'UNK'
    return ph

def text_normalize(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def g2p(norm_text):
    pinyins = lazy_pinyin(norm_text, neutral_tone_with_five=True, style=Style.TONE3, strict=False)
    tones = []
    phones = []
    word2ph = []
    replace_map = {
        'n2': 'en2'
    }
    for pinyin in pinyins:

        if pinyin[-1] in '12345':
            if pinyin in replace_map.keys():
                pinyin = replace_map[pinyin]
            tone = pinyin[-1]
            pinyin = pinyin[:-1]
            phs = pinyin_to_symbol_map[pinyin].split(" ")
            for ph in phs:
                phones.append(ph)
                tones.append(int(tone))
            word2ph.append(len(phs))
        else:
            for i in (pinyin):
                phones.append(replace_pu_symbol(i))
                tones.append(0)
                word2ph.append(1)
    assert len(word2ph) == len(norm_text)
    assert sum(word2ph) == len(phones)

    return phones, tones, word2ph

def clean_pinyin(pinyins):
    tones = []
    phones = []
    replace_map = {
        'n2': 'en2'
    }
    for pinyin in pinyins:
        if pinyin == "SP":
            phones.append(pinyin)
            tones.append(0)

        elif pinyin[-1] in '12345':
            if pinyin in replace_map.keys():
                pinyin = replace_map[pinyin]
            tone = pinyin[-1]
            pinyin = pinyin[:-1]
            phs = pinyin_to_symbol_map[pinyin].split(" ")
            for ph in phs:
                phones.append(ph)
                tones.append(int(tone))
        else:
            for i in (pinyin):
                phones.append(replace_pu_symbol(i))
                tones.append(0)

    return phones, tones


if __name__ == '__main__':
    text = "但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(g2p(text))
