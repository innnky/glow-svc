import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text import symbols

pinyin_to_symbol_map = {line.split("\t")[0]:line.strip().split("\t")[1] for line in open("text/opencpop-strict.txt").readlines()}

pu_symbols = ['!', '?', '…', ",", "."]
def replace_str(ph):
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
        '...': '…'
    }
    if ph in symbols:
        return ph
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph not in symbols:
        ph = 'UNK'
    return ph

replace_map = {
    'n2':'en2'
}
def text_normalize(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text

def get_pinyin(text):
    text = text.lower()
    initials = lazy_pinyin(text, neutral_tone_with_five=False, style=Style.INITIALS, strict=False)
    finals = lazy_pinyin(text, neutral_tone_with_five=False, style=Style.FINALS_TONE3)

    text_phone = []
    for _o in zip(initials, finals):
        if _o[0] != _o[1] and _o[0] != '':
            _o = ['@'+i for i in _o]
            text_phone.extend(_o)
        elif _o[0] != _o[1] and _o[0] == '':
            text_phone.append('@'+_o[1])
        else:
            text_phone.extend(list(_o[0]))

    text_phone = " ".join(text_phone)
    return text_phone

def g2p(norm_text):
    pinyins = lazy_pinyin(norm_text, neutral_tone_with_five=True, style=Style.TONE3, strict=False)
    tones = []
    phones = []
    word2ph = []
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
                phones.append(replace_str(i))
                tones.append(0)
                word2ph.append(1)
    assert len(word2ph) == len(norm_text)
    assert sum(word2ph) == len(phones)

    return phones, tones, word2ph

if __name__ == '__main__':
    text = "但是《原神》是由,米哈游自主，研发的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)

    print(g2p(text))





