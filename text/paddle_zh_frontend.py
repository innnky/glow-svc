
from text.frontend.zh_frontend import Frontend
frontend = Frontend()

pinyin_to_symbol_map = {line.split("\t")[0]:line.strip().split("\t")[1] for line in open("text/opencpop-strict.txt").readlines()}


paddle_dict = [i.strip() for i in open("text/paddle_dict.dict").readlines()]
paddle_dict = {i.split("\t")[0]: i.split("\t")[1] for i in paddle_dict}

reversed_paddle_dict = {}
all_zh_phones = set()
for k, v in paddle_dict.items():
    reversed_paddle_dict[v] = k
    [all_zh_phones.add(i) for i in v.split(" ")]

def paddle_phones_to_pinyins(phones):
    pinyins = []
    accu_ph = []
    for ph in phones:
        accu_ph.append(ph)
        if ph not in all_zh_phones:
            assert len(accu_ph) == 1
            pinyins.append(ph)
            accu_ph = []
        elif " ".join(accu_ph) in reversed_paddle_dict.keys():
            pinyins.append( reversed_paddle_dict[" ".join(accu_ph)])
            accu_ph = []
    if not  accu_ph==[]:
        print(accu_ph)
    return pinyins

def pu_symbol_replace(data):
    chinaTab = ['！', '？', "…", "，", "。",'、', "..."]
    englishTab = ['!', '?', "…", ",", ".",",", "…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

def zh_to_pinyin(text):
    phones = zh_to_paddle_phonemes(text)
    pinyins = paddle_phones_to_pinyins(phones)
    return pinyins

def zh_to_paddle_phonemes(text):
    # 替换标点为英文标点
    text = pu_symbol_replace(text)
    phones = frontend.get_phonemes(text)[0]
    return phones

def zh_to_phonemes(text):
    pinyins = zh_to_pinyin(text)
    phs = []
    tones =[]
    for pinyin in pinyins:
        if pinyin =="#":
            continue
        elif len(pinyin) == 1:
            phs.append("SP")
            tones.append(0)
        else:
            ph, tone = pinyin[:-1], pinyin[-1]
            ph_pair = pinyin_to_symbol_map[ph].split(" ")
            for p in ph_pair:
                phs.append(p)
                tones.append(int(tone))
    return phs, tones

if __name__ == '__main__':
    print(zh_to_phonemes("替换标点为英文标点,这是好事"))

