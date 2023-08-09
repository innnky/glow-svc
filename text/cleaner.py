from text import chinese, japanese, english, cleaned_text_to_sequence


language_module_map = {
    'ZH': chinese,
    "JA": japanese,
    "EN": english
}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return phones, tones

def text_to_sequence(text, language):
    phones, tones = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

if __name__ == '__main__':
    print(text_to_sequence("你好，啊啊啊额、还是到付红四方。", 'ZH'))
    print(text_to_sequence("hello", 'EN'))


