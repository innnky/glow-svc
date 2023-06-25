
pu_symbols = ['!', '?', '…', ",", ".", "SP", "UNK"]

symbols = ['_', 'E', 'En', 'a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h',
'i', 'i0', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'ir', 'iu', 'j', 'k', 'l', 'm', 'n', 'o', 'ong',
'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn',
'w', 'x', 'y', 'z', 'zh'] + pu_symbols


sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]
