import sys
fn1 = sys.argv[1]

def cmi(t):
    word_list = t.split()
    if len(word_list)==0:
        return 0
    else:
        en_words=0
        for w in word_list:
            if w.isascii():
                en_words+=1
        cmi = 1 - (max(en_words,len(word_list)-en_words)/len(word_list))
        return cmi

num_mono = 0

with open(fn1, 'w') as f:
    for l in f:
        l = l.strip()
        tgt = l.split('\t')[1]
        if cmi(tgt)==0:
            num_mono+=1

print(num_mono)