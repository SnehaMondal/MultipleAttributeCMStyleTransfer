import sys
import numpy as np

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

def is_different_script(word1, word2):
    return word1.isascii()^word2.isascii()

def get_i_index(sentence) -> float:
    sp=0
    tokens = sentence.split(' ')
    if len(tokens)>=2:
        for j in range(1, len(tokens)):
            this_word = tokens[j]
            prev_word = tokens[j-1]
            if is_different_script(prev_word, this_word):
                sp+=1
        return sp/(len(tokens)-1)
    else:
        return 0

fn1 = sys.argv[1]
fn2 = sys.argv[2]
mode = sys.argv[3]

of = open(fn2, 'w')

zeros = 0
with open(fn1, 'r') as f:
    for l in f:
        l = l.strip()
        src, tgt = l.split('\t')[:2]
        val1 = 0
        val2 = 0
        if mode=="dev":
            r = np.random.rand()
            if r>0.2:
                val1 = 0.1+cmi(tgt)
                val1 = round(val1, 3)
                val2 = 0.1+get_i_index(tgt)
                val2 = round(val2, 3)
            else: zeros+=1
        elif mode=="oracle":
            val1 = 0.1+cmi(tgt)
            val1 = round(val1, 3)
            val2 = 0.1+get_i_index(tgt)
            val2 = round(val2, 3)
        of.write(src+'\t'+tgt+'\t'+str(val1)+'\t'+str(val2)+'\n')

print(zeros)
