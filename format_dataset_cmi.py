#command to run: python3 format_dataset_cmi.py {input_file} {output_file}

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

fn1 = sys.argv[1]
fn2 = sys.argv[2]
mode = sys.argv[3]

of = open(fn2, 'w')

zeros = 0
with open(fn1, 'r') as f:
    for l in f:
        l = l.strip()
        src, tgt = l.split('\t')[:2]
        val = 0
        if mode=="dev":
            r = np.random.rand()
            if r>0.2:
                val = 0.1+cmi(tgt)
                val = round(val, 3)
            else: zeros+=1
        elif mode=="oracle":
            val = 0.1+cmi(tgt)
            val = round(val, 3)
        of.write(src+'\t'+tgt+'\t'+str(val)+'\n')
