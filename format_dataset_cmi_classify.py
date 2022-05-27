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

CMI_CUTOFF_LO = 0.17
CMI_CUTOFF_MID = 0.30

def get_cmi_bucket_tag(s):
    cmi_score = cmi(s)
    if cmi_score==0:
        cmi_bucket_tag = "cmi_zero"
    elif cmi_score <= CMI_CUTOFF_LO:
        cmi_bucket_tag = "cmi_lo"
    elif cmi_score <= CMI_CUTOFF_MID:
        cmi_bucket_tag = "cmi_mid"
    else:
        cmi_bucket_tag = "cmi_hi"
    return cmi_bucket_tag

fn1 = sys.argv[1]
fn2 = sys.argv[2]
num = int(sys.argv[3])

of = open(fn2, 'w')

def get_random_cmi(tgt_bucket):
    if tgt_bucket=="cmi_lo":
        rand_cmi = np.random.uniform(0, CMI_CUTOFF_LO)
    elif tgt_bucket=="cmi_mid":
        rand_cmi = np.random.uniform(CMI_CUTOFF_LO, CMI_CUTOFF_MID)
    else:
        rand_cmi = np.random.uniform(CMI_CUTOFF_MID, 0.5)
    return rand_cmi 

buckets = ['cmi_lo', 'cmi_mid', 'cmi_hi']

with open(fn1, 'r') as f:
    for l in f:
        l = l.strip()
        src, _ = l.split('\t')[:2]
        val1 = 0
        val1 = cmi(tgt)
        true_bucket = get_cmi_tag(src)
        of.write(src+'\t'+str(val1)+'\t1\n')
        for x in buckets:
            if x!=true_bucket
            fake_cmi = get_random_cmi(x)
            of.write*src+'\t'_str(fake_cmi)+'\t0\n')

