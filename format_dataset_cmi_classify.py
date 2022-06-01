import sys
import numpy as np
from cmi import *
np.random.seed(0)

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
        _, src = l.split('\t')[:2]
        val1 = 0
        val1 = cmi(src)
        true_bucket = get_cmi_bucket_tag(src)
        of.write("\t".join([src, str(val1), "1"]))
        for x in buckets:
            if x!=true_bucket:
                fake_cmi = get_random_cmi(x)
                of.write("\t".join([src, str(fake_cmi), "0"]))

