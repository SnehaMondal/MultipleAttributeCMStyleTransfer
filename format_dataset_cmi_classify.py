import sys
import numpy as np
from cmi import cmi, get_cmi_bucket_tag, CMI_CUTOFF_LO, CMI_CUTOFF_MID

fn1 = sys.argv[1]
fn2 = sys.argv[2]
# num = int(sys.argv[3])

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
		_, tgt = l.split('\t')[:2]
		true_cmi = np.round(cmi(tgt), 3)
		true_bucket = get_cmi_bucket_tag(tgt)
		of.write("\t".join([tgt, str(true_cmi), str(1)]) + "\n")
		for bucket in buckets:
			if bucket != true_bucket:
				fake_cmi = np.round(get_random_cmi(bucket), 3)
				of.write("\t".join([tgt, str(fake_cmi), str(0)]) + "\n")