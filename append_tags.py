import pickle
from cmi import get_cmi_bucket_tag
import sys
#Usage: python3 append_tags.py k cmi_cutoff_dict_file inp_file out_file 
# Sample command for generating train file for 3 bins: python3 append_tags.py 3 cmi_cutoffs_dict.pkl cm_data/raw_splits/hi_cm_train.tsv cm_data/cmi_tag/hi_cm_train_3_bins.tsv
k = int(sys.argv[1])
cmi_cutoffs_dict = pickle.load(open(sys.argv[2],'rb'))
with open(sys.argv[3],'r') as f:
	data = f.readlines()
	data = [line.strip("\n").split("\t") for line in data]
	output_data = []
	for ele in data:
		src = ele[0]
		tgt = ele[1]
		output_data.append(get_cmi_bucket_tag(tgt,cmi_cutoffs_dict[k])+" "+src+"\t"+tgt+"\n")
	with open(sys.argv[4],'w') as output_f:
		output_f.writelines(output_data)
