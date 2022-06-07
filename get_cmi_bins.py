import numpy as np
import sys
import os
import pickle
from cmi import cmi

data = []
directory = sys.argv[1]
for filename in os.listdir(directory):
	file_path = os.path.join(directory,filename)
	with open(file_path) as f:
		l = f.readlines()
		data += [round(cmi(line.split("\t")[1]),3) for line in l]

data = list(filter(lambda x: x!=0, data))
cmi_cutoffs = {}
for k in range(2,8):
	cmi_cutoffs[k]=[]
	for i in range(1,k):
		cmi_cutoffs[k].append(np.round(np.percentile(data,i*100/k),3))

pickle.dump(cmi_cutoffs,open("cmi_cutoffs_dict.pkl",'wb'))
