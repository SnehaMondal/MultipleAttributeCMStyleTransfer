import numpy as np
import bisect

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

def get_cmi_bucket_tag(s,cmi_cutoffs):
        cmi_score = cmi(s)
        if cmi_score==0:
                cmi_bucket_tag = "cmi_0"
        else:
        	cmi_bin=bisect.bisect_left(cmi_cutoffs,cmi_score)+1
        	cmi_bucket_tag = "cmi_"+str(cmi_bin)
        return cmi_bucket_tag

def get_cmi_bucket_accuracy(target,prediction,cmi_cutoffs):
	cmi_bucket_prediction = get_cmi_bucket_tag(prediction,cmi_cutoffs)
	cmi_bucket_target = get_cmi_bucket_tag(target,cmi_cutoffs)
	return float(cmi_bucket_prediction == cmi_bucket_target)

def cmi_bucket_accuracy(targets,predictions,cmi_cutoffs):
	total_accuracy = sum(
			get_cmi_bucket_accuracy(target,prediction,cmi_cutoffs)
			for target,prediction in zip(targets,predictions))
	return {"cmi_bucket_accuracy":total_accuracy/len(predictions)}

def cmi_correlation(targets, predictions):
	cmi_target = [cmi(sentence) for sentence in targets]
	cmi_prediction = [cmi(sentence) for sentence in predictions]
	correlation = np.corrcoef(cmi_target, cmi_prediction)[0,1]
	return {"cmi_correlation" : correlation}
