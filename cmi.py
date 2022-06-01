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

def get_cmi_bucket_accuracy(target,prediction):
	cmi_bucket_prediction = get_cmi_bucket_tag(prediction)
	cmi_bucket_target = get_cmi_bucket_tag(target)
	return float(cmi_bucket_prediction == cmi_bucket_target)

def cmi_bucket_accuracy(targets,predictions):
	total_accuracy = sum(
			get_cmi_bucket_accuracy(target,prediction)
			for target,prediction in zip(targets,predictions))
	return {"cmi_bucket_accuracy":total_accuracy/len(predictions)}

def cmi_correlation(targets, predictions):
	cmi_target = [cmi(sentence) for sentence in targets]
	cmi_prediction = [cmi(sentence) for sentence in predictions]
	correlation = np.corrcoef(cmi_target, cmi_prediction)[0,1]
	return {"cmi_correlation" : correlation}
