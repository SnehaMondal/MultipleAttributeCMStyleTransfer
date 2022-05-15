import numpy as np
from datasets import load_metric
import sys
import os
import sacrebleu

from cmi import cmi, get_cmi_bucket_tag


def bleu(targets, predictions):
	if isinstance(targets[0], list):
		targets = [[x for x in target] for target in targets]
	else:
	# Need to wrap targets in another list for corpus_bleu.
		targets = [targets]

	bleu_score = sacrebleu.corpus_bleu(predictions, targets,
									 smooth_method="exp",
									 smooth_value=0.0,
									 force=False,
									 lowercase=False,
									 tokenize="intl",
									 use_effective_order=False)
	return {"bleu": bleu_score.score}

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

def cmi_acc_bleu_hm(targets, predictions):
	#scale cmi_bucket_accuracy to be in the same range as bleu
	cmi_bucket_accuracy_score = cmi_bucket_accuracy(targets, predictions)["cmi_bucket_accuracy"]*100
	bleu_score = bleu(targets, predictions)["bleu"]

	hm = (2*cmi_bucket_accuracy_score*bleu_score)/(cmi_bucket_accuracy_score+bleu_score)
	return {"cmi_acc_bleu_hm" : np.round(hm, 2)}

def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [label.strip() for label in labels]

	return preds, labels

def compute_metrics(eval_preds, tokenizer, data_args):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	if data_args.ignore_pad_token_for_loss:
		# Replace -100 in the labels as we can't decode them.
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

	# Some simple post-processing
	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	result = bleu(decoded_labels, decoded_preds)
	result = {"bleu": result["bleu"]}

	cmi_acc = cmi_bucket_accuracy(decoded_labels, decoded_preds)
	result["cmi_acc"] = cmi_acc["cmi_bucket_accuracy"]*100

	cmi_corr = cmi_correlation(decoded_labels, decoded_preds)
	result["cmi_corr"] = cmi_corr["cmi_correlation"]

	hm = cmi_acc_bleu_hm(decoded_labels, decoded_preds)
	result["cmi_acc_bleu_hm"] = hm["cmi_acc_bleu_hm"]

	result = {k: round(v, 2) for k, v in result.items()}
	return result