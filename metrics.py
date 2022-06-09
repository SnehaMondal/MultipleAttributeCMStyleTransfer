import numpy as np
from datasets import load_metric
from statistics import harmonic_mean
from spi import spi_bucket_accuracy, spi_correlation
from cmi import cmi_bucket_accuracy, cmi_correlation
from cmi import get_cmi_bucket_tag
import sacrebleu
import pickle

metric = load_metric("sacrebleu")

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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def acc_bleu_hm(cmi_acc, spi_acc,  bleu):
    cmi_acc = cmi_acc*100
    spi_acc = spi_acc*100
    hm = harmonic_mean([cmi_acc, spi_acc, bleu])
    return {"acc_bleu_hm": hm}

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
#    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}
    cmi_cutoffs = pickle.load(open(data_args.cmi_cutoffs_dict,'rb'))[3]
    cmi_acc = cmi_bucket_accuracy(decoded_labels, decoded_preds,cmi_cutoffs)
    result["cmi_acc"] = cmi_acc["cmi_bucket_accuracy"]

    cmi_corr = cmi_correlation(decoded_labels, decoded_preds)
    result["cmi_corr"] = cmi_corr["cmi_correlation"]
    
    result["cmi_bleu_hm"] = harmonic_mean([result["cmi_acc"]*100,result["bleu"]])

    holdout_instances = [(target, prediction) for target, prediction in zip(decoded_labels, decoded_preds) if get_cmi_bucket_tag(target, cmi_cutoffs)==data_args.holdout_bucket]
    print(data_args.holdout_bucket, holdout_instances[0])
    holdout_targets = [v[0] for v in holdout_instances]
    holdout_predictions = [v[1] for v in holdout_instances]

    result["holdout_bleu"] = bleu(holdout_targets, holdout_predictions)["bleu"]
    
    holdout_cmi = cmi_bucket_accuracy(holdout_targets, holdout_predictions, cmi_cutoffs)
    result["holdout_cmi_acc"] = holdout_cmi["cmi_bucket_accuracy"]

    holdout_cmi_corr = cmi_correlation(holdout_targets, holdout_predictions)
    result["holdout_cmi_corr"] = holdout_cmi_corr["cmi_correlation"]

    spi_acc = spi_bucket_accuracy(decoded_labels, decoded_preds)
    result["spi_acc"] = spi_acc["spi_bucket_accuracy"]

    spi_corr = spi_correlation(decoded_labels, decoded_preds)
    result["spi_corr"] = spi_corr["spi_correlation"]
    
    result["spi_bleu_hm"] = harmonic_mean([result["spi_acc"]*100,result["bleu"]])
    
    cmi_spi_bleu_hm = acc_bleu_hm(result["cmi_acc"], result["spi_acc"], result["bleu"])
    result["cmi_spi_bleu_hm"] = cmi_spi_bleu_hm["acc_bleu_hm"]

    result = {k: round(v, 4) for k, v in result.items()}
    return result
