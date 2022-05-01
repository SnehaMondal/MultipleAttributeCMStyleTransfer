import numpy as np
from datasets import load_metric
from statistics import harmonic_mean
from spi import spi_bucket_accuracy, spi_correlation
import sys
sys.path.append('../controllable-codeximing')
from t5.evaluation.metrics import bleu, cmi_bucket_accuracy, cmi_correlation

metric = load_metric("sacrebleu")

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

    cmi_acc = cmi_bucket_accuracy(decoded_labels, decoded_preds)
    result["cmi_acc"] = cmi_acc["cmi_bucket_accuracy"]

    cmi_corr = cmi_correlation(decoded_labels, decoded_preds)
    result["cmi_corr"] = cmi_corr["cmi_correlation"]

    spi_acc = spi_bucket_accuracy(decoded_labels, decoded_preds)
    result["spi_acc"] = spi_acc["spi_bucket_accuracy"]

    spi_corr = spi_correlation(decoded_labels, decoded_preds)
    result["spi_corr"] = spi_corr["spi_correlation"]

    cmi_spi_bleu_hm = acc_bleu_hm(result["cmi_acc"], result["spi_acc"], result["bleu"])
    result["cmi_spi_bleu_hm"] = cmi_spi_bleu_hm["acc_bleu_hm"]

    result = {k: round(v, 4) for k, v in result.items()}
    return result
