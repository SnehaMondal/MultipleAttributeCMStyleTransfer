import numpy as np
from datasets import load_metric
import sys
sys.path.append('../controllable-codeximing')
from t5.evaluation.metrics import bleu, cmi_bucket_accuracy, cmi_correlation

metric = load_metric("sacrebleu")

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
#    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}

    cmi_acc = cmi_bucket_accuracy(decoded_labels, decoded_preds)
    result["cmi_acc"] = cmi_acc["cmi_bucket_accuracy"]

    cmi_corr = cmi_correlation(decoded_labels, decoded_preds)
    result["cmi_corr"] = cmi_corr["cmi_correlation"]
    result = {k: round(v, 4) for k, v in result.items()}
    return result
