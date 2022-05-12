import time
import torch
import os

from transformers import T5Tokenizer
from model import MT5ForStyleConditionalGeneration
from datasets import load_metric
metric = load_metric("sacrebleu")

input_filepath = "../cm_data/pretrain_hd_dev_hi_cm.tsv"

import sys
ckpt = sys.argv[1]
#model_checkpoint = "./models/pretrain_hd"
model_checkpoint = "./models/pretrain_hd/checkpoint-"+ckpt
output_filepath = "tagremoved_test_data_translation.csv"

# model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
# output_filepath = "./output/base_mt_en_hi/tagremoved_creatives_test_data_translation.csv"

print(f"Model name : {model_checkpoint}")
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = MT5ForStyleConditionalGeneration.from_pretrained(model_checkpoint)
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
model.to(device)

fw_hi = open(output_filepath, "w")

def translate(sentences):
    batch_encoding = tokenizer(sentences, return_tensors="pt", padding=True)
    batch_encoding['input_ids'] = batch_encoding['input_ids'].to(device)
    batch_encoding['attention_mask'] = batch_encoding['attention_mask'].to(device)
    batch_encoding['input_cmi_scores'] = torch.zeros(batch_encoding['input_ids'].shape[0]).to(device)
    translated = model.generate(**batch_encoding, num_beams=4)
    translated_sentences = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in translated]
    
    return translated_sentences

predictions = []
references = []
with open(input_filepath) as fr:
    i = 0
    batch = []
    st_time = time.time()
    for line in fr:
        en_sent = line.strip().split("\t")[0]
        references.append([line.strip().split("\t")[1]])
        if len(batch) < batch_size:
            batch.append(en_sent)
        else:
            translated_batch = translate(batch)
            for en, hi in zip(batch, translated_batch):
                predictions.append(hi)
                fw_hi.write(";".join([en, hi]) + "\n")
                i += 1
            fw_hi.flush()
            batch = [en_sent]
            if i % (batch_size*100) == 0:
                total_time = time.time() - st_time
                print("Total time for", (batch_size), "examples:", total_time)
                print("Time per example:", total_time/(batch_size))
                print("Time number of sentences:", i, flush=True)
                st_time = time.time()
    if len(batch) > 0:
        translated_batch = translate(batch)
        for en, hi in zip(batch, translated_batch):
            predictions.append(hi)
            fw_hi.write(";".join([en, hi]) + "\n")
            i += 1

print("Computing BLEU")
#print(len(predictions), len(references))
assert len(predictions) == len(references)
result = metric.compute(predictions=predictions, references=references)
print(result["score"])
