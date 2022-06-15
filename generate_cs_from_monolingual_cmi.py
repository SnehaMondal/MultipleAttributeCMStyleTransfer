import time
import torch
import os

from transformers import T5Tokenizer
from model import MT5ForStyleConditionalGeneration

model_checkpoint="models/translation_multitask_cmi_vec"
input_filepath="data/OpenSubtitles/OpenSubtitles.en-hi-cmi.hi"
output_filepath="data/OpenSubtitles/OpenSubtitles.en-hi-cmi.cs"

print(f"Model name : {model_checkpoint}")
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = MT5ForStyleConditionalGeneration.from_pretrained(model_checkpoint, num_attr=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.eval()
model.to(device)

batch_size = 512
fw_hi = open(output_filepath, "w")
task_prefix = "to_cm "

def generate(sentences, cmi_scores):
    batch_encoding = tokenizer(sentences, return_tensors="pt", padding=True)
    batch_encoding["input_style_scores"] = torch.transpose(torch.tensor([cmi_scores], dtype=torch.float32), 0, 1)
    for k,v in batch_encoding.items():
        batch_encoding[k] = v.to(device)
    outputs = model.generate(**batch_encoding, max_length=256)
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]

predictions = []
references = []
with open(input_filepath) as fr:
    i = 0
    batch = []
    batch_cmi = []
    st_time = time.time()
    for line in fr:
        components = line.strip().split('\t')
        if len(batch) < batch_size:
            batch.append(task_prefix + " ".join(components[0:-1]))
            batch_cmi.append(float(components[-1]))
        else:
            translated_batch = generate(batch, batch_cmi)
            for en, hi in zip(batch, translated_batch):
                fw_hi.write(en + "\t" + hi + "\n")
                i += 1
                fw_hi.flush()
            batch = [task_prefix + " ".join(components[0:-1])]
            batch_cmi = [float(components[-1])]
            if i % (batch_size*100) == 0:
                total_time = time.time() - st_time
                print("Total time for", (batch_size*100), "examples:", total_time)
                print("Time per example:", total_time/(batch_size*100))
                print("Total number of sentences:", i, flush=True)
                st_time = time.time()
    if len(batch) > 0:
        translated_batch = generate(batch, batch_cmi)
        for en, hi in zip(batch, translated_batch):
            fw_hi.write(en + "\t" + hi + "\n")
            i += 1
fw_hi.close()

print("Done")