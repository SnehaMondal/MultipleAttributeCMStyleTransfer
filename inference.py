import time
import torch
import os

from transformers import MT5ForConditionalGeneration, T5Tokenizer

model_checkpoint="models/mt5_tagged_3_bins_ft2"
input_filepath="inputs.txt"
output_filepath="outputs.txt"

print(f"Model name : {model_checkpoint}")
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.eval()
model.to(device)

batch_size = 256
beam_width = 1
fw_hi = open(output_filepath, "w")

def generate(sentences):
    batch_encoding = tokenizer(sentences, return_tensors="pt", padding=True)
    for k,v in batch_encoding.items():
        batch_encoding[k] = v.to(device)
    outputs = model.generate(**batch_encoding, max_length=256, num_beams=beam_width, num_return_sequences=beam_width, return_dict_in_generate=True)
    decoded_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs['sequences']]
    return [decoded_sentences[i:i+beam_width] for i in range(0, len(decoded_sentences), beam_width)]

predictions = []
references = []
with open(input_filepath) as fr:
    i = 0
    batch = []
    st_time = time.time()
    for line in fr:
        sentence = line.strip()
        if len(batch) < batch_size:
            batch.append(sentence)
        else:
            translated_batch = generate(batch)
            for en, hi_list in zip(batch, translated_batch):
                for hi in hi_list:
                    fw_hi.write(hi + "\n")
                i += 1
                fw_hi.flush()
            batch = [sentence]
            if i % (batch_size*100) == 0:
                total_time = time.time() - st_time
                print("Total time for", (batch_size*100), "examples:", total_time)
                print("Time per example:", total_time/(batch_size*100))
                print("Total number of sentences:", i, flush=True)
                st_time = time.time()
    if len(batch) > 0:
        translated_batch = generate(batch)
        for en, hi_list in zip(batch, translated_batch):
            for hi in hi_list:
                fw_hi.write(hi + "\n")
            i += 1
fw_hi.close()

print("Done")
