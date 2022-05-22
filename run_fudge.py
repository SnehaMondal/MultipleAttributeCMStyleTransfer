import time
import torch
import os

from transformers import T5Tokenizer, T5Config, AutoTokenizer, XLMRobertaModelWithHeads, BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList, AdapterConfig
import numpy as np
from datasets import load_metric
from torch import nn
from transformers.adapters.composition import Stack
import argparse
import time
import pprint

from model import MT5ForStyleConditionalGeneration
import metrics as mt

parser = argparse.ArgumentParser()
parser.add_argument('--input_filename', type=str, required=True)
parser.add_argument('--output_directory', type=str, required=True)
parser.add_argument('--path_to_cmgen_model', type=str, required=True)
parser.add_argument('--path_to_predictor', type=str, required=True)
parser.add_argument('--beam_width', type=int, required=True)
args = parser.parse_args()

print(args.input_filename)
print(args.output_directory)
print(args.path_to_predictor)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
metric = load_metric("sacrebleu")

#load codemixed generation model
t5_tokenizer = T5Tokenizer.from_pretrained(args.path_to_cmgen_model)
model = MT5ForStyleConditionalGeneration.from_pretrained(args.path_to_cmgen_model, return_dict=True).to(device)
model.eval()
print("Loaded codemixed generation model")

#load predictor model
xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
conditioning_model = XLMRobertaModelWithHeads.from_pretrained(args.path_to_predictor)

if args.path_to_predictor == 'xlm-roberta-base':
	lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
	conditioning_model.load_adapter("hi/wiki@ukp", config=lang_adapter_config)
	config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=16)
	conditioning_model.load_adapter("/home/snehamondal/fudge-controlled-generation-creative/xlm-roberta-base_formality_classify_gyafc_pfeiffer", config=config)

conditioning_model.active_adapters = Stack("hi", "gyafc")
conditioning_model.to(device)
conditioning_model.eval()
print(f"Loaded prediction model from {conditioning_model.config._name_or_path}")

def fudge_beam_search(
	input_ids,
	beam_scorer,
	condition_lambda,
	input_cmi_scores=None,
	logits_processor = None,
	stopping_criteria = None,
	max_length = None,
	pad_token_id = None,
	eos_token_id = None,
	output_attentions = None,
	output_hidden_states = None,
	output_scores = None,
	return_dict_in_generate = False,
	synced_gpus = False,
	**model_kwargs,
	):

	logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
	pad_token_id = model.config.pad_token_id
	eos_token_id = model.config.eos_token_id
	output_scores = model.config.output_scores
	output_attentions = model.config.output_attentions
	output_hidden_states = (model.config.output_hidden_states)
	return_dict_in_generate = (model.config.return_dict_in_generate)

	effective_vocab_size = 1000
	batch_size = len(beam_scorer._beam_hyps)
	num_beams = beam_scorer.num_beams

	batch_beam_size, cur_len = input_ids.shape

	if num_beams * batch_size != batch_beam_size:
		raise ValueError(
			f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
		)

	scores = () if (return_dict_in_generate and output_scores) else None
	beam_indices = (
		tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
	)
	decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
	cross_attentions = () if (return_dict_in_generate and output_attentions) else None
	decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

	beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
	beam_scores[:, 1:] = -1e9
	beam_scores = beam_scores.view((batch_size * num_beams,))

	this_peer_finished = False  # used by synced_gpus only
	while True:

		model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, input_cmi_scores=input_cmi_scores, **model_kwargs)

		outputs = model(
			**model_inputs,
			return_dict=True,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
		)

		next_token_logits = outputs.logits[:, -1, :]
		next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)

		top_logits, top_indices = next_token_logits.topk(effective_vocab_size, dim=1)
		tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, effective_vocab_size, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:]

		if condition_lambda == 0.0:
			condition_logits = torch.zeros_like(top_logits).float().to(device)
		else:
			partial_prefixes = [t5_tokenizer.decode(t, skip_special_tokens=True).strip() for t in
								tplus1_candidates.view(num_beams * effective_vocab_size, tplus1_candidates.shape[-1])]
			assert len(partial_prefixes) == num_beams * effective_vocab_size

			##batch partial prefixes before evaluating
			batch = []
			condition_logits = []
			formal_batch_size = 2048
			for i in range(len(partial_prefixes)):
				prefix = partial_prefixes[i]
				if len(batch) < formal_batch_size:
					batch.append(prefix)
				else:
					curr_batch = xlm_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
					for k, v in curr_batch.items():
						curr_batch[k] = v.to(device)
					condition_logits += conditioning_model(**curr_batch).logits[:, 0]
					batch = [prefix]
			if len(batch) > 0:
				curr_batch = xlm_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
				for k, v in curr_batch.items():
					curr_batch[k] = v.to(device)
				condition_logits += conditioning_model(**curr_batch).logits[:, 1]
			condition_logits = torch.stack(condition_logits, dim=0).view(num_beams, effective_vocab_size)

		full_logits = top_logits + condition_lambda * condition_logits

		next_token_scores = nn.functional.log_softmax(
			full_logits, dim=-1
		)  # (batch_size * num_beams, effective_vocab_size)

		next_token_scores_processed = logits_processor(input_ids, next_token_scores)
		next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

		next_token_scores = next_token_scores.view(batch_size, num_beams * effective_vocab_size)
		next_token_scores, next_tokens = torch.topk(
			next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
		)
		
		row_indices = next_tokens // effective_vocab_size
		column_indices = next_tokens % effective_vocab_size
		next_tokens = []
		for i, j in zip(row_indices.tolist()[0], column_indices.tolist()[0]):
			next_tokens.append(top_indices[i][j].item())
		next_tokens = torch.tensor(next_tokens, dtype=torch.int).view(batch_size, 2 * num_beams)

		# stateless
		beam_outputs = beam_scorer.process(
			input_ids,
			next_token_scores,
			next_tokens=next_tokens,
			next_indices=row_indices,
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
		)

		beam_scores = beam_outputs["next_beam_scores"]
		beam_next_tokens = beam_outputs["next_beam_tokens"]
		beam_idx = beam_outputs["next_beam_indices"]

		input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
		current_hypothesis = [t5_tokenizer.decode(t, skip_special_tokens=True).strip() for t in input_ids]

		model_kwargs = model._update_model_kwargs_for_generation(
			outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
		)
		if model_kwargs["past"] is not None:
			model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

		# increase cur_len
		cur_len = cur_len + 1

		if beam_scorer.is_done or stopping_criteria(input_ids, scores):
			if not synced_gpus:
				break
			else:
				this_peer_finished = True

	sequence_outputs = beam_scorer.finalize(
		input_ids,
		beam_scores,
		next_tokens,
		row_indices,
		pad_token_id=pad_token_id,
		eos_token_id=eos_token_id,
		max_length=stopping_criteria.max_length,
	)
	return sequence_outputs["sequences"]

def beam_search(sentence, cmi_score, condition_lambda, num_beams):
	with torch.no_grad():        
		encoder_input_ids = t5_tokenizer([sentence], return_tensors="pt").input_ids.to(device)
		input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
		input_ids = input_ids * model.config.decoder_start_token_id
		input_cmi_scores = torch.tensor([cmi_score], dtype=torch.float32, device=model.device)
		model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)}
		beam_scorer = BeamSearchScorer(batch_size=1, num_beams=num_beams, device=model.device)
		stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=model.config.max_length)])
		outputs = fudge_beam_search(input_ids=input_ids, input_cmi_scores=input_cmi_scores, beam_scorer=beam_scorer, stopping_criteria=stopping_criteria, condition_lambda=condition_lambda, **model_kwargs)
		return [t5_tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]


input_texts = []
references = []
cmi_scores = []
task_prefix = "to_cm "
with open(args.input_filename, "r") as f:
	for line in f.readlines()[:250]:
		components = line.strip().split('\t')
		input_texts.append(task_prefix + components[0])
		references.append(components[1])
		cmi_scores.append(float(components[2]))
assert len(references) == len(input_texts)


bleu_dict={}
for cl in [3.0, 4.0, 5.0]:
	print(f"Running beam search with cl : {cl}", flush=True)
	output_file = f"{args.output_directory}/generated_predictions_lambda_{cl}_beam_{args.beam_width}.tsv"
	predictions = []
	st_time = time.time()
	for i in range(len(input_texts)):
		prediction = beam_search(input_texts[i], cmi_scores[i], condition_lambda=cl, num_beams=args.beam_width)
		predictions.append(prediction[0])
		print(f"Evaluated input {i}", flush=True)
	total_time = time.time() - st_time
	print(f"Time taken for lambda {cl} : {total_time/60} minutes")
	
	with open(output_file, "w") as f:
		print(f"Writing predictions to : {output_file}")
		for input_text, prediction in zip(input_texts, predictions):
			f.write(";".join([input_text, prediction]))
			f.write("\n")
			
	bleu = mt.bleu(targets=references, predictions=predictions)
	result = {"bleu": bleu["bleu"]}

	cmi_acc = mt.cmi_bucket_accuracy(targets=references, predictions=predictions)
	result["cmi_acc"] = cmi_acc["cmi_bucket_accuracy"]

	cmi_corr = mt.cmi_correlation(targets=references, predictions=predictions)
	result["cmi_corr"] = cmi_corr["cmi_correlation"]

	cmi_bleu_hm = mt.cmi_acc_bleu_hm(targets=references, predictions=predictions)
	result["cmi_bleu_hm"] = cmi_bleu_hm["cmi_acc_bleu_hm"]

	result = {k: round(v, 4) for k, v in result.items()}
	pprint.pprint(result)

print("Done.")