import torch
import argparse
from collections import defaultdict
import os
from tqdm import tqdm
import json


def load_predictions(model_path, answer_samples, threshold, num_samples):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	query_answer_sample_probs = defaultdict(lambda: defaultdict(list))
	probs = []
	seen_answers = set()
	for prediction in tqdm(pred_list):
		answer_id = prediction['id']
		seen_answers.add(answer_id)
		question_id = prediction['question_id']
		sample_id = prediction['sample_id']
		entail_prob = prediction['entail_prob']
		if entail_prob < threshold:
			continue
		sample_text = answer_samples[answer_id][sample_id]
		sample_data = {
			'sample_id': sample_id,
			'entail_prob': entail_prob,
			'sample_text': sample_text
		}
		query_answer_sample_probs[question_id][answer_id].append(sample_data)
		probs.append(entail_prob)
	print(f'min={min(probs)}')
	print(f'max={max(probs)}')

	kept_answers = set()
	num_examples = 0
	filtered_query_answer_sample_probs = defaultdict(lambda: defaultdict(list))
	for question_id, q_answer_samples in query_answer_sample_probs.items():
		for answer_id, q_a_samples in q_answer_samples.items():
			sorted_samples = sorted(q_a_samples, key=lambda x: x['entail_prob'], reverse=True)
			filtered_q_a_samples = sorted_samples[:num_samples]
			filtered_query_answer_sample_probs[question_id][answer_id] = filtered_q_a_samples
			num_examples += len(filtered_q_a_samples)
			kept_answers.add(answer_id)

	print(f'num_examples={num_examples}')
	print(f'%kept={len(kept_answers)/len(seen_answers):.2f}')
	return filtered_query_answer_sample_probs


def save_predictions(query_answer_sample_probs, output_path):
	with open(output_path, 'w') as f:
		json.dump(query_answer_sample_probs, f, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-e', '--expand_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)
	parser.add_argument('-k', '--num_samples', default=3, type=int)
	args = parser.parse_args()

	model_path = args.model_path
	expand_path = args.expand_path
	output_path = args.output_path
	threshold = args.threshold
	num_samples = args.num_samples

	with open(expand_path, 'r') as f:
		answer_samples = json.load(f)

	query_answer_sample_probs = load_predictions(model_path, answer_samples, threshold, num_samples)
	save_predictions(query_answer_sample_probs, output_path)
