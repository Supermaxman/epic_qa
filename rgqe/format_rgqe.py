import torch
import argparse
from collections import defaultdict
import os
from tqdm import tqdm
import json


def get_id(q_id):
	question_id, answer_id, sample_id = q_id.split('|')
	sample_id = int(sample_id)
	return question_id, answer_id, sample_id


def load_predictions(model_path, threshold):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.ptg'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	sample_entail_pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	probs = []
	for prediction in tqdm(pred_list):
		question_id, answer_a_id, sample_a_id = get_id(prediction['question_a_id'])
		_, answer_b_id, sample_b_id = get_id(prediction['question_b_id'])
		entail_prob = prediction['entail_prob']
		probs.append(entail_prob)
		if entail_prob < threshold:
			continue
		sample_entail_pairs[question_id][answer_a_id][answer_b_id].append((sample_a_id, sample_b_id))
	print(f'min={min(probs)}')
	print(f'max={max(probs)}')
	return sample_entail_pairs


def save_predictions(sample_entail_pairs, output_path):
	with open(output_path, 'w') as f:
		json.dump(sample_entail_pairs, f, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)
	args = parser.parse_args()

	model_path = args.model_path
	output_path = args.output_path
	threshold = args.threshold

	sample_entail_pairs = load_predictions(model_path, threshold)
	save_predictions(sample_entail_pairs, output_path)
