import torch
import argparse
from collections import defaultdict
import os
from tqdm import tqdm
import json


def get_id(q_id):
	answer_id, sample_id = q_id.split('|')
	sample_id = int(sample_id)
	return answer_id, sample_id


def load_predictions(input_path):
	pred_list = []
	for file_name in os.listdir(input_path):
		if file_name.endswith('.question'):
			preds = torch.load(os.path.join(input_path, file_name))
			pred_list.extend(preds)
	sample_entail_pairs = defaultdict(lambda: defaultdict(list))
	probs = []
	for prediction in tqdm(pred_list):
		answer_a_id, entailed_set_id = get_id(prediction['question_a_id'])
		question_id = prediction['question_b_id']
		entail_prob = prediction['entail_prob']
		probs.append(entail_prob)
		sample_entail_pairs[question_id][answer_a_id].append((entailed_set_id, entail_prob))
	print(f'min={min(probs)}')
	print(f'max={max(probs)}')
	return sample_entail_pairs


def save_predictions(sample_entail_pairs, output_path):
	with open(output_path, 'w') as f:
		json.dump(sample_entail_pairs, f, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path

	sample_entail_pairs = load_predictions(input_path)
	save_predictions(sample_entail_pairs, output_path)
