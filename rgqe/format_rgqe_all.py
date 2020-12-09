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


def load_predictions(model_path):
	sample_entail_pairs = defaultdict(list)
	for file_name in tqdm([x for x in os.listdir(model_path) if x.endswith('.all')]):
		preds = torch.load(os.path.join(model_path, file_name))
		for prediction in preds:
			answer_a_id, entailed_set_a_id = get_id(prediction['question_a_id'])
			entailed_set_b_id = prediction['question_b_id']
			entail_prob = prediction['entail_prob']
			if entail_prob < 0.5:
				continue
			sample_entail_pairs[answer_a_id].append(
				(entailed_set_a_id, entailed_set_b_id, entail_prob)
			)

	return sample_entail_pairs


def save_predictions(sample_entail_pairs, output_path):
	with open(output_path, 'w') as f:
		json.dump(sample_entail_pairs, f, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	model_path = args.model_path
	output_path = args.output_path

	sample_entail_pairs = load_predictions(model_path)
	save_predictions(sample_entail_pairs, output_path)