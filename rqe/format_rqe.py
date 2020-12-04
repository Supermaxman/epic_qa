import torch
import argparse
from collections import defaultdict
import os
from tqdm import tqdm


def load_predictions(model_path):
	pred_list = []
	for file_name in os.listdir(model_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(model_path, file_name))
			pred_list.extend(preds)
	query_answer_sample_probs = defaultdict(list)
	probs = []
	for prediction in tqdm(pred_list):
		answer_id = prediction['id']
		question_id = prediction['question_id']
		sample_id = prediction['sample_id']
		entail_prob = prediction['entail_prob']
		query_answer_sample_probs[(question_id, answer_id)].append((sample_id, entail_prob))
		probs.append(entail_prob)
	print(f'min={min(probs)}')
	print(f'max={max(probs)}')
	return query_answer_sample_probs


def save_predictions(query_answer_sample_probs, output_path):
	with open(output_path, 'w') as f:
		for (question_id, answer_id), qas_probs in query_answer_sample_probs.items():
			for sample_id, entail_prob in qas_probs:
				f.write(f'{question_id}\t{answer_id}\t{sample_id}\t{entail_prob:.8f}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	model_path = args.model_path
	output_path = args.output_path

	query_answer_sample_probs = load_predictions(model_path)
	save_predictions(query_answer_sample_probs, output_path)
