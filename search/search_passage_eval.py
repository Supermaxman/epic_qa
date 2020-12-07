
from collections import defaultdict
import json
import os
import sys
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-l', '--label_path', required=True)

	args = parser.parse_args()

	input_path = args.input_path
	label_path = args.label_path

	labels = defaultdict(set)
	with open(label_path, 'r') as f:
		questions = json.load(f)
	for question in questions:
		question_id = question['question_id']
		for annotation in question['annotations']:
			sentence_id = annotation['sentence_id']
			nugget_count = len(annotation['nugget_ids'])
			doc_pass_id = '-'.join(sentence_id.split('-')[:2])
			labels[question_id].add(doc_pass_id)

	passage_qrels = defaultdict(set)
	with open(input_path, 'r') as f:
		for line in f:
			line = line.strip().split()
			if line:
				question_id, _, doc_pass_id, dq_rank, dq_score, dq_run = line
				passage_qrels[question_id].add(doc_pass_id)

	for question_id, q_labels in labels.items():
		if question_id not in passage_qrels:
			continue
		q_passage_qrels = passage_qrels[question_id]
		num_found = len(q_labels.intersection(q_passage_qrels))
		total_count = len(q_labels)
		percent_found = num_found / total_count
		print(f'{question_id}: %found={percent_found:.2f}')
