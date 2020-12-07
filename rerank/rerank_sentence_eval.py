
from collections import defaultdict
import json
import os
import sys
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-l', '--label_path', required=True)
	parser.add_argument('-k', '--top_k', default=None, type=int)

	args = parser.parse_args()

	input_path = args.input_path
	label_path = args.label_path
	top_k = args.top_k

	labels = defaultdict(set)
	with open(label_path, 'r') as f:
		questions = json.load(f)
	for question in questions:
		question_id = question['question_id']
		for annotation in question['annotations']:
			sentence_id = annotation['sentence_id']
			nugget_count = len(annotation['nugget_ids'])
			labels[question_id].add(sentence_id)

	passage_qrels = defaultdict(set)
	with open(input_path, 'r') as f:
		for line in f:
			line = line.strip().split()
			if line:
				question_id, _, answer_id, dq_rank, dq_score, dq_run = line
				start_id, end_id = answer_id.split(':')
				doc_id, pass_id, start_sent_id = start_id.split('-')
				_, _, end_sent_id = end_id.split('-')
				start_sent_idx = int(start_sent_id[1:])
				end_sent_idx = int(end_sent_id[1:])
				for sent_idx in range(start_sent_idx, end_sent_idx+1):
					passage_qrels[question_id].add(f'{doc_id}-{pass_id}-S{sent_idx:03d}')

	for question_id, q_labels in labels.items():
		if question_id not in passage_qrels:
			continue
		q_passage_qrels = passage_qrels[question_id]
		num_found = len(q_labels.intersection(q_passage_qrels))
		total_count = len(q_labels)
		percent_found = num_found / total_count
		print(f'{question_id}: %found={percent_found:.2f}')
