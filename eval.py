import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os


parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query_path', required=True)
parser.add_argument('-r', '--run_path', required=True)
parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')

args = parser.parse_args()

query_path = args.query_path
run_path = args.run_path
label_path = args.label_path

with open(query_path) as f:
	queries = json.load(f)

labels = {}
with open(label_path) as f:
	questions = json.load(f)
	for question in questions:
		labels[question['question_id']] = question


qrels = defaultdict(list)
with open(run_path) as f:
	for line in f:
		line = line.strip().split()
		if line:
			query_id, _, doc_pass_sent_id, rank, score, run_name = line
			query_id = int(query_id)
			doc_id, pass_id, sent_start_id, sent_end_id = doc_pass_sent_id.split('-')
			question_id = queries[query_id]['question_id']
			rel = {
				'doc_id': doc_id,
				'pass_id': pass_id,
				'sent_start_id': int(sent_start_id[1:]),
				'sent_end_id': int(sent_end_id[1:]),
				'id': doc_pass_sent_id,
				'rank': rank,
				'score': score,
				'run_name': run_name
			}
			if question_id in labels:
				qrels[question_id].append(rel)

total_tp = 0.0
total_fp = 0.0
total_fn = 0.0

results = {}
for question_id, question_labels in labels.items():
	question_rels = qrels[question_id]
	label_sentences = set([sent['sentence_id'] for sent in question_labels['annotations']])
	pred_sentences = set()
	for rel in question_rels:
		for sent_id in range(rel['sent_start_id'], rel['sent_end_id'] + 1):
			pred_sentences.add(f'{rel["doc_id"]}-{rel["pass_id"]}-S{sent_id:03d}')

	overlap = pred_sentences.intersection(label_sentences)
	# overlap of pred and true is tp
	tp = len(overlap)
	# subtract tp from pred for fp
	fp = len(pred_sentences - overlap)
	# subtract tp from labels for fn
	fn = len(label_sentences - overlap)

	precision = tp / (tp + fp)

	recall = tp / (tp + fn)

	f1 = 2.0 * ((precision * recall) / (precision + recall))
	results[question_id] = {
		'precision': precision,
		'recall': recall,
		'f1': f1
	}
	total_tp += tp
	total_fp += fp
	total_fn += fn

for question_id, result in results.items():
	print(f'{question_id}: P={result["precision"]:.3f}, R={result["recall"]:.3f}, F1={result["f1"]:.3f}')

total_precision = total_tp / (total_tp + total_fp)
total_recall = total_tp / (total_tp + total_fn)
total_f1 = 2.0 * ((total_precision * total_recall) / (total_precision + total_recall))

print(f'TOTAL Micro: P={total_precision:.3f}, R={total_recall:.3f}, F1={total_f1:.3f}')
