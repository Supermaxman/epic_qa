import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os


parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query_path', required=True)
parser.add_argument('-r', '--run_path', required=True)
parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')
parser.add_argument('-k', '--top_k', default=100, type=int)

args = parser.parse_args()

query_path = args.query_path
run_path = args.run_path
label_path = args.label_path
top_k = args.top_k

with open(query_path) as f:
	queries_list = json.load(f)
	queries = {q['question_id']: q for q in queries_list}
	q_id_lookup = {q_idx: q_id for q_idx, q_id in enumerate(queries, start=1)}

labels = {}
with open(label_path) as f:
	questions = json.load(f)
	for question in questions:
		question_id = question['question_id']
		if question_id in queries:
			labels[question_id] = question


qrels = defaultdict(list)
with open(run_path) as f:
	for line in f:
		line = line.strip().split()
		if line:
			query_id, _, doc_pass_sent_id, rank, score, run_name = line
			query_id = int(query_id)
			doc_id, pass_id, sent_start_id, sent_end_id = doc_pass_sent_id.split('-')
			question_id = q_id_lookup[query_id]
			rel = {
				'doc_id': doc_id,
				'pass_id': int(pass_id),
				'sent_start_id': int(sent_start_id),
				'sent_end_id': int(sent_end_id),
				'id': doc_pass_sent_id,
				'rank': rank,
				'score': score,
				'run_name': run_name
			}
			if question_id in labels:
				if len(qrels[question_id]) < top_k:
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
			pred_sentences.add(f'{rel["doc_id"]}-C{rel["pass_id"]:03d}-S{sent_id:03d}')

	overlap = pred_sentences.intersection(label_sentences)
	# overlap of pred and true is tp
	tp = len(overlap)
	# subtract tp from pred for fp
	fp = len(pred_sentences - overlap)
	# subtract tp from labels for fn
	fn = len(label_sentences - overlap)

	precision = tp / (max(tp + fp, 1))

	recall = tp / (max(tp + fn, 1))

	f1 = 2.0 * ((precision * recall) / (max(precision + recall, 1)))
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

total_precision = total_tp / (max(total_tp + total_fp, 1))
total_recall = total_tp / (max(total_tp + total_fn, 1))
total_f1 = 2.0 * ((total_precision * total_recall) / (max(total_precision + total_recall, 1)))

print(f'TOTAL Micro: P={total_precision:.3f}, R={total_recall:.3f}, F1={total_f1:.3f}')
