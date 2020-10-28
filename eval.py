import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os
import numpy as np
from sklearn.metrics import average_precision_score


parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query_path', required=True)
parser.add_argument('-r', '--run_path', required=True)
parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')
parser.add_argument('-db', '--debug', default=False, action='store_true')

args = parser.parse_args()

query_path = args.query_path
run_path = args.run_path
label_path = args.label_path
debug = args.debug

with open(query_path) as f:
	queries_list = json.load(f)
	queries = {q['question_id']: q for q in queries_list}
	q_id_lookup = {q_idx: q_id for q_idx, q_id in enumerate(queries, start=1)}
	q_lookup_id = {q_id: q_idx for q_idx, q_id in enumerate(queries, start=1)}

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
				'rank': int(rank),
				'score': -float(score),
				'run_name': run_name
			}
			if question_id in labels:
				qrels[question_id].append(rel)

total_tp = 0.0
total_fp = 0.0
total_fn = 0.0

query_aps = {}
for question_id, question_labels in labels.items():
	q_idx = q_lookup_id[question_id]
	if debug and q_idx != 1:
		continue
	question_rels = qrels[question_id]
	label_sentences = set([sent['sentence_id'] for sent in question_labels['annotations']])
	pred_sentences = set()
	pred_scores = []
	pred_labels = []
	nrof_found_true = 0
	for rel in question_rels:
		for sent_id in range(rel['sent_start_id'], rel['sent_end_id'] + 1):
			doc_pass_sent_id = f'{rel["doc_id"]}-C{rel["pass_id"]:03d}-S{sent_id:03d}'
			pred_sentences.add(doc_pass_sent_id)
			pred_scores.append(rel['score'])
			pred_label = 1 if doc_pass_sent_id in label_sentences else 0
			pred_labels.append(pred_label)
			nrof_found_true += pred_label
	nrof_true = len(label_sentences)
	try:
		ap = average_precision_score(
			y_true=np.array(pred_labels),
			y_score=np.array(pred_scores),
			average='macro'
		)
		# adjust ap for total number of true documents which can never contribute regardless of threshold
		ap = (ap * nrof_found_true) / nrof_true
	except IndexError:
		ap = 0.0
	if np.isnan(ap):
		ap = 0.0
	query_aps[question_id] = ap

for question_id, question_ap in query_aps.items():
	print(f'{question_id}: AP={question_ap:.3f}')

map = np.mean([value for key, value in query_aps.items()])
print(f'MAP: {map:.3f}')
