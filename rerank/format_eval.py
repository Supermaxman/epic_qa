
from collections import defaultdict
import json
import os
import sys
import argparse


def write_results(question_id, question_scores, run_name, f, top_k=1000, multiple_per_doc=True, allow_overlap=False, threshold=None):
	num_top = 0
	rel_idx = 0
	seen_docs = set()
	seen_sentences = set()
	while num_top < top_k and rel_idx < len(question_scores):
		answer_id, doc_id, pass_id, sent_start_idx, sent_end_idx, score = question_scores[rel_idx]
		sent_ids = [(doc_id, pass_id, sent_id) for sent_id in range(sent_start_idx, sent_end_idx + 1)]
		if (multiple_per_doc or doc_id not in seen_docs) \
				and (allow_overlap or all([x not in seen_sentences for x in sent_ids]))\
				and (threshold is None or score > threshold):
			f.write(f'{question_id}\tQ0\t{answer_id}\t{num_top + 1}\t{score}\t{run_name}\n')
			num_top += 1
			seen_docs.add(doc_id)
			for x in sent_ids:
				seen_sentences.add(x)
		rel_idx += 1


def read_scores(run_path):
	rerank_scores = defaultdict(list)
	with open(run_path) as f:
		for line in f:
			# f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n'
			line = line.strip().split()
			if len(line) == 6:
				question_id, _, doc_pass_sent_id, rank, score, _ = line
				start_id, end_id = doc_pass_sent_id.split(':')
				doc_id, pass_id, sent_start_id = start_id.split('-')
				_, _, sent_end_id = end_id.split('-')
				sent_start_idx = int(sent_start_id[1:])
				sent_end_idx = int(sent_end_id[1:])
				score = float(score)
				rerank_scores[question_id].append((doc_pass_sent_id, doc_id, pass_id, sent_start_idx, sent_end_idx, score))
	return rerank_scores


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--pred_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-k', '--top_k', default=2000, type=int)
	parser.add_argument('-sd', '--single_per_doc', default=False, action='store_true')
	parser.add_argument('-ao', '--allow_overlap', default=False, action='store_true')
	parser.add_argument('-t', '--threshold', default=None, type=float)

	args = parser.parse_args()

	pred_path = args.pred_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	allow_overlap = args.allow_overlap
	single_per_doc = args.single_per_doc
	top_k = args.top_k
	threshold = args.threshold

	rerank_scores = read_scores(pred_path)
	with open(output_path, 'w') as f:
		for question_id, question_scores in rerank_scores.items():
			write_results(
				question_id,
				question_scores,
				output_name,
				f,
				top_k=top_k,
				multiple_per_doc=not single_per_doc,
				allow_overlap=allow_overlap,
				threshold=threshold
			)



