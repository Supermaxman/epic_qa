
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


def read_scores(search_path, collection_path):
	doc_ids = defaultdict(lambda: defaultdict(set))
	rerank_scores = defaultdict(list)
	with open(search_path) as f:
		for line in f:
			line = line.strip().split()
			if len(line) == 6:
				question_id, _, answer_id, rank, score, run_name = line
				start_id, end_id = answer_id.split(':')
				doc_id, pass_id, sent_start_id = start_id.split('-')
				_, _, sent_end_id = end_id.split('-')
				sent_start_idx = int(sent_start_id[1:])
				sent_end_idx = int(sent_end_id[1:])
				score = float(score)
				answer = {
					'answer_id': answer_id,
					'doc_id': doc_id,
					'pass_id': pass_id,
					'sent_start_id': sent_start_id,
					'sent_end_id': sent_end_id,
					'sent_start_idx': sent_start_idx,
					'sent_end_idx': sent_end_idx,
					'score': score,
					'rank': int(rank),
					'run_name': run_name
				}
				rerank_scores[question_id].append(
					answer
				)
				doc_ids[doc_id][pass_id].add((sent_start_idx, sent_end_idx))
	answer_lookup = {}
	for doc_id, doc_pass_ids in doc_ids.items():
		doc_path = os.path.join(collection_path, f'{doc_id}.json')
		with open(doc_path) as f:
			doc = json.load(f)
		for pass_id, sent_spans in doc_pass_ids.items():
			pass_idx = int(pass_id[1:])
			for sent_start_idx, sent_end_idx in sent_spans:
				passage = doc['contexts'][pass_idx]
				sentences = passage['sentences'][sent_start_idx:sent_end_idx + 1]
				text = passage['text'][sentences[0]['start']:sentences[-1]['end']]
				answer_id = f'{sentences[0]["sentence_id"]}:{sentences[-1]["sentence_id"]}'
				answer_lookup[answer_id] = text
	results = {
		'rerank_scores': rerank_scores,
		'answer_text_lookup': answer_lookup
	}
	return results


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--search_path', required=True)
	parser.add_argument('-c', '--collection_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	search_path = args.search_path
	collection_path = args.collection_path
	output_path = args.output_path

	results = read_scores(search_path, collection_path)

	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)


