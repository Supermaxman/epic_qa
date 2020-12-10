
from collections import defaultdict
import json
import os
from tqdm import tqdm
import argparse


def parse_id(doc_pass_sent_id):
	start_id, end_id = doc_pass_sent_id.split(':')
	id_list = start_id.split('-')
	doc_id = '-'.join(id_list[:-2])
	pass_id = id_list[-2]
	sent_start_id = id_list[-1]
	sent_end_id = end_id.split('-')[-1]
	return doc_id, pass_id, sent_start_id, sent_end_id


def read_scores(search_path, collection_path):
	doc_ids = defaultdict(lambda: defaultdict(set))
	rerank_scores = defaultdict(list)
	with open(search_path) as f:
		for line in f:
			line = line.strip().split()
			if len(line) == 6:
				question_id, _, answer_id, rank, score, run_name = line
				doc_id, pass_id, sent_start_id, sent_end_id = parse_id(answer_id)
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
	for doc_id, doc_pass_ids in tqdm(doc_ids.items()):
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


