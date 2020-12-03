from collections import defaultdict
import json
import os
import sys
import argparse


def format_answer_id(doc_id, pass_id, sent_id):
	return f'{doc_id}-C{pass_id:03}-S{sent_id:03}'


def format_answer_span_id(doc_id, pass_id, sent_start_id, sent_end_id):
	start_id = format_answer_id(doc_id, pass_id, sent_start_id)
	end_id = format_answer_id(doc_id, pass_id, sent_end_id)
	return f'{start_id}:{end_id}'


def write_results(question_id, question_scores, run_name, f, top_k=1000, multiple_per_doc=True, allow_overlap=False):
	num_top = 0
	rel_idx = 0
	seen_docs = set()
	seen_sentences = set()
	while num_top < top_k and rel_idx < len(question_scores):
		doc_id, pass_id, sent_start_id, sent_end_id, score = question_scores[rel_idx]
		sent_ids = [(doc_id, pass_id, sent_id) for sent_id in range(sent_start_id, sent_end_id + 1)]
		if (multiple_per_doc or doc_id not in seen_docs) and (
				allow_overlap or all([x not in seen_sentences for x in sent_ids])):
			answer_id = format_answer_span_id(doc_id, pass_id, sent_start_id, sent_end_id)
			f.write(f'{question_id}\tQ0\t{answer_id}\t{num_top + 1}\t{score}\t{run_name}\n')
			num_top += 1
			seen_docs.add(doc_id)
			for x in sent_ids:
				seen_sentences.add(x)
		rel_idx += 1


def read_scores(run_path):
	rerank_scores = defaultdict(list)
	# TODO open pickle format by output
	with open(run_path) as f:
		for line in f:
			# {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
			line = line.strip().split()
			if len(line) == 6:
				question_id, _, doc_pass_sent_id, rank, score, _ = line
				ids = doc_pass_sent_id.split('-')
				doc_id, pass_id = ids[0], ids[1]
				pass_id = int(pass_id)
				if len(ids) == 3:
					sent_start_id, sent_end_id = ids[2], ids[2]
				elif len(ids) == 4:
					sent_start_id, sent_end_id = ids[2], ids[3]
				else:
					sent_start_id, sent_end_id = ids[2], ids[4]
				sent_start_id = int(sent_start_id)
				sent_end_id = int(sent_end_id)
				pass_id = int(pass_id)
				score = float(score)
				rerank_scores[question_id].append((doc_id, pass_id, sent_start_id, sent_end_id, score))
	return rerank_scores


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-c', '--collection_path', required=True)
	parser.add_argument('-r', '--run_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()
	# 'data/consumer/consumer_questions_prelim.json'
	with open(args.query_path) as f:
		queries = json.load(f)
	# 'data/consumer/version_2_split'
	collection_path = args.collection_path
	# 'runs/consumer/pruned_biobert_msmarco_multi_sentence'
	run_path = args.run_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '')

	rerank_scores = read_scores(run_path)
	with open(output_path, 'w') as f:
		for query in queries:
			question_id = query['question_id']
			if question_id in rerank_scores:
				question_scores = rerank_scores[question_id]
				write_results(question_id, question_scores, output_name, f, top_k=1000)
			else:
				print(f'No scores found for question: {question_id}')



