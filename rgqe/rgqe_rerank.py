
import numpy as np
import json
import argparse
from collections import defaultdict

from ndns_utils import get_ranking


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--all_path', default=None)
	parser.add_argument('-c', '--cc_path', required=True)
	parser.add_argument('-a', '--answers_path', required=True)
	parser.add_argument('-q', '--queries_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)
	parser.add_argument('-r', '--ratio', default=1.0, type=float)

	args = parser.parse_args()

	all_path = args.all_path
	cc_path = args.cc_path
	answers_path = args.answers_path
	queries_path = args.queries_path
	output_path = args.output_path
	threshold = args.threshold
	ratio = args.ratio
	if all_path is not None:
		with open(all_path, 'r') as f:
			# [question_id][answer_id] -> entailed sets
			question_answer_sets = json.load(f)
	else:
		question_answer_sets = {}

	with open(queries_path, 'r') as f:
		queries = json.load(f)
		# [question_id] -> query
		queries = {q['question_id']: q for q in queries}

	with open(cc_path, 'r') as f:
		# [question_id] -> entailed_sets
		rgqe_top_q_cc = json.load(f)

	with open(answers_path, 'r') as f:
		# [answer_id] -> answer
		answers = json.load(f)
	rerank_scores = answers['rerank_scores']
	answer_text_lookup = answers['answer_text_lookup']
	results = {}
	for question_id, question_answers in rerank_scores.items():
		if all_path is not None:
			if question_id not in question_answer_sets:
				answer_sets = {}
				print(f'WARNING: {question_id} has no sets outside top 100')
			else:
				answer_sets = question_answer_sets[question_id]
		rgqe_top_cc = rgqe_top_q_cc[question_id]
		top_answer_sets = rgqe_top_cc['answer_sets']
		entailed_sets_text = {
			e['entailed_set_id']: e['entailed_set'][0]['entailed_set_text'] for e in
			rgqe_top_cc['entailed_sets']
		}
		num_answers_with_set = 0
		answer_lookup = {}
		top_answers = []
		top_seen_set_answers = set()
		max_non_top_score = 0.0
		for answer in question_answers:
			answer_id = answer['answer_id']
			answer_lookup[answer_id] = answer
			if answer_id in top_answer_sets:
				qa_sets = top_answer_sets[answer_id]
				top_answers.append(answer)
				if len(qa_sets) > 0:
					top_seen_set_answers.add(answer_id)
			else:
				max_non_top_score = max(max_non_top_score, answer['score'])
				qa_sets = set()
				if all_path is not None:
					if answer_id in answer_sets:
						for entailed_set_a_id, entailed_set_b_id, entail_prob in answer_sets[answer_id]:
							if entail_prob < threshold:
								continue
							qa_sets.add(entailed_set_b_id)
			qa_sets = list(qa_sets)
			if len(qa_sets) > 0:
				num_answers_with_set += 1
			answer['entailed_sets'] = qa_sets
			answer['text'] = answer_text_lookup[answer_id]
			answer['entailed_sets_text'] = [entailed_sets_text[x] for x in qa_sets]

		total_top_set_answers = len(top_seen_set_answers)
		ndns_rank = 0
		while len(top_seen_set_answers) > 0:
			ranking = get_ranking(question_id, top_answers, entailed_sets_text, top_seen_set_answers)
			for idx, ndns_scored_answer in enumerate(ranking.answers):
				a_answer = ndns_scored_answer.answer
				answer_id = f'{a_answer.start_sent_id}:{a_answer.end_sent_id}'
				top_seen_set_answers.remove(answer_id)
				answer = answer_lookup[answer_id]
				ndns_gain = ndns_scored_answer.gain
				answer['ndns_gain'] = ndns_gain
				# TODO two ideas:
				# TODO 1. use bert score if gain is 0, not arbitrary ranking
				#
				# print(ndns_gain)
				# if ndns_gain == 1:
				# 	max_non_top_score = max(max_non_top_score, answer['score'])
				# 	num_top_zero += 1
				# goes from 1.0 to 0.0
				# add max non top answer score to make sure ndns is ranked higher
				answer['ndns_score'] = (1.0 - (ndns_rank / total_top_set_answers)) + max_non_top_score
				ndns_rank += 1

		print(f'{question_id}: {num_answers_with_set / len(question_answers):.2f}% '
					f'percent answers with at least one entailed set)')
		for answer in question_answers:
			if 'ndns_score' not in answer:
				answer['ndns_score'] = answer['score']
			else:
				answer['score'] = answer['ndns_score']
		query = queries[question_id]
		results[question_id] = {
			'query': query,
			'answers': list(sorted(question_answers, key=lambda x: x['score'], reverse=True))
		}

	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)
