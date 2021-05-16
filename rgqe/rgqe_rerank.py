
import numpy as np
import json
import argparse
from collections import defaultdict

from ndns_utils import get_ranking


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cc_path', required=True)
	parser.add_argument('-a', '--answers_path', required=True)
	parser.add_argument('-q', '--queries_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	cc_path = args.cc_path
	answers_path = args.answers_path
	queries_path = args.queries_path
	output_path = args.output_path
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
		if question_id in rgqe_top_q_cc:
			rgqe_top_cc = rgqe_top_q_cc[question_id]
			top_answer_sets = rgqe_top_cc['answer_sets']
			entailed_sets_text = {
				e['entailed_set_id']: e['entailed_set'][0]['entailed_set_text'] for e in
				rgqe_top_cc['entailed_sets']
			}
		else:
			top_answer_sets = {}
			entailed_sets_text = {}
		num_answers_with_set = 0
		answer_lookup = {}
		top_answers = []
		max_non_top_score = 0.0
		for answer in question_answers:
			answer_id = answer['answer_id']
			answer_lookup[answer_id] = answer
			if answer_id in top_answer_sets:
				qa_sets = top_answer_sets[answer_id]
				top_answers.append(answer)
			else:
				max_non_top_score = max(max_non_top_score, answer['score'])
				qa_sets = set()
			qa_sets = list(qa_sets)
			if len(qa_sets) > 0:
				num_answers_with_set += 1
			answer['entailed_sets'] = qa_sets
			answer['text'] = answer_text_lookup[answer_id]
			answer['entailed_sets_text'] = [entailed_sets_text[x] for x in qa_sets]
		print(f'{question_id}: {num_answers_with_set / len(question_answers):.2f}% '
					f'percent answers with at least one entailed set')

		ranking = get_ranking(question_id, top_answers, entailed_sets_text)
		ndns_rank = 0
		for idx, ndns_scored_answer in enumerate(ranking.answers):
			a_answer = ndns_scored_answer.answer
			answer_id = f'{a_answer.start_sent_id}:{a_answer.end_sent_id}'
			answer = answer_lookup[answer_id]
			answer['ndns_gain'] = ndns_scored_answer.gain
			# goes from 1.0 to 0.0
			# add max non top answer score to make sure ndns is ranked higher
			answer['ndns_score'] = (1.0 - (ndns_rank / len(ranking.answers))) + max_non_top_score
			ndns_rank += 1

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
