
import numpy as np
import json
import argparse
from collections import defaultdict

from ndns_utils import get_ranking


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-ccp', '--cc_path', default=None)
	parser.add_argument('-ap', '--answers_path', default=None)
	parser.add_argument('-qp', '--queries_path', default=None)
	parser.add_argument('-qep', '--qe_path', default=None)
	parser.add_argument('-rrp', '--rr_path', default=None)
	parser.add_argument('-op', '--output_path', default=None)
	parser.add_argument('-qet', '--qe_threshold', default=0.001, type=float)
	parser.add_argument('-rrt', '--rr_threshold', default=3.0, type=float)

	args = parser.parse_args()

	cc_path = args.cc_path
	answers_path = args.answers_path
	queries_path = args.queries_path
	qe_path = args.qe_path
	rr_path = args.rr_path
	output_path = args.output_path

	with open(queries_path) as f:
		queries = json.load(f)
		# [question_id] -> query
		queries = {q['question_id']: q for q in queries}
	with open(cc_path) as f:
		# [question_id] -> entailed_sets
		rgqe_cc = json.load(f)
	with open(answers_path) as f:
		# [answer_id] -> answer
		answers = json.load(f)
	with open(rr_path) as f:
		# [answer_id][str(entailed_set_id)] -> score
		rr = json.load(f)
	with open(qe_path) as f:
		# [answer_id][str(entailed_set_id)] -> list[es_id, es_prob]
		qe = json.load(f)

	rerank_scores = answers['rerank_scores']
	answer_text_lookup = answers['answer_text_lookup']

	results = {}
	for question_id, question_answers in rerank_scores.items():

		rgqe_top_cc = rgqe_top_q_cc[question_id]
		top_answer_sets = rgqe_top_cc['answer_sets']
		entailed_sets_text = {
			e['entailed_set_id']: e['entailed_set'][0]['entailed_set_text'] for e in
			rgqe_top_cc['entailed_sets']
		}
		num_answers_with_set = 0
		answer_lookup = {}
		top_answers = []
		max_non_top_score = 0.0
		for answer in question_answers:
			answer_id = answer['answer_id']
			answer_lookup[answer_id] = answer
			if answer_id in rr:
				# TODO

				answer['top'] = True
				# answer['score']
			else:
				answer['top'] = False

			answer['entailed_sets'] = qa_sets
			answer['text'] = answer_text_lookup[answer_id]
			answer['entailed_sets_text'] = [entailed_sets_text[x] for x in qa_sets]

		query = queries[question_id]
		results[question_id] = {
			'query': query,
			'answers': list(sorted(question_answers, key=lambda x: x['score'], reverse=True))
		}

	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)
