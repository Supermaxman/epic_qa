
import sys
from collections import defaultdict
import argparse
import json
import numpy as np


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-c', '--cc_path', required=True)
	parser.add_argument('-a', '--answers_path', required=True)
	parser.add_argument('-q', '--queries_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)
	parser.add_argument('-r', '--ratio', default=1.0, type=float)

	args = parser.parse_args()

	input_path = args.input_path
	cc_path = args.cc_path
	answers_path = args.answers_path
	queries_path = args.queries_path
	output_path = args.output_path
	threshold = args.threshold
	ratio = args.ratio

	with open(input_path, 'r') as f:
		# [question_id][answer_id] -> entailed sets
		question_answer_sets = json.load(f)

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
	entailed_set_counts = defaultdict(lambda: defaultdict(int))
	for question_id, question_answers in rerank_scores.items():
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
		for answer in question_answers:
			answer_id = answer['answer_id']
			if answer_id in top_answer_sets:
				qa_sets = top_answer_sets[answer_id]
			else:
				qa_sets = set()
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
			for entailed_set in qa_sets:
				entailed_set_counts[question_id][entailed_set] += 1

		print(f'{question_id}: {num_answers_with_set/len(question_answers):.2f}% '
					f'percent answers with at least one entailed set')

	results = {}
	for question_id, question_answers in rerank_scores.items():
		query = queries[question_id]
		seen_entailed_sets = set()
		top_100_set_counts = []
		outside_top_100_set_counts = []
		answer_count = len(question_answers)
		q_set_counts = entailed_set_counts[question_id]
		# overall scores for each entailed_set
		q_set_score = {
			entailed_set_id: np.log(answer_count / entailed_set_df)
			for (entailed_set_id, entailed_set_df) in q_set_counts.items()
		}
		qa_entailed_set_counts = defaultdict(int)
		qa_entailed_set_scores = {
			entailed_set_id: 1.0
			for (entailed_set_id, entailed_set_df) in q_set_counts.items()
		}

		for answer in question_answers:
			answer_id = answer['answer_id']
			text = answer['text']
			rerank_score = answer['score']
			answer['rerank_score'] = rerank_score
			entailed_sets = set(answer['entailed_sets'])
			num_entailed = len(entailed_sets)
			overlap_sets = entailed_sets.intersection(seen_entailed_sets)
			num_overlap = len(overlap_sets)
			# 0 means all seen, 1 means all novel
			novelty_ratio = 1.0 - (num_overlap / max(num_entailed, 1))
			novel_sets = entailed_sets.difference(overlap_sets)
			novel_count = len(novel_sets)
			new_score = 0.0
			# set_score = rerank_score / max(num_entailed, 1)
			set_score = rerank_score / max(num_entailed, 1)
			for entailed_set_id in entailed_sets:
				novelty_score = qa_entailed_set_scores[entailed_set_id]
				if rerank_score > 0:
					new_score += 0.5 * set_score + 0.5 * novelty_score * set_score
				else:
					new_score += 0.5 * set_score + (1.0 - (1.0 - novelty_score)) * set_score

			if answer['rank'] <= 100:
				top_100_set_counts.append(num_entailed)
			else:
				outside_top_100_set_counts.append(num_entailed)


			answer['score'] = new_score
			for entailed_set_id in entailed_sets:
				qa_entailed_set_counts[entailed_set_id] += 1
				# qa_entailed_set_scores[entailed_set_id] = ratio ** (2.0 * qa_entailed_set_counts[entailed_set_id])
				qa_entailed_set_scores[entailed_set_id] = np.exp(-qa_entailed_set_counts[entailed_set_id])
			seen_entailed_sets = seen_entailed_sets.union(entailed_sets)

		print(f'{question_id}: #top_100_avg_set_counts={np.mean(top_100_set_counts):.2f}')
		print(f'{question_id}: #outside_top_100_avg_set_counts={np.mean(outside_top_100_set_counts):.2f}')
		results[question_id] = {
			'query': query,
			'answers': list(sorted(question_answers, key=lambda x: x['score'], reverse=True))
		}

	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)






