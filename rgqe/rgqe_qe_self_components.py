
import sys
from collections import defaultdict
import argparse
import json

import networkx as nx


def create_components(sample_entail_pairs, answer_samples, self_threshold, qe_probs, qe_threshold):
	q_results = {}
	for question_id, q_a_probs in qe_probs.items():
		entailed_set_id = 0
		results = defaultdict(list)
		for answer_id, as_probs in q_a_probs.items():
			a_graph = nx.Graph()
			sample_texts = answer_samples[answer_id]
			if answer_id in sample_entail_pairs:
				samples = sample_entail_pairs[answer_id]
				for sample_a_id, sample_b_id, score in samples:
					a_qe_prob = as_probs[sample_a_id]
					b_qe_prob = as_probs[sample_b_id]
					if a_qe_prob < qe_threshold or b_qe_prob < qe_threshold or score < self_threshold:
						continue

					a_graph.add_edge(
						sample_a_id,
						sample_b_id,
						weight=score
					)
			for node in range(len(sample_texts)):
				qe_prob = as_probs[node]
				if qe_prob < qe_threshold:
					continue
				a_graph.add_node(node)
			entailed_sets = nx.connected_components(a_graph)
			answer_sets = []
			for entailed_nodes in entailed_sets:
				answer_set_samples = []
				for v in entailed_nodes:
					num_connected = len(set(a_graph.neighbors(v)).intersection(set(entailed_nodes)))
					answer_sample = {
						'sample_id': v,
						'sample_text': sample_texts[v],
						'num_connected': num_connected
					}
					answer_set_samples.append(answer_sample)
				answer_set_samples = list(sorted(answer_set_samples, key=lambda x: x['num_connected'], reverse=True))
				answer_set = {
					'entailed_set_id': entailed_set_id,
					'entailed_set': answer_set_samples
				}

				answer_sets.append(
					answer_set
				)
				entailed_set_id += 1
			results[answer_id] = answer_sets
		q_results[question_id] = results
	return q_results


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-e', '--expand_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-qe', '--qe_path', required=True)
	parser.add_argument('-st', '--self_threshold', default=0.5, type=float)
	parser.add_argument('-qt', '--qe_threshold', default=0.5, type=float)

	args = parser.parse_args()

	sys.setrecursionlimit(10 ** 6)
	input_path = args.input_path
	expand_path = args.expand_path
	output_path = args.output_path
	self_threshold = args.self_threshold
	qe_threshold = args.qe_threshold
	qe_path = args.qe_path

	with open(input_path) as f:
		sample_entail_pairs = json.load(f)

	with open(expand_path) as f:
		answer_samples = json.load(f)
	with open(qe_path) as f:
		qe_probs = json.load(f)

	component_results = create_components(
		sample_entail_pairs,
		answer_samples,
		self_threshold,
		qe_probs,
		qe_threshold
	)

	with open(output_path, 'w') as f:
		json.dump(component_results, f, indent=2)






