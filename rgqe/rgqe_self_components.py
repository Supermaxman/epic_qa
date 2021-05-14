
import sys
from collections import defaultdict
import argparse
import json

import networkx as nx


def create_components(sample_entail_pairs, answer_samples, threshold):
	results = defaultdict(list)
	entailed_set_id = 0
	for answer_id, sample_texts in answer_samples.items():
		a_graph = nx.Graph()
		if answer_id in sample_entail_pairs:
			samples = sample_entail_pairs[answer_id]
			for sample_a_id, sample_b_id, score in samples:
				if score < threshold:
					continue
				a_graph.add_edge(
					sample_a_id,
					sample_b_id,
					weight=score
				)
		for node in range(len(sample_texts)):
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
	return results


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-e', '--expand_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)

	args = parser.parse_args()

	sys.setrecursionlimit(10 ** 6)
	input_path = args.input_path
	expand_path = args.expand_path
	output_path = args.output_path
	threshold = args.threshold

	with open(input_path) as f:
		sample_entail_pairs = json.load(f)

	with open(expand_path) as f:
		answer_samples = json.load(f)

	component_results = create_components(
		sample_entail_pairs,
		answer_samples,
		threshold
	)

	with open(output_path, 'w') as f:
		json.dump(component_results, f, indent=2)






