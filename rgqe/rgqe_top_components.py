
import sys
from collections import defaultdict
import argparse
import json
import os

import networkx as nx


def create_components(question_entail_set_pairs, answer_sets, cc_threshold):
	entailed_set_text_lookup = {}
	for answer_id, a_sets in answer_sets.items():
		for a_set in a_sets:
			entailed_set_id = a_set['entailed_set_id']
			entailed_set_text = a_set['entailed_set'][0]['sample_text']
			entailed_set_text_lookup[entailed_set_id] = entailed_set_text

	results = {}
	graphs = {}
	for question_id, entail_set_pairs in question_entail_set_pairs.items():
		new_entailed_set_id = 0
		q_graph = nx.Graph()
		graphs[question_id] = q_graph

		entailed_set_answer_lookup = defaultdict(set)
		for answer_a_id, answer_b_id, entail_set_a_id, entail_set_b_id, entail_prob in entail_set_pairs:
			entailed_set_answer_lookup[answer_a_id].add(entail_set_a_id)
			entailed_set_answer_lookup[answer_b_id].add(entail_set_b_id)

			q_graph.add_node(entail_set_a_id)
			q_graph.add_node(entail_set_b_id)

			if entail_prob >= cc_threshold:
				q_graph.add_edge(
					entail_set_a_id,
					entail_set_b_id,
					weight=entail_prob
				)

		merged_entailed_sets = []
		entailed_sets = nx.connected_components(q_graph)
		merged_mapping = {}
		for entailed_nodes in entailed_sets:
			set_samples = []
			for v in entailed_nodes:
				merged_mapping[v] = new_entailed_set_id
				num_connected = len(set(q_graph.neighbors(v)).intersection(set(entailed_nodes)))
				set_sample = {
					'entailed_set_id': v,
					'entailed_set_text': entailed_set_text_lookup[v],
					'num_connected': num_connected
				}
				set_samples.append(set_sample)
			set_samples = list(sorted(set_samples, key=lambda x: x['num_connected'], reverse=True))
			merged_set = {
				'entailed_set_id': new_entailed_set_id,
				'entailed_set': set_samples
			}

			merged_entailed_sets.append(
				merged_set
			)
			new_entailed_set_id += 1

		unconnected_sets = set()
		for merged_entailed_set in merged_entailed_sets:
			if len(merged_entailed_set["entailed_set"]) < 2:
				unconnected_sets.add(merged_entailed_set["entailed_set"][0]['entailed_set_id'])
			else:
				print(f'({len(merged_entailed_set["entailed_set"])}): '
							f'{merged_entailed_set["entailed_set"][0]["entailed_set_text"]}')

		print(f'({len(unconnected_sets)}) unconnected sets')

		merged_entailed_set_answer_lookup = {}
		seen_answers = len(entailed_set_answer_lookup)
		connected_answers = 0
		for answer_id, a_sets in entailed_set_answer_lookup.items():
			new_entailed_sets = set()
			num_unconnected = 0
			for entailed_set_id in a_sets:
				if entailed_set_id in unconnected_sets:
					num_unconnected += 1
				else:
					new_entailed_set_id = merged_mapping[entailed_set_id]
					new_entailed_sets.add(new_entailed_set_id)
			if num_unconnected != len(a_sets):
				connected_answers += 1
			merged_entailed_set_answer_lookup[answer_id] = sorted(list(new_entailed_sets))
		results[question_id] = {
			'entailed_sets': merged_entailed_sets,
			'answer_sets': merged_entailed_set_answer_lookup,
		}
		print(f'{connected_answers/seen_answers:.2f}% answers part of at least one connected set '
					f'({connected_answers}/{seen_answers})')
	return results, graphs


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-c', '--cc_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-og', '--graph_path', required=True)
	parser.add_argument('-tc', '--cc_threshold', default=0.01, type=float)

	args = parser.parse_args()

	sys.setrecursionlimit(10 ** 6)
	input_path = args.input_path
	cc_path = args.cc_path
	output_path = args.output_path
	cc_threshold = args.cc_threshold
	graph_path = args.graph_path

	if not os.path.exists(graph_path):
		os.mkdir(graph_path)

	with open(input_path) as f:
		question_entail_set_pairs = json.load(f)

	with open(cc_path) as f:
		answer_sets = json.load(f)

	component_results, q_graphs = create_components(
		question_entail_set_pairs,
		answer_sets,
		cc_threshold,
	)

	for q_id, q_graph in q_graphs.items():
		q_graph_path = os.path.join(graph_path, f'{q_id}.cc_graph')
		nx.write_adjlist(q_graph, q_graph_path)

	with open(output_path, 'w') as f:
		json.dump(component_results, f, indent=2)






