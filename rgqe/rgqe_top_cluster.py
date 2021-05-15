
import os
import sys
import json
import argparse
from collections import defaultdict

import networkx as nx
import umap
import hdbscan


def cluster_umap(q_graph):
	m = nx.linalg.graphmatrix.adjacency_matrix(q_graph, weight='q_weight').toarray()
	lowd_m = umap.UMAP(
		metric='cosine',
		n_neighbors=10,
		min_dist=0.0,
		spread=1.0,
		n_components=2,
		random_state=0,
		disconnection_distance=1000
	).fit_transform(m)
	labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=2).fit_predict(lowd_m)

	label_list = defaultdict(list)
	for node, label in zip(q_graph.nodes, labels):
		label_list[label].append(node)

	label_lists = []
	for l_id, l_nodes in label_list.items():
		if l_id >= 0:
			for node in l_nodes:
				label_lists.append(int(node))

	return label_lists


def create_entail_sets(question_entail_set_pairs, answer_sets, cc_threshold):
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
					q_weight=entail_prob
				)

		remove_nodes = set()
		for a_id, a_degree in q_graph.degree():
			if a_degree == 0:
				remove_nodes.add(a_id)

		for node in remove_nodes:
			q_graph.remove_node(node)

		entailed_sets = cluster_umap(q_graph)
		merged_entailed_sets = []
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

	component_results, q_graphs = create_entail_sets(
		question_entail_set_pairs,
		answer_sets,
		cc_threshold,
	)

	for q_id, q_graph in q_graphs.items():
		q_graph_path = os.path.join(graph_path, f'{q_id}.cc_graph')
		nx.write_adjlist(q_graph, q_graph_path)

	with open(output_path, 'w') as f:
		json.dump(component_results, f, indent=2)






