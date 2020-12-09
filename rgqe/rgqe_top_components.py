
import sys
from collections import defaultdict
import argparse
import json


class DFS(object):
	def __init__(self, nodes, edges):
		self.visited = {}
		self.nodes = nodes
		self.edges = edges
		for v in self.nodes:
			self.visited[v] = False

	def find_connected(self):
		components = []
		for v in self.nodes:
			if not self.visited[v]:
				c_list = set()
				c = self.dfs_search(v, c_list)
				components.append(c)
		return components

	def dfs_search(self, v, c_list):
		self.visited[v] = True
		c_list.add(v)
		for c in self.edges[v]:
			if c in self.visited and not self.visited[c]:
				self.dfs_search(c, c_list)
		return c_list


def create_components(entail_set_pairs, answer_sets, threshold):
	entailed_set_text_lookup = {}
	new_entailed_set_id = 0
	for answer_id, a_sets in answer_sets.items():
		for a_set in a_sets:
			entailed_set_id = a_set['entailed_set_id']
			entailed_set_text = a_set['entailed_set'][0]['sample_text']
			entailed_set_text_lookup[entailed_set_id] = entailed_set_text

	nodes = set()
	edges = defaultdict(set)

	entailed_set_answer_lookup = defaultdict(set)
	for answer_a_id, answer_b_id, entail_set_a_id, entail_set_b_id, entail_prob in entail_set_pairs:
		entailed_set_answer_lookup[answer_a_id].add(entail_set_a_id)
		entailed_set_answer_lookup[answer_b_id].add(entail_set_b_id)
		nodes.add(entail_set_a_id)
		nodes.add(entail_set_b_id)
		if entail_prob < threshold:
			continue
		edges[entail_set_a_id].add(entail_set_b_id)
		edges[entail_set_b_id].add(entail_set_a_id)

	dfs = DFS(
		nodes=nodes,
		edges=edges,
	)
	merged_entailed_sets = []
	entailed_sets = dfs.find_connected()
	merged_mapping = {}
	for entailed_nodes in entailed_sets:
		set_samples = []
		for v in entailed_nodes:
			merged_mapping[v] = new_entailed_set_id
			num_connected = len(edges[v].intersection(entailed_nodes))
			set_sample = {
				'entailed_set_id': v,
				'entailed_set_text': entailed_set_text_lookup[v],
				'num_connected': num_connected
			}
			set_samples.append(set_sample)
		set_samples = list(sorted(set_samples, key=lambda x: x['num_connected'], reverse=True))
		answer_set = {
			'entailed_set_id': new_entailed_set_id,
			'entailed_set': set_samples
		}

		merged_entailed_sets.append(
			answer_set
		)
		new_entailed_set_id += 1

	num_less = 0
	for merged_entailed_set in merged_entailed_sets:
		if len(merged_entailed_set["entailed_set"]) < 2:
			num_less += 1
		else:
			print(f'{merged_entailed_set["entailed_set_id"]}({len(merged_entailed_set["entailed_set"])}): '
						f'{merged_entailed_set["entailed_set"][0]["entailed_set_text"]}')
	print(f'{num_less} sets with 1 entailed set')

	merged_entailed_set_answer_lookup = {}
	for answer_id, a_sets in entailed_set_answer_lookup.items():
		new_entailed_sets = set()
		for entailed_set_id in a_sets:
			new_entailed_set_id = merged_mapping[entailed_set_id]
			new_entailed_sets.add(new_entailed_set_id)
		merged_entailed_set_answer_lookup[answer_id] = list(new_entailed_sets)
	results = {
		'entailed_sets': merged_entailed_sets,
		'answer_sets': merged_entailed_set_answer_lookup,
	}
	return results


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-c', '--cc_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.5, type=float)

	args = parser.parse_args()

	sys.setrecursionlimit(10 ** 6)
	input_path = args.input_path
	cc_path = args.cc_path
	output_path = args.output_path
	threshold = args.threshold

	with open(input_path) as f:
		entail_set_pairs = json.load(f)

	with open(cc_path) as f:
		answer_sets = json.load(f)

	component_results = create_components(
		entail_set_pairs,
		answer_sets,
		threshold
	)

	with open(output_path, 'w') as f:
		json.dump(component_results, f, indent=2)






