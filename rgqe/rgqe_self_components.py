
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
		components = set()
		for v in self.nodes:
			if not self.visited[v]:
				c_list = set()
				c = self.dfs_search(v, c_list)
				components.add(c)
		return components

	def dfs_search(self, v, c_list):
		self.visited[v] = True
		c_list.add(v)
		for c in self.edges[v]:
			if c in self.visited and not self.visited[c]:
				self.dfs_search(c, c_list)
		return c_list


def create_components(sample_entail_pairs, answer_samples, threshold):
	results = defaultdict(list)
	entailed_set_id = 0
	for answer_id, samples in sample_entail_pairs.items():
		nodes = set()
		edges = defaultdict(set)
		for sample_a_id, sample_b_id, score in samples:
			nodes.add(sample_a_id)
			nodes.add(sample_b_id)
			if score < threshold:
				continue
			edges[sample_a_id].add(sample_b_id)
			edges[sample_b_id].add(sample_a_id)

		dfs = DFS(
			nodes=nodes,
			edges=edges,
		)
		entailed_sets = dfs.find_connected()
		answer_sets = []
		sample_texts = answer_samples[answer_id]
		for entailed_nodes in entailed_sets:
			answer_set_samples = []
			for v in entailed_nodes:
				num_connected = len(edges[v].intersection(entailed_nodes))
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






