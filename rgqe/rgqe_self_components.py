
import sys
from collections import defaultdict
import argparse
import json
import itertools


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
				c_list = []
				c = self.dfs_search(v, c_list)
				components.append(c)
		return components

	def dfs_search(self, v, c_list):
		self.visited[v] = True
		c_list.append(v)
		for c in self.edges[v]:
			if c in self.visited and not self.visited[c]:
				self.dfs_search(c, c_list)
		return c_list


class QuestionSampleNode(object):
	def __init__(self, parent, sample_id, entail_prob, sample_text):
		self.id = f'{parent.answer_id}|{sample_id}'
		self.parent = parent
		self.sample_id = sample_id
		self.entail_prob = entail_prob
		self.sample_text = sample_text
		self.entail_edges = {}
		self.merged_nodes = {}
		self.merged_parent = None

	def add_entailment(self, other):
		if other.id != self.id:
			self.entail_edges[other.id] = other

	def remove_entailment(self, other):
		if other.id in self.entail_edges:
			del self.entail_edges[other.id]

	def entails(self, other):
		return other.id in self.entail_edges


class AnswerNode(object):
	def __init__(self, question_id, answer_id, rerank_score):
		self.question_id = question_id
		self.answer_id = answer_id
		self.rerank_score = rerank_score
		self.children = {}
		self.score = self.rerank_score
		self.entailed_sets = set()

	def add_child(self, child):
		self.children[child.sample_id] = child

	def get_child(self, sample_id):
		return self.children[sample_id]

	def find_connected(self):
		dfs = DFS(self.children.values())
		entailed_components = dfs.find_connected()
		return entailed_components

	def __len__(self):
		return len(self.children)


class QuestionEntailmentGraph(object):
	def __init__(self, question_id, question_scores, question_samples, edge_threshold):
		self.question_id = question_id
		self.answers = {}
		for answer in question_scores:
			answer_node = AnswerNode(
				question_id,
				answer['answer_id'],
				answer['score']
			)
			self.answers[answer_node.answer_id] = answer_node
			for sample in answer['samples']:
				sample_node = QuestionSampleNode(
					answer_node,
					sample['sample_id'],
					sample['entail_prob'],
					sample['sample_text']
				)
				answer_node.add_child(sample_node)
		for answer_a_id, answer_b_entailed_answers in question_samples.items():
			answer_a = self.answers[answer_a_id]
			for answer_b_id, a_b_sample_pairs in answer_b_entailed_answers.items():
				answer_b = self.answers[answer_b_id]
				for sample_a_id, sample_b_id, a_b_entail_prob in a_b_sample_pairs:
					if a_b_entail_prob > edge_threshold:
						sample_a = answer_a.get_child(sample_a_id)
						sample_b = answer_b.get_child(sample_b_id)
						sample_a.add_entailment(sample_b)
						sample_b.add_entailment(sample_a)

	def prune_answers(self, overlap_ratio, overall_ratio, top_k):
		nodes = []
		for answer in self.answers.values():
			for sample in answer.children.values():
				nodes.append(sample)

		dfs = DFS(
			nodes=nodes
		)

		entailed_sets = dfs.find_connected()
		for entail_set_id, entailed_nodes in enumerate(entailed_sets):
			for node in entailed_nodes:
				answer = node.parent
				answer.entailed_sets.add(entail_set_id)

		# for idx, entail_fact_nodes in enumerate(entailed_sets):
		# 	print(f'  CM{idx}: ')
		# 	for node in entail_fact_nodes[:3]:
		# 		print(f'    {node.sample_text}')
		# 		# input()
		# input()
		reranked_answers = []
		total_entailed_set_count = max(len(entailed_sets), 1)
		seen_entailed_sets = set()
		num_assigned = 0
		num_removed = 0
		num_removed_overlap_ratio = 0
		num_removed_overall_ratio = 0
		for answer in sorted(self.answers.values(), key=lambda x: x.score, reverse=True):
			answer_entailed_sets = answer.entailed_sets
			if len(answer_entailed_sets) > 0:
				num_assigned += 1
			num_entailed_sets_overlapping = len(answer_entailed_sets.intersection(seen_entailed_sets))
			entailed_overlap_ratio = num_entailed_sets_overlapping / max(len(answer_entailed_sets), 1)
			entailed_overall_ratio = len(answer_entailed_sets) / total_entailed_set_count
			# overlap_ratio of 0 means result only contains new entailed sets
			# overlap_ratio of 1.0 means result must contain at least one new entailed set
			# overlap ratio above 1 means result can completely overlap with seen entailed sets
			if entailed_overlap_ratio >= overlap_ratio:
				num_removed_overlap_ratio += 1
				num_removed += 1
			elif entailed_overall_ratio <= overall_ratio:
				num_removed_overall_ratio += 1
				num_removed += 1
			else:
				reranked_answers.append(answer)
				for a_entailed_set in answer_entailed_sets:
					seen_entailed_sets.add(a_entailed_set)
		assigned_ratio = num_assigned / len(self.answers)
		print(
			f'{self.question_id} # Entail-Components: {len(entailed_sets)} (%assigned={assigned_ratio:.2f})(#removed={num_removed}, '
			f'#overlap={num_removed_overlap_ratio}, #overall={num_removed_overall_ratio})'
		)
		reranked_answers = list(sorted(reranked_answers, key=lambda x: x.score, reverse=True))
		return reranked_answers[:top_k]


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
			answer_sets.append(
				{
					'entailed_set_id': entailed_set_id,
					'entailed_set': [sample_texts[v] for v in entailed_nodes]
				}
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






