
from collections import defaultdict
import argparse
import json
import itertools


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
		self.entail_edges[other.id] = other

	def remove_entailment(self, other):
		del self.entail_edges[other.id]

	def entails(self, other):
		return other.id in self.entail_edges

	def set_merged(self, other):
		assert self.merged_parent is None
		self.merged_parent = other

	def is_merged(self):
		return self.merged_parent is not None

	def left_merge(self, other):
		self.remove_entailment(other)
		other.remove_entailment(self)
		for o_child_id, o_child in other.entail_edges.items():
			o_child.remove_entailment(other)
			o_child.add_entailment(self)
			self.add_entailment(o_child)
		self.merged_nodes[other.id] = other
		other.set_merged(self)

	def __len__(self):
		return len(self.entail_edges)


class AnswerNode(object):
	def __init__(self, question_id, answer_id, rerank_score):
		self.question_id = question_id
		self.answer_id = answer_id
		self.rerank_score = rerank_score
		self.children = {}

	def add_child(self, child):
		self.children[child.sample_id] = child

	def get_child(self, sample_id):
		return self.children[sample_id]

	def num_unique(self):
		num_unique = 0
		for child_a, child_b in itertools.combinations(self.children, r=2):
			if child_a.entails(child_b):
				num_unique += 1
		return num_unique

	def merge_children(self):
		for child_a, child_b in itertools.combinations(list(self.children.values()), r=2):
			if child_a.entails(child_b):
				child_b.left_merge(child_a)
		for child in self.children:
			if child.is_merged():
				del self.children[child.id]

	def __len__(self):
		return len(self.children)


class QuestionEntailmentGraph(object):
	def __init__(self, question_id, question_scores, question_samples):
		self.answers = {}
		for answer in question_scores:
			answer_node = AnswerNode(
				question_id,
				answer['answer_id'],
				answer['score']
			)
			self.answers[answer_node.answer_id] = answer
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
				for sample_a_id, sample_b_id in a_b_sample_pairs:
					sample_a = answer_a.get_child(sample_a_id)
					sample_b = answer_b.get_child(sample_b_id)
					sample_a.add_entailment(sample_b)
					sample_b.add_entailment(sample_a)

	def rerank_answers(self):
		reranked_answers = []
		for answer_id, answer in self.answers.items():
			answer.merge_children()
			num_unique = len(answer)
			reranked_answers.append(answer)
		reranked_answers = list(sorted(reranked_answers, key=lambda x: x.score, reverse=True))
		return reranked_answers


def create_results(query_results, sample_entail_pairs):
	results = defaultdict(list)
	for question_id, q_scores in query_results.items():
		q_samples = sample_entail_pairs[question_id]
		qe_graph = QuestionEntailmentGraph(
			question_id,
			q_scores,
			q_samples
		)
		q_answers = qe_graph.rerank_answers()
		results[question_id] = q_answers
	return results

# def create_results(query_results, sample_entail_pairs):
# 	lower_ranked_entailed_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 	higher_ranked_entailed_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 	for question_id, answer_a_entailed_answers in sample_entail_pairs.items():
# 		for answer_a_id, answer_b_entailed_answers in answer_a_entailed_answers.items():
# 			for answer_b_id, a_b_sample_pairs in answer_b_entailed_answers.items():
# 				for sample_a, sample_b in a_b_sample_pairs:
# 					higher_ranked_entailed_answers[question_id][answer_b_id][answer_a_id].append((sample_b, sample_a))
# 					lower_ranked_entailed_answers[question_id][answer_a_id][answer_b_id].append((sample_a, sample_b))
# 	results = defaultdict(list)
# 	for question_id, q_scores in query_results.items():
# 		q_results = []
# 		for answer in q_scores:
# 			answer_id = answer['answer_id']
# 			# list of answers and samples are ranked lower than current answer
# 			answer_lower_entailed = lower_ranked_entailed_answers[question_id][answer_id]
# 			# list of answers and samples are ranked higher than current answer
# 			answer_higher_entailed = higher_ranked_entailed_answers[question_id][answer_id]
# 			# query_samples = answer['samples']
# 			# TODO more in-depth entailment checks
# 			# for q_sample in query_samples:
# 			# 	sample_id = q_sample['sample_id']
# 			# num answers with at least one sampled question which entails a current answer question
# 			num_higher_entailed_answers = 0
# 			num_higher_entailed_samples = 0
# 			for higher_answer_id, sample_pairs in answer_higher_entailed.items():
# 				if higher_answer_id != answer_id:
# 					num_higher_entailed_answers += 1
# 					num_higher_entailed_samples += len(sample_pairs)
# 				else:
# 					# TODO self-entailed samples need to be handed explicitly
# 					pass
#
# 			num_lower_entailed_answers = 0
# 			num_lower_entailed_samples = 0
# 			for lower_answer_id, sample_pairs in answer_lower_entailed.items():
# 				if lower_answer_id != answer_id:
# 					num_lower_entailed_answers += 1
# 					num_lower_entailed_samples += len(sample_pairs)
#
# 			# if an answer which is ranked higher has at least one overlapping entailed sample question
# 			# then we assume duplicate information is provided by the lower-ranked answer.
# 			# TODO work on these
# 			if num_higher_entailed_answers > 0:
# 				continue
# 			q_results.append(answer)
#
# 		results[question_id] = q_results
# 	return results


def write_run(results, output_path, output_name):
	with open(output_path, 'w') as f:
		for question_id, q_results in results.items():
			rank = 1
			for answer in q_results:
				answer_id = answer.answer_id
				answer_score = answer.score
				f.write(f'{question_id}\tQ0\t{answer_id}\t{rank}\t{answer_score:.8f}\t{output_name}\n')
				rank += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--results_path', required=True)
	parser.add_argument('-g', '--rgqe_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	# TODO figure out
	results_path = args.results_path
	rgqe_path = args.rgqe_path
	output_path = args.output_path
	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	with open(results_path) as f:
		query_results = json.load(f)

	with open(rgqe_path) as f:
		sample_entail_pairs = json.load(f)

	rerank_results = create_results(query_results, sample_entail_pairs)
	write_run(rerank_results, output_path, output_name)





