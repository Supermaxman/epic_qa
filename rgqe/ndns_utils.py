
import functools
import math
import operator
import heapq

from dataclasses import dataclass, field
from typing import Dict, Set, List


@dataclass
class JudgedSentence:
	""" Represents the nugget annotations for a single sentence. """
	sentence_id: str
	nugget_ids: Set[str] = field(default_factory=set)

	@property
	def sentence_idx(self) -> int:
		return JudgedSentence.id2idx(self.sentence_id)

	@staticmethod
	def id2idx(sentence_id: str) -> int:
		return int(sentence_id.rpartition('-S')[-1])

	@staticmethod
	def get_context_id(sentence_id: str) -> str:
		return sentence_id.rpartition('-')[0]


@dataclass
class JudgedContext:
	""" Holds the sentence annotations for a single context. """
	context_id: str
	sentences: Dict[str, JudgedSentence] = field(default_factory=dict)	# Sentence ID -> Sentence Annotations

	def sentence_idx2id(self, sentence_idx: int) -> str:
		"""
				Converts a numeric sentence index to its corresponding ID
				:param sentence_idx: numeric sentence index
				:return: corresponding sentence ID
				"""
		return f'{self.context_id}-S{sentence_idx:0>3d}'

	def sentence_for_index(self, sentence_idx: int) -> JudgedSentence:
		"""
				Get the annotations for the sentence at the given numeric index
				:param sentence_idx: numeric sentence index
				:return: sentence annotations
				"""
		sentence_id = self.sentence_idx2id(sentence_idx)
		return self.sentences.get(sentence_id, JudgedSentence(sentence_id))


@dataclass
class JudgedQuestion:
	question_id: str
	nuggets: Dict[str, str] = field(default_factory=dict)	# Nugget ID -> Nugget Name
	contexts: Dict[str, JudgedContext] = field(default_factory=dict)	# Context ID -> Context Annotations


@dataclass
class Answer:
	start_sent_id: str
	end_sent_id: str

	def __hash__(self):
		return hash(self.start_sent_id) + 31 * hash(self.end_sent_id)

	def __eq__(self, other):
		return self.start_sent_id == other.start_sent_id and self.end_sent_id == other.end_sent_id

	@property
	def context_id(self) -> str:
		return JudgedSentence.get_context_id(self.start_sent_id)

	@classmethod
	def from_string(cls, s):
		return Answer(*s.split(':', maxsplit=2))


@dataclass
class ScoredAnswer:
	answer: Answer
	gain: float
	nuggets: Set[str] = field(default_factory=set)


@dataclass
class Ranking:
	answers: List[ScoredAnswer] = field(default_factory=list)
	nuggets: Set[str] = field(default_factory=set)
	score: float = 0


def score_answer(answer: Answer,
								 seen_nuggets: Set[str],
								 context: JudgedContext,
								 count_redundant_sentences: bool = False,
								 count_only_novel_nuggets: bool = True,
								 count_filler_sentences: bool = True,
								 merge_novel_sentences: bool = False,
								 ignore_sentence_factor: bool = False):
	"""
		Compute the novelty score of an answer given previously seen nuggets and the judgments for the answer's context

		(Novelty Score) = (# of nuggets) * (# of nuggets + 1) / ((# of nuggets) + (sentence factor))

		where (# of nuggets) is restricted to only novel nuggets if count_only_novel_nuggets=True
		and the sentence factor is defined as:
		(sentence factor) =
			((# of redundant sentences) if count_redundant_sentences=True) +
			((# of filler sentences) if count_filler_sentences=True) +
			(max((# of novel sentences), 1) if merge_novel_sentences=True,
			 (# of novel sentences), otherwise)

		and "redundant sentences" are sentences containing only nuggets retrieved in earlier ranked answers,
		"filler sentences" are sentences with no nuggets at all, and
		"novel sentences" are sentences with novel (i.e., previously unseen) nuggets.


		:param answer: Answer to score
		:param seen_nuggets: Nuggets seen previously in the ranked list of answers
		:param context: Annotated context with sentence-level nugget judgments
		:param count_redundant_sentences: If True, the number of sentences with previously-seen nuggets will be
		added to the sentence factor for scoring, used for Partial scoring
		:param count_only_novel_nuggets: If True, only novel nuggets will contribute to the score, defaults to True
		(setting this to False reduces the measure to NDCG)
		:param count_filler_sentences: If True, sentences without any nuggets will be added to the sentence factor,
		this is included mostly for debugging purposes and should be left as True
		:param merge_novel_sentences: If True, the number of sentences with novel nuggets will be reduced to a maximum
		of 1, with the idea being to not penalize providing multiple sentences as evidence of a new nugget,
		used for Relaxed scoring
		:param ignore_sentence_factor: If True, the sentence factor is ignored and the score is just the number of nuggets
		in the answer, primarily included for debugging purposes
		:return:
		"""
	start_idx = JudgedSentence.id2idx(answer.start_sent_id)
	end_idx = JudgedSentence.id2idx(answer.end_sent_id)

	# Count the number of sentences of each type
	num_redundant_sentences = 0
	num_novel_sentences = 0
	num_filler_sentences = 0
	answer_nuggets = set()
	for sent_idx in range(start_idx, end_idx + 1):
		sent = context.sentence_for_index(sent_idx)
		sent_nuggets = sent.nugget_ids
		novel_nuggets = sent_nuggets - seen_nuggets
		if not novel_nuggets:
			if sent_nuggets:
				# We only have redundant nuggets (i.e., nuggets we've seen at earlier ranks)
				num_redundant_sentences += 1
			else:
				# We have no nuggets at all
				num_filler_sentences += 1
		else:
			# We have novel nuggets, yay!
			num_novel_sentences += 1
		# All all nuggets in this sentence to the set of nuggets for the answer
		answer_nuggets.update(sent_nuggets)

	# Sanity check that we have not under- or over-counted any sentences
	num_sentences = end_idx + 1 - start_idx
	assert num_redundant_sentences + num_novel_sentences + num_filler_sentences == num_sentences

	# Determine the sentence factor
	sentence_factor = 0

	if count_filler_sentences:
		sentence_factor += num_filler_sentences

	if count_redundant_sentences:
		sentence_factor += num_redundant_sentences

	if merge_novel_sentences:
		sentence_factor += min(num_novel_sentences, 1)
	else:
		sentence_factor += num_novel_sentences

	if ignore_sentence_factor:
		sentence_factor = 1

	if count_only_novel_nuggets:
		# Remove previously seen nuggets from the set of nuggets in this answer
		# since both datastructures are sets, this does set difference
		answer_nuggets = answer_nuggets - seen_nuggets

	num_nuggets = len(answer_nuggets)

	if num_nuggets == 0:
		# If we neglect to count filter and redundant sentences, and have no novel sentences, we will
		# end up diving by zero if ignore_sentence_factor=False, this isn't really a valid configuration
		# of parameters for scoring, but we return zero here just to be safe
		score = 0.
	else:
		score = num_nuggets * (num_nuggets + 1) / (num_nuggets + sentence_factor)

	return ScoredAnswer(answer, score, answer_nuggets)


def get_ideal_ranking(question: JudgedQuestion,
											answers: List[Answer],
											score_fn,
											max_len: int = 1000,
											k: int = 10) -> Ranking:
	"""
		Perform a beam-search over candidate answers to produce an "ideal" ranking of answers to the given question
		based on its judgments and the given scoring function
		:param question: Judgments for a single question
		:param answers: Candidate answers for that question (i.e., from	`get_potential_answers`)
		:param score_fn: scoring function (used to compute the gain of an answer)
		:param max_len: maximum ranking length
		:param k: beam-width
		:return: Optimal ranking for the question based on the given score_fn and judgments
		"""

	# Candidate rankings in the beam, initialize with a single empty ranking
	rankings = [Ranking()]
	for r in range(max_len):
		# DCG denominator is log2(r + 1) where r starts at 1, so we need to add 2 since our r starts at 0
		dcg_denom = math.log2(r + 2)
		# Store all potential rankings up to depth r for all candidates in the beam
		candidates = list()
		# Iterate over all candidate rankings in the beam
		for ranking in rankings:
			# Keep track of answers we've seen so we don't include duplicates in our ideal ranking
			seen_answers = frozenset(map(operator.attrgetter('answer'), ranking.answers))
			# Iterate over all potential answers
			for a, answer in enumerate(answers):
				if answer not in seen_answers:
					# Calculate the score of this answer given the nuggets we've seen in this candidate ranking
					scored_answer = score_fn(answer, ranking.nuggets, question.contexts[answer.context_id])
					# Prune answers with zero scores
					if scored_answer.gain > 0:
						# Calculate discounted score for the ranking obtained by adding this answer to the
						# current candidate
						dcg = ranking.score + (scored_answer.gain / dcg_denom)
						candidates.append((ranking, scored_answer, dcg))
		# If none of the answers increase the score for any ranking in the beam, we stop early
		if not candidates:
			print('Exhausted all answers by rank %d', r + 1)
			break
		# Take the top K scoring rankings from all candidates
		top_k = heapq.nlargest(k, candidates, key=operator.itemgetter(2))
		# Update our current beam candidates!
		rankings = []
		for p, a, dcg in top_k:
			rankings.append(
				Ranking(
					answers=p.answers + [a],	# Add the best answer to its candidate ranking
					nuggets=p.nuggets.union(a.nuggets),	# Update the nuggets in the new ranking
					score=dcg
				)
			)

	# Return the best ranking from our beam
	return max(rankings, key=operator.attrgetter('score'))


def get_ranking(question_id, question_answers, entailed_sets_text):
	question = JudgedQuestion(question_id)
	question.nuggets = {
		nugget_id: nugget_text for (nugget_id, nugget_text) in entailed_sets_text.items()
	}
	answer_list = []
	for answer in question_answers:
		a = Answer.from_string(answer['answer_id'])
		sentence_id = a.start_sent_id
		answer_list.append(a)
		context_id = JudgedSentence.get_context_id(sentence_id)
		if context_id not in question.contexts:
			question.contexts[context_id] = JudgedContext(context_id)
		context = question.contexts[context_id]
		# TODO re-write this later for multi-sentence spans
		if sentence_id not in context.sentences:
			context.sentences[sentence_id] = JudgedSentence(
				sentence_id,
				nugget_ids=set(answer['entailed_sets'])
			)
		else:
			context.sentences[sentence_id].nugget_ids.update(answer['entailed_sets'])

	ideal_ranking = get_ideal_ranking(
		question,
		answers=answer_list,
		score_fn=functools.partial(
			score_answer,
			count_redundant_sentences=True
		),
		k=10
	)
	# Save the score of the ideal ranking for this metric for this question
	return ideal_ranking
