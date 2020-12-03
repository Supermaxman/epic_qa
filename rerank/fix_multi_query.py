from collections import defaultdict
import json

def read_scores(run_path):
	rerank_scores = defaultdict(list)
	with open(run_path) as f:
		for line in f:
			# {query_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.4f}\t{run_name}
			line = line.strip().split()
			if len(line) == 6:
				question_id, _, doc_pass_sent_id, rank, score, _ = line
				ids = doc_pass_sent_id.split('-')
				doc_id, pass_id = ids[0], ids[1]
				pass_id = int(pass_id)
				if len(ids) == 3:
					sent_start_id, sent_end_id = ids[2], ids[2]
				elif len(ids) == 4:
					sent_start_id, sent_end_id = ids[2], ids[3]
				else:
					sent_start_id, sent_end_id = ids[2], ids[4]
				sent_start_id = int(sent_start_id)
				sent_end_id = int(sent_end_id)
				pass_id = int(pass_id)
				score = float(score)
				rerank_scores[question_id].append((doc_id, pass_id, sent_start_id, sent_end_id, score))
	return rerank_scores



run_path = 'models/pt-biobert-base-msmarco-multi-sentence/HLTRI_RERANK_2.pred'
query_path = 'data/epic_qa_prelim/consumer/questions.json'
with open(query_path) as f:
	queries = json.load(f)

scores = read_scores(run_path)

fixed_scores = defaultdict(list)

for query in queries:
	question_id = query['']
	fixed_scores[question_id]

