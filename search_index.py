
import json
from tqdm import tqdm
from pyserini.search import SimpleSearcher

doc_type = 'expert'
index_path = f'indices/{doc_type}/baseline_doc'
query_path = f'data/{doc_type}/expert_questions_prelim.json'
run_name = 'baseline_doc'
run_path = f'runs/{doc_type}/{run_name}'
top_k = 1000

with open(query_path) as f:
	queries = json.load(f)

searcher = SimpleSearcher(index_path)
print(f'Running search and writing results to {run_path}...')
with open(run_path, 'w') as fo:
	for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
		hits = searcher.search(query['question'], k=top_k)
		for rank, hit in enumerate(hits[:top_k], start=1):
			line = f'{query_id}\tQ0\t{hit.docid}\t{rank}\t{hit.score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
