
import json
from tqdm import tqdm
from pyserini.search import SimpleSearcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dt', '--doc_type', required=True)
parser.add_argument('-i', '--index', required=True)
parser.add_argument('-q', '--query', required=True)
parser.add_argument('-r', '--run_name', required=True)
parser.add_argument('-k', '--top_k', default=1000, type=int)

args = parser.parse_args()

# 'expert'
doc_type = args.doc_type
# 'baseline_doc'
run_name = args.run_name
index_path = f'indices/{doc_type}/{args.index}'
# expert_questions_prelim.json
query_path = f'data/{doc_type}/{args.query}'
run_path = f'runs/{doc_type}/{run_name}'
top_k = args.top_k

with open(query_path) as f:
	queries = json.load(f)

searcher = SimpleSearcher(index_path)
print(f'Running search and writing results to {run_path}...')
with open(run_path, 'w') as fo:
	for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
		hits = searcher.search(query['question'], k=top_k)
		seen_docids = set()
		for rank, hit in enumerate(hits[:top_k], start=1):
			doc_id = hit.docid
			if '-' in doc_id:
				doc_id = doc_id.split('-')[0]
			if doc_id in seen_docids:
				continue
			seen_docids.add(doc_id)
			score = hit.score
			line = f'{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
