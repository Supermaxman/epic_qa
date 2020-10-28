
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
parser.add_argument('-bk1', '--bm25_k1', default=0.82, type=float)
parser.add_argument('-bb', '--bm25_b', default=0.68, type=float)
parser.add_argument('-db', '--debug', default=False, action='store_true')

args = parser.parse_args()

# 'expert'
doc_type = args.doc_type
# 'baseline_doc'
run_name = args.run_name
debug = args.debug
index_path = f'indices/{doc_type}/{args.index}'
# expert_questions_prelim.json
query_path = f'data/{doc_type}/{args.query}'
run_path = f'runs/{doc_type}/{run_name}'
top_k = args.top_k

with open(query_path) as f:
	queries = json.load(f)

searcher = SimpleSearcher(index_path)
searcher.set_bm25(args.bm25_k1, args.bm25_b)
print(f'Running search and writing results to {run_path}...')
with open(run_path, 'w') as fo:
	for query_id, query in tqdm(enumerate(queries, start=1), total=len(queries)):
		if debug and query_id != 1:
			continue
		hits = searcher.search(query['question'] + ' ' + query['query'], k=top_k)
		for rank, hit in enumerate(hits[:top_k], start=1):
			line = f'{query_id}\tQ0\t{hit.docid}\t{rank}\t{hit.score:.4f}\t{run_name}\n'
			fo.write(line)
print('Done!')
