
import json
from tqdm import tqdm
from pyserini.search import SimpleSearcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index_path', required=True)
parser.add_argument('-q', '--query_path', required=True)
parser.add_argument('-r', '--output_path', required=True)
parser.add_argument('-k', '--top_k', default=2000, type=int)
parser.add_argument('-bk1', '--bm25_k1', default=0.82, type=float)
parser.add_argument('-bb', '--bm25_b', default=0.68, type=float)
parser.add_argument('-l', '--label_path', default='data/prelim_judgments.json')

args = parser.parse_args()

index_path = args.index_path
query_path = args.query_path
output_path = args.output_path
label_path = args.label_path
output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

top_k = args.top_k

keep_ids = None
if label_path:
	keep_ids = set()
	with open(label_path) as f:
		questions = json.load(f)
		for question in questions:
			question_id = question['question_id']
			keep_ids.add(question_id)

queries = []
with open(query_path) as f:
	all_queries = json.load(f)
	for query in all_queries:
		if keep_ids is not None and query['question_id'] not in keep_ids:
			continue
		else:
			queries.append(query)

searcher = SimpleSearcher(index_path)
# searcher.set_bm25(args.bm25_k1, args.bm25_b)
print(f'Running search and writing results to {output_path}...')

with open(output_path, 'w') as fo:
	for query in tqdm(queries):
		question_id = query['question_id']
		hits = searcher.search(query['question'] + ' ' + query['query'] + ' ' + query['background'], k=top_k)
		for rank, hit in enumerate(hits[:top_k], start=1):
			line = f'{question_id}\tQ0\t{hit.docid}\t{rank}\t{hit.score:.8f}\t{output_name}\n'
			fo.write(line)
print('Done!')
