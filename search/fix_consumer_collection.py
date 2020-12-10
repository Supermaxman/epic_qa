
from collections import defaultdict
import json
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection_path', required=True)
parser.add_argument('-o', '--output_path', required=True)
parser.add_argument('-ps', '--num_processes', default=16, type=int)

args = parser.parse_args()

collection_path = args.collection_path
output_path = args.output_path
num_processes = args.num_processes


def extract_doc(doc_name):
	doc_path = os.path.join(collection_path, doc_name)
	with open(doc_path) as f:
		doc = json.load(f)
	doc_id = doc['document_id']
	new_doc_path = os.path.join(output_path, doc_id + '.json')
	with open(new_doc_path, 'w') as f:
		json.dump(doc, f, indent=2)
	return doc_id


file_names = [d_name for d_name in os.listdir(collection_path) if d_name.endswith('.json')]
with Pool(processes=num_processes) as p:
	for doc_id in tqdm(
			p.imap_unordered(extract_doc, file_names), total=len(file_names)):
		pass


