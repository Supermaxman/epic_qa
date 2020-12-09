
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
# TODO make args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection_path', required=True)
parser.add_argument('-o', '--output_path', required=True)
parser.add_argument('-ex', '--expand_path', default=None)

args = parser.parse_args()

# collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded/'
collection_path = args.collection_path
# json_collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded_doc_json/'
output_path = args.output_path
expand_path = args.expand_path

expand_lookup = {}
if expand_path is not None:
  print(f'Loading expanded questions...')
  with open(expand_path, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        passage = json.loads(line)
        expand_lookup[passage['answer_id']] = passage['samples']

print(f'Creating collection...')
with open(collection_path, 'r') as fi:
  with open(output_path, 'w') as fo:
    for line in tqdm(fi, total=len(expand_lookup)):
      line = line.strip()
      if line:
        passage = json.loads(line)
        passage_id = passage['context_id']
        passage_text = passage['text']
        answer_id = passage['sentences'][0]['sentence_id'] + ':' + passage['sentences'][-1]['sentence_id']
        if expand_path is not None:
          expand_samples = expand_lookup[answer_id]
          expand_text = ' '.join(expand_samples)
          passage_text += ' ' + expand_text
        fo.write(
          json.dumps({
            'id': passage_id,
            'contents': passage_text
          }) + '\n'
        )
