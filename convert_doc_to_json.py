
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
# TODO make args
import argparse

collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded/'
json_collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded_doc_json/'
nrof_processes = 8
expand_docs = True

if not os.path.exists(json_collection_path):
  os.mkdir(json_collection_path)


def extract_text(doc):
  # TODO add expanded sentence text, maybe doc title
  doc_text = ' '.join([c['text'] for c in doc['contexts']])
  if expand_docs:
    exp_text = []
    for context in doc['contexts']:
      for sentence in context['sentences']:
        exp_text.append(sentence['expanded_query'])
    doc_text += ' '.join(exp_text)
  return doc_text


def convert_doc(doc_name):
  # ignore already existing expanded files
  input_path = os.path.join(collection_path, doc_name)
  output_path = os.path.join(json_collection_path, doc_name)
  if os.path.exists(output_path):
    return None
  try:
    with open(input_path, 'r') as f:
      doc = json.load(f)
  except Exception as e:
    print('----------------------')
    print('ERROR with doc:')
    print(input_path)
    print(e)
    print('----------------------')
    return None
  # gather all passages for batch processing
  id = doc['document_id']
  contents = extract_text(doc)
  doc_dict = {
    'id': id,
    'contents': contents
  }

  with open(output_path, 'w') as f:
    json.dump(doc_dict, f)
  return output_path


print('Loading files...')
docs = [x for x in os.listdir(collection_path) if x.endswith('.json')]
print(f'Total files: {len(docs)}')
print(f'Converting files...')
with Pool(processes=nrof_processes) as p:
  for doc in tqdm(p.imap_unordered(convert_doc, docs), total=len(docs)):
    pass

print('Done!')