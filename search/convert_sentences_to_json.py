
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
# TODO make args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection_path', required=True)
parser.add_argument('-jc', '--json_collection_path', required=True)
parser.add_argument('-p', '--nrof_processes', default=8, type=int)

args = parser.parse_args()

# collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded/'
collection_path = args.collection_path
# json_collection_path = 'data/expert/epic_qa_cord_2020-06-19_v2_expanded_doc_json/'
json_collection_path = args.json_collection_path
nrof_processes = args.nrof_processes


if not os.path.exists(json_collection_path):
  os.mkdir(json_collection_path)


def extract_sentences(doc):
  sentences = []
  for context in doc['contexts']:
    passage_text = context['text']
    for sentence in context['sentences']:
      sentences.append(
        {
          'sentence_id': sentence['sentence_id'],
          'text': passage_text[sentence['start']:sentence['end']],
        }
      )
  return sentences


def convert_doc(doc_name):
  # ignore already existing expanded files
  input_path = os.path.join(collection_path, doc_name)
  output_path = os.path.join(json_collection_path, doc_name.replace('.json', '.jsonl'))
  # if os.path.exists(output_path):
  #   return None
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
  sentences = extract_sentences(doc)

  with open(output_path, 'w') as f:
    for sentence in sentences:
      p_dict = {
        'id': sentence['sentence_id'],
        'contents': sentence['text']
      }
      f.write(json.dumps(p_dict) + '\n')
  return output_path


print('Loading files...')
docs = [x for x in os.listdir(collection_path) if x.endswith('.json')]
print(f'Total files: {len(docs)}')
print(f'Converting files...')
with Pool(processes=nrof_processes) as p:
  for doc in tqdm(p.imap_unordered(convert_doc, docs), total=len(docs)):
    pass

print('Done!')