
import os
import json
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

collection_path = '/users/max/data/corpora/epic_qa/expert/epic_qa_cord_2020-06-19_v2/'
expanded_path = '/users/max/data/corpora/epic_qa/expert/epic_qa_cord_2020-06-19_v2_expanded/'
batch_size = 256

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    '/users/max/data/models/t5/docT5query_base/model.ckpt-1004000', from_tf=True, config=config)
model.to(device)
model.eval()


def expand(passages, num_samples=1, max_length=64, top_k=10):
  all_samples = []
  for b_idx in range(int(np.ceil(len(passages) / batch_size))):
    batch = tokenizer.batch_encode_plus(
      # query, passage
      batch_text_or_text_pairs=[f'{x} </s>' for x in passages[b_idx * batch_size:(b_idx + 1) * batch_size]],
      padding=True,
      return_tensors='pt'
    )

    outputs = model.generate(
      input_ids=batch['input_ids'].to(device),
      max_length=max_length,
      do_sample=True,
      top_k=top_k,
      num_return_sequences=num_samples
    )

    samples = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    all_samples.extend(samples)

  return all_samples

def extract_passages(doc):
  passages = []
  for context in doc['contexts']:
    context_text = context['text']
    context_passages = []
    for sentence in context['sentences']:
      context_passages.append(context_text[sentence['start']:sentence['end']])
    passages.extend(context_passages)
  return passages

def expand_doc(doc_name):
  with open(os.path.join(collection_path, doc_name), 'r') as f:
    doc = json.load(f)
  # gather all passages for batch processing
  passages = extract_passages(doc)
  expanded_queries = expand(passages)
  passage_idx = 0
  for context in doc['contexts']:
    for sentence in context['sentences']:
      sentence['expanded_query'] = expanded_queries[passage_idx]
      passage_idx += 1

  with open(os.path.join(expanded_path, doc_name), 'w') as f:
    json.dump(doc, f)

print('Loading files...')
docs = [x for x in os.listdir(collection_path)]
print(f'Total files: {len(docs)}')
print(f'Generating queries...')
for doc in tqdm(docs, total=len(docs)):
  expand_doc(doc)

print('Done!')