
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



tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    '/users/max/data/models/t5/docT5query_base/model.ckpt-1004000', from_tf=True, config=config)
model.to(device)
model.eval()


def expand(passage_text, num_samples=3, max_length=64, top_k=10):
  input_ids = tokenizer.encode(
    f'{passage_text} </s>',
    return_tensors='pt'
  ).to(device)

  outputs = model.generate(
      input_ids=input_ids,
      max_length=max_length,
      do_sample=True,
      top_k=top_k,
      num_return_sequences=num_samples
  )

  samples = []
  for i in range(num_samples):
    sample_txt = tokenizer.decode(outputs[i], skip_special_tokens=True)
    samples.append(sample_txt)
  return samples

def expand_doc(doc_name):
  with open(os.path.join(collection_path, doc_name), 'r') as f:
    doc = json.load(f)

  for context in doc['contexts']:
    context_text = context['text']
    expanded_queries = expand(context_text)
    context['expanded_queries'] = expanded_queries

  with open(os.path.join(expanded_path, doc_name), 'w') as f:
    json.dump(doc)

print('Loading files...')
docs = [x for x in os.listdir(collection_path)]
print(f'Total files: {len(docs)}')
print(f'Generating queries...')
for doc in tqdm(docs, total=len(docs)):
  expand_doc(doc)

print('Done!')