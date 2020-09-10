from pyserini.search import SimpleSearcher


index_path = 'indices/expert/baseline_doc'
query = 'what is the origin of COVID-19'
print(f'Query: {query}')
print(f'Index: {index_path}')
searcher = SimpleSearcher(index_path)
hits = searcher.search(query)

for i, hit in enumerate(hits[:10]):
    print(f'{i+1:2} {hit.docid:15} {hit.score:.5f}')
