

qrel_input = 'data/expert/qrels-covid_d4_j3.5-4.txt'
doc_type = 'expert'
qrel_name = 'baseline_doc'
qrel_path = f'qrels/{doc_type}/{qrel_name}'

with open(qrel_path, 'w') as fo:
	with open(qrel_input) as f:
		for line in f:
			line = line.strip().split()
			if line:
				query_id, _, doc_id, dq_rank = line
				query_id = int(query_id)
				dq_rank = int(dq_rank)
				if dq_rank > 1:
					dq_rank = 1
				line = f'{query_id}\tQ0\t{doc_id}\t{dq_rank}\n'
				fo.write(line)
