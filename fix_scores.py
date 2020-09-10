
fixed_rows = []
with open('runs/expert/baseline_pass_full_doc_rerank', 'r') as fi:
	for line in fi:
		line = line.strip().split()
		if line:
			fixed_score = -float(line[4])
			line[4] = str(fixed_score)
			fixed_rows.append('\t'.join(line))

with open('runs/expert/baseline_pass_full_doc_rerank', 'w') as fo:
	for row in fixed_rows:
		fo.write(row + '\n')
