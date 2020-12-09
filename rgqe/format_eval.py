
import argparse
import json


def write_run(results, output_path, output_name):
	with open(output_path, 'w') as f:
		for question_id, q_results in results.items():
			rank = 1
			for answer in q_results:
				answer_id = answer['answer_id']
				answer_score = answer['score']
				f.write(f'{question_id}\tQ0\t{answer_id}\t{rank}\t{answer_score:.8f}\t{output_name}\n')
				rank += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--results_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	results_path = args.results_path
	output_path = args.output_path

	output_name = output_path.split('/')[-1].replace('.txt', '').replace('.pred', '')

	with open(results_path) as f:
		results = json.load(f)
		rerank_results = results['answers']

	write_run(rerank_results, output_path, output_name)





