import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--network", type=str, default='VGG')
con = parser.parse_args()

def configuration():
	output_predictions = 'karpathy_test_predictions_{}_512.csv'.format(con.network)
	output_candidates = 'candidates_karpathy_test_predictions_{}_512.csv'.format(con.network)
	config ={
			 'output_predictions': output_predictions,
			 'output_candidates':output_candidates
			 }
	return config

def main():
	config = configuration()
	df_results = pd.read_csv(config['output_predictions'])
	with open('refs.json') as json_file:
		refs = json.load(json_file)
	candidates = [[k, None] for k, _ in refs.items()]
	cont = 0
	for index, row in df_results.iterrows():
		candidates[cont][1] = row['prediction']
		candidates[cont] = tuple(candidates[cont])
		cont += 1
	candidates = dict(candidates)

	with open(config['output_candidates'], 'w') as fp:
		json.dump(candidates, fp)

if __name__ == '__main__':
	main()