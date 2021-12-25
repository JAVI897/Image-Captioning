import clip
import statistics as stats
import torch
import skimage.io as io
import PIL.Image
import numpy as np
from utility import  get_inference_model, generate_caption
import json
import tensorflow as tf
from settings_inference import *
from settings import *
import pandas as pd
import os
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
from rouge_score import rouge_scorer
nltk.download('punkt')
nltk.download('wordnet')
D = torch.device

system_caption_file = 'system_caption_file_{}_{}.json'.format(CNN_TOP_MODEL, EMBED_DIM)


# Get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(tokernizer_path)
tokenizer = tokenizer.layers[1]

# Get model
model = get_inference_model(get_model_config_path)

# Load model weights
model.load_weights(get_model_weights_path)

with open(get_model_config_path) as json_file:
	model_config = json.load(json_file)

# Generate new captions from txt file


#Get ground truth captions validation
with open('COCO_dataset/karpathy_validation_captions.json') as json_file:
	captions_valid_test = json.load(json_file)


if not os.path.isfile(system_caption_file):
	captions = []
	predictions = []
	file = open('COCO_dataset/karpathy_valid_images.txt','r')
	for test_img in file.readlines():
		file_path, number_instance = test_img.split()
		_, name_img = file_path.split('/')
		name_img = 'COCO_dataset/val2014/'+ name_img
		caption_img = captions_valid_test[number_instance][:5]

		text_caption = generate_caption(name_img, model, tokenizer, model_config["SEQ_LENGTH"])
		print("PREDICT CAPTION : %s" %(text_caption))
		caption_img.append(text_caption)

		captions.append(caption_img)
		predictions.append([number_instance, caption_img])

	df = pd.DataFrame(captions, columns = ['caption 1', 'caption 2', 'caption 3', 'caption 4', 'caption 5', 'prediction'])
	df.to_csv('karpathy_test_predictions_{}_{}.csv'.format(CNN_TOP_MODEL, EMBED_DIM))

	coco_res_df = pd.DataFrame(predictions, columns = ['image_id', 'caption'])
	print('\nWriting predictions to file "{}".'.format(system_caption_file))
	coco_res_df.to_json(system_caption_file, orient='records')

df_results = pd.read_csv('karpathy_test_predictions_{}_{}.csv'.format(CNN_TOP_MODEL, EMBED_DIM))

def compute_metrics(df_results):
	N = 0
	BLEU_1 = 0
	BLEU_2 = 0
	BLEU_3 = 0
	BLEU_4 = 0
	BLEU_comb = 0
	METEOR = 0
	ROUGE_L = 0

	for index, row in df_results.iterrows():
		caption1, caption2, caption3, caption4, caption5, prediction = row['caption 1'], row['caption 2'], row['caption 3'], row['caption 4'], row['caption 5'], row['prediction']
		references = [word_tokenize(caption1), word_tokenize(caption2), word_tokenize(caption3), 
					  word_tokenize(caption4), word_tokenize(caption5) ]

		candidate = word_tokenize(prediction)

		# BLEU
		bleu_1 = bleu_score(references, candidate, weights=(1, 0, 0, 0))
		bleu_2 = bleu_score(references, candidate, weights=(0, 1, 0, 0))
		bleu_3 = bleu_score(references, candidate, weights=(0, 0, 1, 0))
		bleu_4 = bleu_score(references, candidate, weights=(0, 0, 0, 1))
		bleu = bleu_score(references, candidate, weights=(1/4, 1/4, 1/4, 1/4))

		N += 1
		BLEU_1 += bleu_1
		BLEU_2 += bleu_2
		BLEU_3 += bleu_3
		BLEU_4 += bleu_4
		BLEU_comb += bleu

		# METEOR
		#meteor = 0
		#for h, r in zip([candidate]*5, references):
		#    meteor += single_meteor_score(r, h)
		#meteor = meteor/5
		meteor = meteor_score.meteor_score(references, candidate)
		METEOR += meteor

		# ROUGE-L
		rouge_l = 0
		for h, r in zip([prediction]*5, [caption1, caption2, caption3, caption4, caption5]):
			scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
			score = scorer.score(r, h)['rougeL']
			score = score.fmeasure
			rouge_l += score
		rouge_l = rouge_l/5
		ROUGE_L += rouge_l

	BLEU_1 = BLEU_1/N
	BLEU_2 = BLEU_2/N
	BLEU_3 = BLEU_3/N
	BLEU_4 = BLEU_4/N
	BLEU_comb = BLEU_comb/N
	METEOR = METEOR/N
	ROUGE_L = ROUGE_L/N
	return BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L

def clipscore_karpathy_directories(dir_images, df_results, device, clip_model, preprocess):
	'''
	Code for CLIPScore (https://arxiv.org/abs/2104.08718)
	@inproceedings{hessel2021clipscore,
	  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
	  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
	  booktitle={EMNLP},
	  year={2021}
	}
	'''
	N = 0
	CLIP_SCORE = 0
	REFCLIP_SCORE = 0
	file = open(dir_images,'r')
	for ind_image, test_img in enumerate(file.readlines()):
		file_path, number_instance = test_img.split()
		_, name_img = file_path.split('/')
		name_img = 'COCO_dataset/val2014/'+ name_img
		image = io.imread(name_img)
		pil_image = PIL.Image.fromarray(image)
		image = preprocess(pil_image).unsqueeze(0).to(device)
		df_row = df_results.iloc[ind_image]
		caption1, caption2, caption3, caption4, caption5, prediction = df_row['caption 1'], df_row['caption 2'], df_row['caption 3'], df_row['caption 4'], df_row['caption 5'], df_row['prediction']  

		# CLIP-S
		print('[INFO] Computing ClipScore for image {}'.format(name_img))
		with torch.no_grad():
			tokens = clip.tokenize([prediction]).to(device).long()
			text_features = clip_model.encode_text(tokens).detach()
			image_features = clip_model.encode_image(image).to(device, dtype=torch.float32)
		clip_score = 2.5*np.clip( torch.cosine_similarity(text_features, image_features).cpu().numpy()[0], 0, None)
		
		# RefCLIPScore
		clip_score_references = []
		for ref in [caption1, caption2, caption3, caption4, caption5]:
			tokens_ref = clip.tokenize([ref]).to(device).long()
			ref_text_feature = clip_model.encode_text(tokens_ref).detach()
			ref_score = np.clip( torch.cosine_similarity(text_features, ref_text_feature).cpu().numpy()[0], 0, None)
			clip_score_references.append(ref_score)
		max_clip_score_references = max(clip_score_references)
		refclip_score = stats.harmonic_mean([clip_score, max_clip_score_references])

		N += 1
		CLIP_SCORE += clip_score
		REFCLIP_SCORE += refclip_score

	CLIP_SCORE = CLIP_SCORE / N
	REFCLIP_SCORE = REFCLIP_SCORE / N
	return CLIP_SCORE, REFCLIP_SCORE

def get_device(device_id: int) -> D:
		if not torch.cuda.is_available():
			return CPU
		device_id = min(torch.cuda.device_count() - 1, device_id)
		return torch.device(f'cuda:{device_id}')

BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L = compute_metrics(df_results)

is_gpu = True
CPU = torch.device('cpu')
CUDA = get_device
device = CUDA(0) if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
CLIP_SCORE, REFCLIP_SCORE = clipscore_karpathy_directories('COCO_dataset/karpathy_valid_images.txt', df_results, device, clip_model, preprocess)

df_scores = pd.DataFrame({'bleu_1': [BLEU_1], 'bleu_2': [BLEU_2], 
							  'bleu_3': [BLEU_3], 'bleu_4': [BLEU_4],
							  'BLEU_comb' : [BLEU_comb], 'METEOR' : [METEOR],
							  'ROUGE_L' : [ROUGE_L], 'CLIPScore' : [CLIP_SCORE],
							  'REFCLIP_SCORE' : [REFCLIP_SCORE] })

df_scores.to_csv('scores_karpathy_test_predictions_{}_{}.csv'.format(CNN_TOP_MODEL, EMBED_DIM))

print('[INFO] Scores. Bleu 1 = {:.4} Bleu 2 = {:.4} Bleu 3 = {:.4} Bleu 4 = {:.4} Bleu_comb = {:.4} METEOR = {:.4} ROUGE_L = {:.4} CLIPScore = {:.4} RefCLIPScore = {:.4}'.format(BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L, CLIP_SCORE, REFCLIP_SCORE))