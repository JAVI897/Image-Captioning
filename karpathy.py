from utility import  get_inference_model, generate_caption
import json
import tensorflow as tf
from settings_inference import *
from settings import *
import pandas as pd
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import nltk
from rouge_score import rouge_scorer
nltk.download('punkt')
nltk.download('wordnet')

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
    #references = [caption1.replace('.', '').split(), caption2.replace('.', '').split(), 
    #              caption3.replace('.', '').split(), caption4.replace('.', '').split(), 
    #              caption5.replace('.', '').split()]
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
    meteor = 0
    for h, r in zip([candidate]*5, references):
        meteor += single_meteor_score(r, h)
    meteor = meteor/5
    METEOR += meteor

    # ROUGE-L
    rouge_l = 0
    for h, r in zip([candidate]*5, references):
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

df_scores = pd.DataFrame({'bleu_1': [BLEU_1], 'bleu_2': [BLEU_2], 
                          'bleu_3': [BLEU_3], 'bleu_4': [BLEU_4],
                          'BLEU_comb' : [BLEU_comb], 'METEOR' : [METEOR] })

df_scores.to_csv('scores_karpathy_test_predictions_{}_{}.csv'.format(CNN_TOP_MODEL, EMBED_DIM))

print('[INFO] Scores. Bleu 1 = {:.4} Bleu 2 = {:.4} Bleu 3 = {:.4} Bleu 4 = {:.4} Bleu_comb = {:.4} METEOR = {:.4} ROUGE_L = {:.4}'.format(BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L))