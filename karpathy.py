from utility import  get_inference_model, generate_caption
import json
import tensorflow as tf
from settings_inference import *
from settings import *
import pandas as pd
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
reference_caption_file = 'COCO_dataset/annotations/captions_val2014.json'
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
    for test_img in lines:
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


coco = COCO(reference_caption_file)
coco_system_captions = coco.loadRes(system_caption_file)
coco_eval = COCOEvalCap(coco, coco_system_captions)
coco_eval.params['image_id'] = coco_system_captions.getImgIds()

coco_eval.evaluate()

print('\nScores:')
print('=======')
for metric, score in coco_eval.eval.items():
    print('{}: {:.3f}'.format(metric, score))