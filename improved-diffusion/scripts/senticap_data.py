import pdb

import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import nltk
import random


def main():
    device = 'cuda:0'
    coco_json_path = "../datasets/senticap/senticap_dataset.json"
    with open(coco_json_path) as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    image_folder = '/home/lgs/data/coco/'
    train_out_path = '../datasets/senticap/senticap_train_text_img.pkl'
    val_out_path = '../datasets/senticap/senticap_val_text_img.pkl'
    test_out_path_pos = '../datasets/senticap/senticap_test_text_img_pos.pkl'
    test_out_path_neg = '../datasets/senticap/senticap_test_text_img_neg.pkl'

    data_train_file_name = os.listdir(image_folder+"train2014")
    print("len(train)",len(data_train_file_name))
    train_all_embedding =[]
    val_all_embedding =[]
    test_all_embedding_pos=[]
    test_all_embedding_neg = []
    i=0
    j=0
    k=0
    for d in tqdm(data['images']):
        if d['split'] =='train':
            i=i+1
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = prefix
                if c["sentiment"]==1:
                    data_dict['label']=4
                else:
                    data_dict['label']=5
                train_all_embedding.append(data_dict)
        elif d['split'] =='test':
            j=j+1
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()

            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = prefix
                data_dict['img_id'] = d['imgid']
                if c["sentiment"] == 1:
                    data_dict['label'] = 4
                    test_all_embedding_pos.append(data_dict)
                else:
                    data_dict['label'] = 5
                    test_all_embedding_neg.append(data_dict)
        elif d['split'] =='val' or d['split'] =='restval':
            k=k+1
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = prefix
                if c["sentiment"] == 1:
                    data_dict['label'] = 4
                else:
                    data_dict['label'] = 5
                val_all_embedding.append(data_dict)
    with open(train_out_path, 'wb') as f:
        pickle.dump({"captions": train_all_embedding}, f)
    with open(test_out_path_pos, 'wb') as f:
        pickle.dump({"captions": test_all_embedding_pos}, f)
    with open(test_out_path_neg, 'wb') as f:
        pickle.dump({"captions": test_all_embedding_neg}, f)
    with open(val_out_path, 'wb') as f:
        pickle.dump({"captions": val_all_embedding}, f)
    print('Done')
    print(i,":",j,":",k)
    print("%0d train embeddings saved " % len(train_all_embedding))
    print("%0d test embeddings saved " % len(test_all_embedding_pos))
    print("%0d test embeddings saved " % len(test_all_embedding_neg))
    print("%0d val embeddings saved " % len(val_all_embedding))


if __name__=='__main__':
    main()
