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
    device = 'cuda:1'
    coco_json_path = "../datasets/coco/dataset_coco.json"
    with open(coco_json_path) as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    image_folder = '/home/lgs/data/coco/'
    train_out_path = '../datasets/coco_vit/coco_train_text_img.pkl'
    val_out_path = '../datasets/coco_vit/coco_val_text_img.pkl'
    test_out_path = '../datasets/coco_vit/coco_test_text_img.pkl'
    data_train_file_name = os.listdir(image_folder+"train2014")
    print("len(train)",len(data_train_file_name))
    data_val_file_name = os.listdir(image_folder + "val2014")
    train_all_embedding =[]
    val_all_embedding =[]
    test_all_embedding=[]

    for d in tqdm(data['images']):
        if d['split'] =='train':
            file_name = os.path.join(image_folder+"train2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image))
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = image
                train_all_embedding.append(data_dict)
        elif d['split'] =='test':
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image))
            data_dict = dict()
            raw_lst =[]
            token_lst = []
            for c in d['sentences']:
                raw_lst.append(c['raw'])
                token_lst.append(c['tokens'])
            data_dict['raw'] =raw_lst
            data_dict['tokens']=token_lst
            data_dict['img_id']=d['imgid']
            data_dict['img'] = image
            test_all_embedding.append(data_dict)
        elif d['split'] =='val' or d['split'] =='restval':
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image))
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = image
                val_all_embedding.append(data_dict)
        if len(train_all_embedding)%1000==0 and len(train_all_embedding)>0:
            with open(train_out_path, 'ab+') as f:
                for q in train_all_embedding:
                    pickle.dump(q, f)
            train_all_embedding = []
        if len(test_all_embedding)%1000==0 and len(test_all_embedding)>0:
            with open(test_out_path, 'ab+') as f:
                for q in test_all_embedding:
                    pickle.dump(q, f)
            test_all_embedding=[]
        if len(val_all_embedding)%1000==0 and len(val_all_embedding)>0:
            with open(val_out_path, 'ab+') as f:
                for q in val_all_embedding:
                    pickle.dump(q, f)
            val_all_embedding=[]
    if len(train_all_embedding) > 0:
        with open(train_out_path, 'ab+') as f:
            for q in train_all_embedding:
                pickle.dump(q, f)
    if len(test_all_embedding) > 0:
        with open(test_out_path, 'ab+') as f:
            for q in test_all_embedding:
                pickle.dump(q, f)
    if len(val_all_embedding) > 0:
        with open(val_out_path, 'ab+') as f:
            for q in val_all_embedding:
                pickle.dump(q, f)
    print('Done')



if __name__=='__main__':
    main()
