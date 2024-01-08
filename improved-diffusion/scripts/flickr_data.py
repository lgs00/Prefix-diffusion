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
    coco_json_path = "../datasets/flickr/dataset_flickr30k.json"
    with open(coco_json_path) as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    image_folder = '/home/lgs/data/flickr30k-images/'
    # train_out_path = '../datasets/flickr/flickr_train_text_img.pkl'
    # val_out_path = '../datasets/flickr/flickr_val_text_img.pkl'
    # test_out_path = '../datasets/flickr/flickr_test_text_img.pkl'
    test_out_path = '/home/lgs/CapDec-main/data/flickr/flickr_test_text_img.pkl'
    data_train_file_name = os.listdir(image_folder)
    print("len(train)",len(data_train_file_name))
    train_all_embedding =[]
    val_all_embedding =[]
    test_all_embedding=[]

    for d in tqdm(data['images']):
        if d['split'] =='train':
            continue
            file_name = os.path.join(image_folder,d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = prefix
                train_all_embedding.append(data_dict)
        elif d['split'] =='test':
            file_name = os.path.join(image_folder,d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            data_dict = dict()
            raw_lst =[]
            token_lst = []
            for c in d['sentences']:
                raw_lst.append(c['raw'])
                token_lst.append(c['tokens'])
                img_id = c['imgid']
            data_dict['raw'] =raw_lst
            data_dict['tokens'] = token_lst
            data_dict['img_id'] = img_id
            data_dict['img'] = prefix
            test_all_embedding.append(data_dict)
        elif d['split'] =='val' or d['split'] =='restval':
            continue
            file_name = os.path.join(image_folder,d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['tokens'] = c['tokens']
                data_dict['img'] = prefix
                val_all_embedding.append(data_dict)
    # with open(train_out_path, 'wb') as f:
    #     pickle.dump({"captions": train_all_embedding}, f)
    with open(test_out_path, 'wb') as f:
        pickle.dump({"captions": test_all_embedding}, f)
    # with open(val_out_path, 'wb') as f:
    #     pickle.dump({"captions": val_all_embedding}, f)
    print('Done')
    print("%0d train embeddings saved " % len(train_all_embedding))
    print("%0d test embeddings saved " % len(test_all_embedding))
    print("%0d val embeddings saved " % len(val_all_embedding))


if __name__=='__main__':
    main()
