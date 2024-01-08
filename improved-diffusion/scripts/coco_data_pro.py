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
    image_folder = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/yangjunjie/data/coco/coco2014/'
    train_out_path = '../datasets/coco/coco_train_text_img.pkl'
    val_out_path = '../datasets/coco/coco_val_text_img.pkl'
    test_out_path = '../datasets/coco/coco_test_text_img.pkl'
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
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['img'] = prefix
                file_name0 = random.choice(data_train_file_name)
                while file_name0 == d['filename']:
                    file_name0 = random.choice(data_train_file_name)
                image0 = io.imread(os.path.join(image_folder+"train2014",file_name0))
                image0 = preprocess(Image.fromarray(image0)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix0 = clip_model.encode_image(image0).cpu()
                data_dict['img0'] = prefix0
                train_all_embedding.append(data_dict)
        elif d['split'] =='test':
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['img'] = prefix
                file_name0 = random.choice(data_val_file_name)
                while file_name0 == d['filename']:
                    file_name0 = random.choice(data_val_file_name)
                image0 = io.imread(os.path.join(image_folder+"val2014",file_name0))
                image0 = preprocess(Image.fromarray(image0)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix0 = clip_model.encode_image(image0).cpu()
                data_dict['img0'] = prefix0
                test_all_embedding.append(data_dict)
        elif d['split'] =='val' or d['split'] =='restval':
            file_name = os.path.join(image_folder+"val2014",d['filename'])
            image = io.imread(file_name)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            for c in d['sentences']:
                data_dict = dict()
                data_dict['raw'] = c['raw']
                data_dict['img'] = prefix
                file_name0 = random.choice(data_val_file_name)
                while file_name0 == d['filename']:
                    file_name0 = random.choice(data_val_file_name)
                image0 = io.imread(os.path.join(image_folder+"val2014",file_name0))
                image0 = preprocess(Image.fromarray(image0)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix0 = clip_model.encode_image(image0).cpu()
                data_dict['img0'] = prefix0
                val_all_embedding.append(data_dict)
    with open(train_out_path, 'wb') as f:
        pickle.dump({"captions": train_all_embedding}, f)
    with open(test_out_path, 'wb') as f:
        pickle.dump({"captions": test_all_embedding}, f)
    with open(val_out_path, 'wb') as f:
        pickle.dump({"captions": val_all_embedding}, f)
    print('Done')
    print("%0d train embeddings saved " % len(train_all_embedding))
    print("%0d test embeddings saved " % len(test_all_embedding))
    print("%0d val embeddings saved " % len(val_all_embedding))







if __name__=='__main__':
    main()
