#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
from tqdm import tqdm
# resFile1= "/home/lgs/diffusion/improved-diffusion/generation_out/flickr/927_ema/diff_model_flickr_927_ema_0.9999_200000.pt.json"
# with open(resFile1,'r') as f:
#     data = json.load(f)
# result_lst = []
# for key, value in data.items():
#     out_dict = dict()
#     out_dict['image_id'] = int(key)
#     out_dict['caption'] = "".join(value).strip()
#     result_lst.append(out_dict)
# resFile= "/home/lgs/diffusion/improved-diffusion/generation_out/flickr/927_ema/captions_gen.json"
# with open(resFile, 'w', encoding='utf-8') as f:
#     json.dump(result_lst, f)
#
# annFile= "/home/lgs/diffusion/datasets/flickr/test_ref_flickr.json"
#
# # create coco object and cocoRes object
# coco = COCO(annFile)
# cocoRes = coco.loadRes(resFile)
#
# cocoEval = COCOEvalCap(coco, cocoRes)
#
# cocoEval.params['image_id'] = cocoRes.getImgIds()
#
# cocoEval.evaluate()
# # print output evaluation scores
# result_file = "/home/lgs/diffusion/improved-diffusion/generation_out/flickr/927_ema/off_result.txt"
# for metric, score in cocoEval.eval.items():
#     with open(result_file,'a') as fo:
#         fo.write(f'{metric} = {round(score,4)}\n')


# coco_json_path = "/root/nlp/diffusion/datasets/flickr/dataset_flickr30k.json"
# with open(coco_json_path) as f:
#     data = json.load(f)
# print("%0d captions loaded from json " % len(data))
#
# test_all =[]
# info_lst =[]
# for d in tqdm(data['images']):
#     if d['split'] == 'test':
#         info_dict = dict()
#         for c in d['sentences']:
#             data_dict = dict()
#             data_dict["image_id"] = c["imgid"]
#             data_dict["id"] = c["sentid"]
#             data_dict["caption"]= c["raw"]
#
#             test_all.append(data_dict)
#         info_dict["id"] = d["imgid"]
#         info_dict["file_name"] = d["filename"]
#         info_lst.append(info_dict)
#
# with open("/root/nlp/diffusion/datasets/flickr/test_ref_flickr.json",'w') as f:
#     json.dump({"images": info_lst,"annotations":test_all},f)
#
# print(len(test_all))

import torch
import torch.nn as nn
# from timm.models.layers import DropPath, to_2tuple
# from improved_diffusion.transformer_token import TransformerMapper, Merge_attention
# from tqdm import tqdm
# import json
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# img = torch.randn(64,3,224,224)
# model = PatchEmbed(224,16,3,768)
# y = model(img)
# print(y.shape)
#transformer_model = nn.Transformer(d_model=5,nhead=1, num_encoder_layers=5, num_decoder_layers = 0, dim_feedforward=1024, activation='relu', dropout=0.5)
# img = torch.randn(2,2,16)
# text = torch.randn(2,3,16)
# token_type_embeddings = torch.nn.Embedding(2,  5)
# type_text=token_type_embeddings(torch.zeros_like(text[:,:]).long())
# type_img=token_type_embeddings(torch.ones_like(img[:,:]).long())
# y=torch.arange(2).expand((1, -1))
# print(y.shape)
# print(type_img.shape)
# print(type_text.shape)
# out = transformer_model(src, tgt)
# print(out)
# print(out.shape)
# model = Merge_attention(h=4,d_model=16)
# y = model(img,text,text)

# /home/lgs/diffusion/improved-diffusion/test_coco_result.txt


def main():
    coco_json_path = "/home/lgs/data/caption_datasets/dataset_coco.json"
    with open(coco_json_path) as f:
        data = json.load(f)
    for d in tqdm(data['images']):
        if d['imgid'] ==3842:
            print("file: ", d['filename'])
            for j in d['sentences']:
                print("cap: ",j["raw"])


if __name__=="__main__":
    main()