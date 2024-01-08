#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
from collections import Counter
from spacy.lang.en import English
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from eval_metrics import sentence_pro
from eval_metrics import ref_pro


def distinctness(sentences):
    d1 = set()
    d2 = set()
    d3 = set()
    total_words = 0
    for sentence in sentences:
        o = sentence.split(' ')
        total_words += len(o)
        d1.update(o)
        for i in range(len(o) - 1):
            d2.add(o[i] + '_' + o[i+1])
        for i in range(len(o) - 2):
            d3.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
    return round(len(d1) / total_words,4), round(len(d2) / total_words,4), round(len(d3) / total_words,4)


def main():
    cap_file = "/home/lgs/diffusion/improved-diffusion/generation_out/coco/1018/diff_model_coco_1018_model200000.pt.json"
    result_file = "/home/lgs/diffusion/improved-diffusion/generation_out/coco/1018/diff_model_coco_1018_model200000.pt_coco_result.txt"
    captions=[]
    with open(cap_file,"r",encoding='utf-8') as file:
        generation_df = json.load(file)
    for row in generation_df:
        captions.append(row["caption"])
    nlp = English()
    tokenizer = nlp.tokenizer
    word_lst = []
    for sentence in captions:
        sen_lst = [x.text for x in tokenizer(sentence)]
        word_lst.append(sen_lst)
    counter = Counter()
    for input_ids in word_lst:
        counter.update(input_ids)
    voc = round(len(counter)/23532,4)
    dist1,dist2,dist3=distinctness(captions)
    with open(result_file,'a')  as fo:
        for i,dist_n in enumerate([dist1,dist2,dist3]):
            fo.write(f'dist-{i+1} = {dist_n}\n')
        fo.write(f'voc = {voc}')

if __name__=="__main__":
    # main()
    out_path_pipe = "/home/lgs/diffusion/improved-diffusion/generation_out/coco/1019/diff_model_coco_1019_model200000.pt.json"
    out_path_pipe_t = "/home/lgs/diffusion/improved-diffusion/generation_out/coco/1019/diff_model_coco_1019_model200000.pt_up.json"
    result_lst = []
    with open(out_path_pipe) as f:
        data = json.load(f)
    for dict_row in data:
        out_dict = dict()
        out_dict['image_id'] = dict_row['image_id']
        out_dict['caption'] = dict_row['caption'].capitalize()
        result_lst.append(out_dict)
    with open(out_path_pipe_t, 'w', encoding='utf-8') as f:
        json.dump(result_lst, f)
    annFile = "../datasets/coco/test_ref_coco.json"
    coco = COCO(annFile)
    cocoRes = coco.loadRes(out_path_pipe_t)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    result_file = "/home/lgs/diffusion/improved-diffusion/generation_out/coco/1019/diff_model_coco_1019_model200000.pt_coco_result.txt"
    for metric, score in cocoEval.eval.items():
        with open(result_file, 'a') as fo:
            fo.write(f'up-{metric} = {round(score, 4)}\n')

