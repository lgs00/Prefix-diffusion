from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import string
import json
from collections import Counter
from spacy.lang.en import English
import torch as th
import clip
import pdb
from tqdm import tqdm


class Flickr_Eval:
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self, gts, res, result_file):

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        with open(result_file, 'w') as fo:
            fo.write('flickr result\n')
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
                    with open(result_file, 'a') as fo:
                        fo.write(f'{m} = {round(sc, 4)}\n')
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
                with open(result_file, 'a') as fo:
                    fo.write(f'{method} = {round(score, 4)}\n')
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def sentence_pro(word_lsts, punc_move=True):
    for i in range(len(word_lsts)):
        word_lsts[i] = word_lsts[i].split('END')[0].replace('START', '').strip().capitalize()
        # word_lsts[i] = word_lsts[i].split('PAD')[0].replace('START', '').strip().capitalize()
        if punc_move:
            for j in string.punctuation:
                word_lsts[i] = word_lsts[i].replace(j, '')
    return word_lsts


def ref_pro(ref_lsts, punc_move=True):
    for ref_lst in ref_lsts:
        for i in range(len(ref_lst)):
            if punc_move:
                for j in string.punctuation:
                    ref_lst[i] = ref_lst[i].replace(j, '').strip()
    return ref_lsts


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
            d2.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            d3.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    return round(len(d1) / total_words, 4), round(len(d2) / total_words, 4), round(len(d3) / total_words, 4)


def calculate_distinct(cap_file, result_file, voc_size):
    captions = []
    with open(cap_file, "r", encoding='utf-8') as file:
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
    voc = round(len(counter) / voc_size, 4)
    dist1, dist2, dist3 = distinctness(captions)
    with open(result_file, 'a') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i + 1} = {dist_n}\n')
        fo.write(f'voc = {voc}\n')


def clip_choose(text_all, img_lst):
    clip_model, _ = clip.load('ViT-L/14', "cuda", jit=False)
    # clip_model, _ = clip.load('ViT-B/32', "cuda", jit=False)
    text_all_out = []
    for i in tqdm(range(len(text_all))):
        text_tmp = text_all[i]
        with th.no_grad():
            encoded_captions = [clip_model.encode_text(clip.tokenize(c).cuda()) for c in text_tmp]
        img_feature = img_lst[i] / img_lst[i].norm(dim=-1, keepdim=True)
        img_feature = img_feature.cuda()
        text_feature = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (th.cat(text_feature) @ img_feature.t()).squeeze().argmax().item()
        text_all_out.append([text_tmp[best_clip_idx]])
    return text_all_out