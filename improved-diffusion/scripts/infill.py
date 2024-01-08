"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import stanza
import spacy_stanza
import numpy as np
import torch as th
from torch import nn
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, '../transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree,Classifier_image
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English
# sys.path.append("clip/")
# from model_creation import create_clip_model
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import clip
import pdb
from PIL import Image
import skimage.io as io
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
th.cuda.current_device()
th.cuda._initialized = True

def main():
    set_seed(101)
    args = create_argparser().parse_args()
    # pdb.set_trace()
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')  # args.clip_denoised=False
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()
    args.batch_size=10
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')


    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    # args.modality=e2e-tgt,experiment=random,model_name_or_path=predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None
    # in_channel=16 os.path.split(model_path)[0]=diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):    # args.training_mode=e2e
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()  # model_embs=Embedding(821,16)
    model3 = get_weights(model_embs, args)  # model3 = Embedding(821,16)
    logger.log('load the partial sequences')
    if args.partial_seq:    # 空
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']
    elif args.partial_seq_file:     # 空
        # implies that we should read from the files
        nlp = English()
        tokenizer_spacy = nlp.tokenizer
        print(f'reading from the file {args.partial_seq_file}', '-*'*20)
        with open(args.partial_seq_file, 'r') as f:
            sent_lst = json.load(f)
        partial_seq = []
        partial_seq_idx = []
        for idx, (key, val) in enumerate(sent_lst.items()):
            if idx < int(args.start_idx) or idx > int(args.end_idx):
                continue
            partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
            word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
            partial_seq_ = " ".join(word_lst)
            print(partial_seq_, idx)
            partial_seq.append(partial_seq_)
            partial_seq_idx.append(str(idx))
    else:
        partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                       'Alimentum , situated by the river , is quite child friendly .']
        partial_seq_idx = ['0', '1']
    # else:  generate them by randomly preturbing the inputs data.
    # args.modality=e2e-tgt
    if args.modality in ['synth', 'pos']:   # false
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = tokens2id['END']
        print(f'pad token = {todo_pad_token}')
        encoded_partial_seq = [th.LongTensor([tokens2id[x] for x in seq.split()]) for seq in partial_seq]
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    elif args.modality in ['e2e-tgt', 'roc', 'roc-aug']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq] # j句子的tokenize
        if args.eval_task_ == 'infill':
            todo_pad_token = tokens2id['PAD']
            print(f'pad token = {todo_pad_token}')
            partial_seq = [(b, a) for (a,b) in zip(partial_seq, partial_seq_idx)]
            pass
        elif args.eval_task_.startswith('control'):
            # right_pad = th.empty(args.tgt_len+2).fill_(pad_token).long()
            # TO FIX... IMPORTANT.
            if 'length' not in args.eval_task_:
                right_pad = th.empty(32).fill_(pad_token).long()    # right_pad=tensor([64个3])

                encoded_partial_seq = [th.cat([right_pad], dim=0)]  # encoded_partial_seq=[tensor([64个3])]
                encoded_partial_seq[0][0] = tokens2id['START']
                # args.tgt_len=15
                encoded_partial_seq[0][args.tgt_len] = tokens2id['END']     # encoded_partial_seq=[tensor([首位和目标长度位是1，其它是3])]
            if args.eval_task_ == 'control_pos':
                model_control = Classifier_image.from_pretrained('../classifier_models/classify_923/').cuda()
                pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
                pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                           'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                           'PUNCT', 'SYM', 'X']
                for x in pos_lst:
                    pos_vocab[x] = len(pos_vocab)
                pos_vocab_rev = {v:k for k,v in pos_vocab.items()}
                ################33
                control_label_lst = []
                with open('../datasets/control_target/target_pos.json', 'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print(control_label_lst[:2])
                control_constraints = []    # control_label_lst=[{pos:**,words_:[**]},{},...] 其中pos是词性，words_是token列表
                for label_class_dict in control_label_lst[:50]:#control_label_lst[:100]:
                    label_class = label_class_dict['pos']
                    words_ = label_class_dict['words_']
                    label_class = [pos_vocab.get(x, pos_vocab['UNK']) for x in label_class] # 将词性转换为标签
                    label_class = label_class + [pos_vocab['PAD']] * (32 - len(label_class))    # 将词性标签填充到64个
                    label_ids = th.LongTensor(label_class).unsqueeze(0) # 增加一个维度
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn4, debug_lst, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1),   # label_ids.expand(args.batch_size, -1).shape=64*64
                                                   0.1)     # 为langevin_fn4固定后面几个参数
                    control_constraints.append((langevin_fn_selected, label_class_dict['pos']))
                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)     # len(partial_seq)=50
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)
            elif args.eval_task_ == 'control_length':
                control_length_lst = list(range(10, 41)) #[40] #[10, 20, 30]
                control_constraints = []
                for target_length in control_length_lst:
                    encoded_partial_seq = [th.LongTensor([0])]
                    print(encoded_partial_seq)  # [tensor([0])]
                    assert len(encoded_partial_seq) == 1
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])# args.image_size=8,right_length=63
                    # right_length = args.tgt_len - len(encoded_partial_seq[0])
                    # assert args.tgt_len > len(encoded_partial_seq[0])
                    # right_pad=tensor([63个-1]) right_length=63,len(encoded_partial_seq[0])=1
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    print(right_pad, right_length, len(encoded_partial_seq[0]))
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    encoded_partial_seq[0][target_length - 1] = tokens2id['END']
                    # encoded_partial_seq[0][target_length] = tokens2id['START']
                    # encoded_partial_seq[0]开头是0，目标长度位是1(end)，其余是-1. todo_pad_token=-1
                    print(encoded_partial_seq[0], todo_pad_token)
                    # partial_mask.shape=64*64,第一个是batch_size,除了首位和目标end位，其它都是true
                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    # print(partial_mask[0])
                    # 10/0
                    label = encoded_partial_seq[0]
                    label_ids = th.tensor(label).unsqueeze(0)
                    label_ids = label_ids.masked_fill(label_ids == todo_pad_token, 3)   # 将-1的pad换成了3，即首位是0，目标长度位是1，其余是3
                    tgt_embs = model3.cuda()(label_ids.cuda())  # 1*64*16,在下一步扩展成为64*64*16
                    langevin_fn_selected = partial(langevin_fn_length, 0.01, diffusion, partial_mask, model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, (str(target_length),)))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)    # length中len(partial_seq)=31
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    # else: text, using huggingface tokenizer.
    logger.log("sampling...")
    sample_dict = {}

    # model3 = get_weights(model_embs, args)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    img_path = "image/7.jpg"
    image = io.imread(img_path)
    pil_image = Image.fromarray(image)
    image_feature = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with th.no_grad():
        image_feature = clip_model.encode_image(image_feature).to(device)
    if True:
        for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq) :
            all_images = []
            all_labels = []
            # encoded_seq的size是64，开头是0，目标长度位是1，其余是-1(在pos中是3)
            print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape') # num_samples=50,encoded_seq.shape=torch.size([64])

            while len(all_images) * args.batch_size < args.num_samples: # batch_size=64
                model_kwargs = {}
                encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)   # 扩展成64*64，一次生成64条句子
                print(model_embs.weight.device, encoded_seq.device) # cuda0,cpu
                partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)   # todo_pad_token=-1,partial_mask_temp.shape=64*64
                # encoded_seq[encoded_seq == todo_pad_token] = 0
                encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)  # encoded_seq开始是0，目标长度位是1，其余的3(pad)
                encoded_seq_hidden = model_embs(encoded_seq.cuda()) # encoded_seq.shape=64*64,编码后成64*64*16,无用
                #seqlen = encoded_seq.size(1)    # 64
                seqlen=32
                # args.model_arch=transformer）
                if args.model_arch == '1d-unet':
                    encoded_seq_hidden = encoded_seq_hidden.permute(0, 2, 1)
                    partial_mask = partial_mask_temp.unsqueeze(1).expand(-1, args.in_channel, -1)
                    sample_shape = (args.batch_size, args.in_channel, seqlen)
                else:
                    # partial_mask.shape=64*64*16,在开头和目标结束位是false,其余是true，in_channel=16
                    partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
                    sample_shape = (args.batch_size, seqlen, args.in_channel, )     # sample_shape=64*64*16
                # print(partial_mask, encoded_seq_hidden.shape)
                if args.eval_task_.startswith('control'):
                    langevin_fn_selected, label_class_attributes = control_helper
                    print('-*'*200, label_class_attributes, '-*'*200)   # 控制目标，例如长度，则为（'10'，）
                    # loop_func_ = diffusion.p_sample_loop_langevin_progressive
                    if args.use_ddim:   # true
                        loop_func_ = diffusion.ddim_sample_loop_progressive
                    else:
                        loop_func_ = diffusion.p_sample_loop_progressive

                    # setup guidance function for clip model
                    # cond_fn = None
                    # langevin_fn_selected=None
                    for sample in loop_func_(
                            model,
                            sample_shape,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                            #                     label_ids.expand(args.batch_size, -1), 0.1),
                            clip_denoised=args.clip_denoised,   # clip_denoised=false
                            model_kwargs=model_kwargs,  # model_kwargs={},w为空字典
                            device=encoded_seq_hidden.device,   #
                            langevin_fn=langevin_fn_selected,   # langevin_fn_selected中的tgt_embd 和encoded_seq_hidden一样
                            eta=args.eta,   # eta=1.0
                            image_feature = image_feature
                            # cond_fn=cond_fn,
                            # langevin_func=partial(langevin_func, model_control,
                            #                       label_ids.expand(args.batch_size, -1), 0.01),
                    ):
                        final = sample["sample"]    # sample["sample"].shape=64*64*16,sample["pre_xstart"]=64*64*16
                sample = final  # sample.shape=64*64*16
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                logger.log(f"created {len(all_images) * args.batch_size} samples")
                break
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if args.verbose == 'pipe':
                sample_dict[tuple(label_class_attributes)] = arr
                print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')
            break
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    dist.barrier()
    logger.log("sampling complete")
    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
        for k, v in sample_dict.items():
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()     # 50*64*16
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab 50*64*821
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e     # len(word_lst=50,每个条件生成50个sample
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict

    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.json")
        fout = open(out_path_pipe, 'w')
        result_dict = decode_helper(args, sample_dict, diff_model=model)
        for k, word_lst in result_dict.items():
            print({k:word_lst}, file=fout)
        fout.close()
        print(f'written the decoded output to {out_path_pipe}')
        out_path2 = out_path_pipe
    elif args.verbose == 'yes':
        if diffusion.training_mode.startswith('e2e'):
            word_lst_e2e = []
            print('decoding for e2e', )
            print(sample.shape)
            x_t = sample
            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            for seq in cands.indices:
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                word_lst_e2e.append(tokens)
            word_lst = word_lst_e2e
        else:
            logger.log('decode by rounding. ')
            print('load_models')
            set_seed(101)
            print(os.path.split(args.model_path)[0])
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel, os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.txt")
        fout = open(out_path2, 'w')
        for (xx) in zip( word_lst):
            print(xx[0], file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')
    args.out_path2 = out_path2
    return args

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="generation_out",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
              f"--model_path {args.model_path} " \
              f"--modality {args.modality}  --experiment random " \
              f"--model_name_or_path {model_name_path} " \
              f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    import numpy as np
    if args.verbose != 'pipe':
        eval(args)

