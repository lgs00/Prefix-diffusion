import torch, wandb
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Config, GPT2Model, BertPreTrainedModel, BertModel, \
    GPT2LMHeadModel, BertForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, NextSentencePredictorOutput
from transformers.models.bert.modeling_bert import BertOnlyNSPHead
from torch import nn
from transformers import Trainer, GPT2PreTrainedModel, PreTrainedModel, DataCollator, TrainingArguments, EvalPrediction, \
    TrainerCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import collections
from transformers.utils import logging
from  transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
import pdb

logger = logging.get_logger(__name__)

class Classifier_GPT2(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class Classifier_Times(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class AutoEncoderWithNoise(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class GPT2LMHeadModelCompress(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class BERTModelCompress(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class GPT2VAE(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class AR_for_cont(GPT2PreTrainedModel):
    print("ar for cont")


class Classifier_Consistency(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class Classifier_POS2(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class Classifier_Tree(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]


class Classifier_POS(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = BertModel(config)
        # input_emb_dim=16
        self.transformer.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        # self.pos_wte = nn.Embedding(config.pos_vocab_size, config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)
        # config.hidden_size=768
        # 上采样，16-64-768
        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.hidden_size))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)    # train_diff_steps=200,hidden_size=768

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        self.lm_head2 = nn.Linear(config.hidden_size, config.pos_vocab_size, bias=False)    # pos_vocab_size=21


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # return_dict=true
        labels = pos_ids
        # print(input_ids.shape, 'input_ids', )     # input_ids.shape=10*64,10是batch_size，4块卡，每块卡的batch_size=10

        if input_embs is None:
            # print(input_ids.shape, pos_ids.shape)     # pos_ids.shape=10*64
            input_embs = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs.shape=10*64*16

        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t, train_diff_steps=200
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)  # t为10个int，即batch_size个T(采样步数)
                t_mask = (t >= 0)   # 10个true
                input_embs_rand = self.diffusion.q_sample(input_embs, t)    # input_embs_rand.shape=10*64*16,得到xt
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps  # t.shape=torch.size([10])
                # self.time_embeddings(t).shape=10*768
                time_emb = self.time_embeddings(t).unsqueeze(1)     # time_emb.shape=10*1*768

                # t = torch.randint(0, self.train_diff_steps+1, (input_embs.shape[0],)).to(input_embs.device)
                # t_mask = (t < self.train_diff_steps)
                # input_embs_rand = self.diffusion.q_sample(input_embs, t)
                # input_embs[t_mask] = input_embs_rand[t_mask]
                # print(input_embs.shape, t[:3])
                # print(self.time_embeddings, t)
                # time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None:
            # print(t, input_embs.shape, 'should see this')
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        input_embs = self.up_proj(input_embs)   # input_embs=10*64*768  上采样

        if t_aware: # true
            input_embs = torch.cat([time_emb, input_embs], dim=1)   # input_embs=10*65*768


        transformer_outputs = self.transformer(
            past_key_values=past_key_values,    # none
            attention_mask=attention_mask,  # none
            token_type_ids=token_type_ids,  # none
            position_ids=position_ids,  # none
            head_mask=head_mask,    # none
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,    # none
            encoder_attention_mask=encoder_attention_mask,  # none
            use_cache=use_cache,    # none
            output_attentions=output_attentions,    # None
            output_hidden_states=output_hidden_states,  # none
            return_dict=return_dict,    # true
        )
        # transformer_outputs[0].shape=10*65*768
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ] # hidden_states.shape=10*65*768
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel: # false
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)    # lm_logits.shape=10*64*21
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if pos_ids is not None:     # pos_ids.shape=10*64
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = labels   # labels=pos_ids
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))    # shift_logits.size(-1)=21

        if not return_dict: # return_dict=ture
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,    # none
            hidden_states=transformer_outputs.hidden_states,    # none
            attentions=transformer_outputs.attentions,  # none
            cross_attentions=transformer_outputs.cross_attentions,  # none
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_image(BertPreTrainedModel):

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = BertModel(config)
        # input_emb_dim=16
        self.transformer.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        # self.pos_wte = nn.Embedding(config.pos_vocab_size, config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)
        # config.hidden_size=768
        # 上采样，16-64-768
        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4),nn.ReLU(),
                                     nn.Linear(config.input_emb_dim * 4, config.hidden_size))

        #self.img_up_proj = nn.Sequential(nn.Linear(512, 640), nn.ReLU(), nn.Linear(640, config.hidden_size))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)    # train_diff_steps=200,hidden_size=768

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        #self.lm_head2 = nn.Linear(512, 2, bias=False)    # pos_vocab_size=21
        self.lm_head2 = nn.Sequential(nn.Dropout(p=0.1),nn.Linear(2*512,512),nn.ReLU(),nn.Linear(512,2,bias=False))
        self.down_proj = nn.Sequential(nn.Linear(32,1)) 
        self.down_proj2 = nn.Sequential(nn.Linear(768,512))
        #self.mediu_classify = nn.Sequential(nn.Linear(32*16, 512), nn.Tanh(), nn.Linear(512,2,bias=False))
        #self.mediu_pool = nn.Sequential(nn.AdaptiveAvgPool2d((16,768)),nn.AdaptiveAvgPool2d((1,768)))


        # def get_output_embeddings(self):
    #     return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
            img=None,
            img0=None,
            img_label=None,
    ):
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # return_dict=true
        # print(input_ids.shape, 'input_ids', )     # input_ids.shape=10*64,10是batch_size，4块卡，每块卡的batch_size=10
        labels=img_label
        if input_embs is None:
            # print(input_ids.shape, pos_ids.shape)     # pos_ids.shape=10*64
            input_embs = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs.shape=10*64*16
        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t, train_diff_steps=200
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)  # t为10个int，即batch_size个T(采样步数)
                t_mask = (t >= 0)   # 10个true
                input_embs_rand = self.diffusion.q_sample(input_embs, t)    # input_embs_rand.shape=10*64*16,得到xt
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps  # t.shape=torch.size([10])
                # self.time_embeddings(t).shape=10*768
                time_emb = self.time_embeddings(t).unsqueeze(1)     # time_emb.shape=10*1*768
        if self.diffusion is None and t is not None:
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)
        input_embs = self.up_proj(input_embs)   # input_embs=10*64*768  上采样
        input_img_embs = img.float()
        if t_aware: # true
            input_embs = torch.cat([time_emb, input_embs ], dim=1)   # input_embs=10*65*768
            #input_embs = torch.cat([input_img_embs.expand([-1,32,768]),input_embs],dim=1)
        transformer_outputs = self.transformer(
            past_key_values=past_key_values,    # none
            attention_mask=attention_mask,  # none
            token_type_ids=token_type_ids,  # none
            position_ids=position_ids,  # none
            head_mask=head_mask,    # none
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,    # none
            encoder_attention_mask=encoder_attention_mask,  # none
            use_cache=use_cache,    # none
            output_attentions=output_attentions,    # None
            output_hidden_states=output_hidden_states,  # none
            return_dict=return_dict,    # true
        )
        #transformer_outputs[0].shape=10*65*768
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ] # hidden_states.shape=10*65*768
         
        else:
            hidden_states = transformer_outputs[0]
        hidden_states = self.down_proj(hidden_states.permute(0,2,1))
        hidden_states = self.down_proj2(hidden_states.permute(0,2,1))
        # Set device for model parallelism
        if self.model_parallel: # false
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        #lm_logits = self.down_proj(hidden_states)    # lm_logits.shape=10*64*21
        #hidden_states = self.mediu_pool(hidden_states)
        lm_logits = torch.cat([input_img_embs,hidden_states],dim=1)
        #lm_logits = self.down_proj(lm_logits)
        #print("lm_logits",lm_logits)
        logits_emb = self.lm_head2(lm_logits.view(lm_logits.size(0),-1))
        shift_labels = labels.long()
        #print(shift_labels)
        #loss_fct = torch.nn.BCEWithLogitsLoss()
        #pdb.set_trace()
        loss_fct = CrossEntropyLoss()
        #print("logits_emb",logits_emb)
        print(logits_emb)
        loss = loss_fct(logits_emb, shift_labels)    # shift_logits.size(-1)=21
        print("****loss:",loss)
        #if not return_dict: # return_dict=ture

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits_emb,
            past_key_values=transformer_outputs.past_key_values,    # none
            hidden_states=transformer_outputs.hidden_states,    # none
            attentions=transformer_outputs.attentions,  # none
            cross_attentions=transformer_outputs.cross_attentions,  # none
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
