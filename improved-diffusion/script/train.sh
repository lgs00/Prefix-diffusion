export CUDA_VISIBLE_DEVICES=1
OPENAI_LOGDIR=diffusion_model/coco/1230 \
TOKENIZERS_PARALLELISM=false \
python scripts/train.py  \
--checkpoint_path diffusion_model/coco/1230 \
--model_arch transformer \
--modality e2e-tgt \
--save_interval 100000 \
--lr 0.0001 \
--batch_size 128 \
--diffusion_steps 1000 \
--noise_schedule trunc_cos \
--use_kl False \
--learn_sigma False \
--image_size 8 \
--num_channels 128 \
--seed 3407 \
--dropout 0.1 \
--in_channel 48 \
--out_channel 48 \
--padding_mode pad \
--experiment random \
--lr_anneal_steps 200000 \
--weight_decay 0.01 \
--num_res_blocks 2 \
--predict_xstart True \
--training_mode e2e \
--vocab_size 14145 \
--e2e_train ../datasets/coco_vit_L14

