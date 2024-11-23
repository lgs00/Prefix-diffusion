# Diffusion-for-image-captioning
## diffusion for image captioning
The continuous diffusion model is applied to the field of image description and generated in a non-autoregressive way to obtain more diverse image descriptions.

### Setup

```bash
cd diffusion
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0
pip install huggingface_hub==0.4.0
pip install wandb
```

### Experiment
train
```bash
cd improved-diffusion
bash script/train_diff.sh
```
test
```bash
bash script/generation.sh
```
Reference data formats are provided below：

链接：https://pan.baidu.com/s/1x2bvPC3oxr0r4OaTz7E2tQ 

提取码：7bdx

## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{liu-etal-2024-prefix,
    title = "Prefix-diffusion: A Lightweight Diffusion Model for Diverse Image Captioning",
    author = "Liu, Guisheng  and
      Li, Yi  and
      Fei, Zhengcong  and
      Fu, Haiyan  and
      Luo, Xiangyang  and
      Guo, Yanqing",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.lrec-main.1134",
    pages = "12954--12965",
}
```
