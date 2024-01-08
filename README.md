# Diffusion-for-image-captioning
## diffusion for image captioning
将连续扩散模型应用到图像描述领域，以非自回归的方式进行生成，得到更加多样的图像描述。

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
训练
```bash
cd improved-diffusion
bash script/train_diff.sh
```
测试
```bash
bash script/generation.sh
```

## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@article{liu2023prefix,
  title={Prefix-diffusion: A Lightweight Diffusion Model for Diverse Image Captioning},
  author={Liu, Guisheng and Li, Yi and Fei, Zhengcong and Fu, Haiyan and Luo, Xiangyang and Guo, Yanqing},
  journal={arXiv preprint arXiv:2309.04965},
  year={2023}
}
```
