# efficient-detr-pytorch
An improved DETR model for MCR defect detection, featuring an EfficientNet-B2 backbone with SimAM attention and optimized loss functions (Smooth L1 + DIoU).
# Improved DETR for MCR Defect Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-v1.5+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

This repository contains the PyTorch implementation of an improved **DETR** (**DE**tection **TR**ansformer) model, specifically optimized for **MCR (Magnetic Controlled Reactor) defect detection**. 

While the original DETR model achieves strong results on standard datasets, it faces challenges with small-sized targets, imbalanced classes, and limited sample volumes in industrial defect detection. To overcome these limitations, we propose an enhanced DETR architecture with a lightweight backbone, advanced attention mechanisms, and optimized loss functions.
                                                                  <img width="807" height="269" alt="image" src="https://github.com/user-attachments/assets/d7cfddd7-4447-480f-93b0-f832ca292066" />

## 🚀 Key Improvements

1. **Lightweight & Efficient Backbone**: 
   We replaced the standard ResNet-50 backbone with **EfficientNet-B2**. This significantly improves the model's capacity to extract features while maintaining a lightweight architecture.
   
2. **SimAM 3D Attention Mechanism**: 
   To better capture subtle object features, we replaced the original Squeeze-and-Excitation (SE) attention mechanism within the EfficientNet's MBConv modules with **SimAM** (a Simple, Parameter-Free Attention Module). SimAM evaluates the 3D weights of features, enhancing the model's sensitivity to defect details without adding extra parameters.
                               <img width="857" height="203" alt="image" src="https://github.com/user-attachments/assets/cc209691-b28c-453a-9379-b2f4eefc792f" />

3. **Optimized Loss Functions**: 
   We upgraded the bounding box regression loss to improve localization precision and convergence speed. The standard L1 Loss and GIoU Loss have been replaced with a linear combination of **Smooth L1 Loss** and **DIoU (Distance-IoU) Loss**. DIoU directly minimizes the distance between the center points of the predicted and ground-truth boxes, leading to faster convergence and better handling of overlapping defects.

## 🧠 Model Architecture

The improved model follows an end-to-end Transformer-based architecture:
1. **Backbone**: EfficientNet-B2 with integrated SimAM modules extracts multi-scale feature maps from the input image.
2. **Transformer**: A standard Transformer encoder-decoder architecture processes the flattened features alongside positional encodings to globally reason about object relations.
3. **Prediction Head**: A Feed-Forward Network (FFN) outputs class predictions and bounding boxes supervised by our optimized **Smooth L1 + DIoU** loss.

## 🛠️ Installation

There are no extra compiled components in this modified DETR, and package dependencies are minimal.

First, clone the repository locally:
```bash
git clone https://github.com/xiao-student12/efficient-detr-pytorch.git
Then, install PyTorch 1.5+ and torchvision 0.6+:

Bash
conda install -c pytorch pytorch torchvision
Install additional dependencies (for training and evaluation):

Bash
conda install cython scipy
pip install -U 'git+[https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI](https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI)'
📊 Data Preparation
We expect the dataset directory structure to follow the standard COCO format:

Plaintext
path/to/dataset/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
🏃 Training
To train the improved DETR model on a single node with 8 GPUs, run:

Bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/dataset 
(Note: You can adjust the batch size and learning rate inside main.py according to your hardware constraints and MCR dataset specifics.)

📈 Evaluation
To evaluate the trained model on your validation set with a single GPU, run:

Bash
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/your/checkpoint.pth --coco_path /path/to/dataset
🙏 Acknowledgements
This project is built upon the official DETR repository by Facebook Research. We thank the original authors for their groundbreaking work on end-to-end object detection with Transformers.
## 📑 Citation

This repository contains the official code for our submission to ***The Visual Computer***. If you find our work, including the improved EfficientNet-B2 backbone, SimAM attention integration, or the optimized loss functions useful for your research, please consider citing our paper:

```bibtex
@article{58d969e4-5304-43d9-b6d2-9819011a0e3f,
  title={ Heterogeneous Surface Defect Detection via Dual-Branch Transformer and Connected Component Analysis for Metallized Ceramic Rings},
  author={YiSong Xiao , Xian Wang ,  YunLong Liu , TianLong Yang , LongTao Ma },
  journal={},
  year={},
  publisher={}
}
📄 License
This project is released under the Apache 2.0 license.
