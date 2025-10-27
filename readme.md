# SimCLR: Self-Supervised Contrastive Learning on CIFAR-10

This repository implements a **from scratch SimCLR framework** in PyTorch for **self-supervised visual representation learning**.  
It is built to be clear, modular, and educational — showing exactly how SimCLR learns meaningful visual features *without labels*.

---

## Concept Overview

**SimCLR (Chen et al., 2020)** learns image representations by comparing *different augmented views* of the same image.  
Instead of predicting labels, it learns to make similar images *closer* in embedding space and different images *further apart*.

### Key Idea
1. Take one image from the dataset.
2. Generate **two random augmentations** (“views”) using color jitter, cropping, rotation, and Gaussian noise.
3. Encode both views with the **same ResNet encoder**.
4. Use a **contrastive loss (NT-Xent)** to pull the two embeddings together, while pushing all other images in the batch apart.

## Project Structure
simclr/
├── main.py                 # Main self-supervised training loop
├── model.py                # ResNet encoder + projection head
├── loss_nt_xent.py         # NT-Xent contrastive loss
├── data_aug.py             # SimCLR-style data augmentations
├── config.py               # Training configuration
├── train_encoder_eval.py   # Example: freeze encoder and train classifier
└── README.md               # This file


##  Implementation Details

###  Model (`model.py`)
- **Base encoder:** ResNet-18 (from torchvision)
- **Projection head:** 2-layer MLP  
  `Linear -> ReLU -> Linear `
- The encoder output before projection is used as the **representation**.

---

### Loss Function (`loss_nt_xent.py`)
Implements the **Normalized Temperature scaled Cross Entropy (NT-Xent)** loss

\[
L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i, z_k)/\tau)}
\]

- `sim(a,b)` = cosine similarity  
- `τ` = temperature parameter (default 0.5)

This encourages positive pairs to have higher similarity than negatives.

### Data Augmentation (`data_aug.py`)
Following the original paper, two distinct augmentations are applied per image.

**View 1**
- Random resized crop (scale 0.9–1.0)
- Small random rotation (±10°)

**View 2**
- Color jitter (brightness/contrast)
- Gaussian noise injection

Each image returns `[view1(x), view2(x)]`, used for contrastive training.

### Training Pipeline (`main.py`)
- Dataset: **CIFAR-10**
- Device: **CUDA / MPS / CPU auto-detection**
- Optimizer: **Adam**
- Scheduler: **ReduceLROnPlateau**
- Optional subsampling for quick experiments (`config["size_dataset_train"]`)
- Model saved after pretraining as `simclr_encoder.pth`


## 🧠 Encoder Extraction

After self-supervised pretraining, extract and freeze the encoder:

```python
encoder = nn.Sequential(*list(model.resnet.children())[:-1])
torch.save(encoder.state_dict(), "simclr_encoder.pth")
print("Encoder saved successfully!")
```

Then reuse it in a downstream classifier:

```python
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    encoder,
    nn.Flatten(),
    nn.Linear(512, 10)  # CIFAR-10 → 10 classes
)
```

Train only the classifier to measure representation quality.

---

##  Performance Tips

| Issue | Cause | Solution |
|-------|--------|-----------|
| Low GPU usage (<50%) | CPU data loading bottleneck | Increase `num_workers`, enable `pin_memory` |
| High CPU load | On-the-fly augmentations | Pre-cache or simplify transforms |
| Slow convergence | Small batch size | Increase to 128–256 if GPU allows |
| Overfitting | Small dataset subset | Increase `size_dataset_train` or add stronger augmentations |

Example optimized DataLoader:
```python
dataloader = DataLoader(
    small_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```


##  Dataset

- **CIFAR-10** (automatically downloaded via `torchvision.datasets.CIFAR10`)
- Image size: 32×32
- Total: 50,000 training images, 10,000 test images  
- 10 classes (used only for evaluation)

## Future improvments 

- Add **SimCLR v2** (deeper projection head + momentum encoder)
- Integrate **MAE** or **BYOL** for comparison
- Add **linear evaluation protocol**

## Citation

If you use or reference this work:
```bibtex
@article{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

---

## Summary

This implementation demonstrates:
- Full **SimCLR pipeline** in PyTorch (from augmentation → encoder → loss)
- Practical **training and optimization** workflow for self-supervised learning
- Modular, reusable design suitable for research or CV showcase

**Goal:** Learn rich, transferable visual embeddings from unlabeled data.


## Results

The SimCLR model was trained on **CIFAR-10** for **200 epochs** using **ResNet-18** as the encoder and a small neural network for linear classification.  

### Linear Evaluation Accuracy

| Epoch | Loss  | Accuracy |
|-------|-------|----------|
| 200   | 1.1386 | 58.65% |

### Observations

- Considering the **limited computational resources**, training for 200 epochs on ResNet-18 gives a **reasonable performance**.  
- ResNet-18 is a relatively small model for CIFAR-10, so achieving around **60% accuracy** is **good** for this setup.  
- With more computational power, a larger encoder (like ResNet-50) or longer training could significantly improve performance.  
- Despite the limitations, the model learned meaningful representations that can be further refined with hyperparameter tuning and a more powerful backbone.
