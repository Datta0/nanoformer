# NanoFormer

NanoFormer is a lightweight transformer model implementation designed for efficient training and inference. It is a collection of transformer architectures (and a few variants) that can be easily experimented with

## Features

- Supports transformers variants like:
    * Multi Head Attention (MHA)
    * Grouped Query Attention (GQA)
    * [Differential Transformer](https://arxiv.org/abs/2410.05258)
    * [normalised GPT (nGPT)](https://arxiv.org/abs/2410.01131)
- Dynamic batch size handling with efficient padding
- Mixed precision training (bfloat16)
- Gradient checkpointing for memory efficiency
- Gradient accumulation support
- Wandb integration for experiment tracking
- Automatic model checkpointing
- Custom training loop with validation
- Torch compile support 

## Installation

``` bash
git clone https://github.com/yourusername/nanoformer.git
cd nanoformer
```

## Usage

### Training

To train the model with default parameters:

``` bash
python train.py \
    --dataset "imdatta0/wikipedia_en_sample" \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --lr 5e-4 \
    --hidden_dim 256 \
    --num_hidden_layers 8 \
    --attention_type="gqa" \ # or [one of "diff", "ngpt"]
    --compile
    # --logit_cap = None 
    # --logit_scale = 1.0
```

To estimate the number of tokens in a dataset and the model's param count with given config:
(will need to refactor this to not create the model for estimation)

```bash
python train.py \
    --dataset "imdatta0/wikipedia_en_sample" \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --lr 5e-4 \
    --hidden_dim 256 \
    --num_hidden_layers 8 \
    --estimate
```

**Note:** If you use --compiple, it might take a while to get the initial batch started. So the tqdm ETA estimates might be off. The GPU utilisation will pick up after a few batches and training speeds up. So please be patient.

## TODO
- [x] Implement [Differential Transformer](https://datta0.substack.com/i/150138108/differential-transformer)
- [x] Implement [nGPT](https://arxiv.org/abs/2410.01131)
- [ ] Implement custom optimisers like [Shampoo](https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py), [SOAP](https://arxiv.org/abs/2409.11321) and whatnot
- [ ] Add support for Sliding Window Attention
- [x] Modify configs to be closer to Chinchilla Optimal Ratios

# WIP
- [ ] Inference support
- [ ] Loading from checkpoint