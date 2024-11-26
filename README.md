# NanoFormer

NanoFormer is a lightweight transformer model implementation designed for efficient training and inference. It features grouped query attention (GQA) and various architectural optimizations.

## Features

- Configurable transformer architecture with GQA support
- Dynamic batch size handling with efficient padding
- Mixed precision training (bfloat16)
- Gradient checkpointing for memory efficiency
- Gradient accumulation support
- Wandb integration for experiment tracking
- Automatic model checkpointing
- Custom training loop with validation

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
    --num_hidden_layers 8
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

## TODO

- [x] Implement [Differential Transformer](https://datta0.substack.com/i/150138108/differential-transformer)
- [x] Implement [nGPT](https://arxiv.org/abs/2410.01131)
- [ ] Implement custom optimisers like [Shampoo](https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py), [SOAP](https://arxiv.org/abs/2409.11321) and whatnot
- [ ] Add support for Sliding Window Attention
- [ ] Modify configs to be closer to Chinchilla Optimal Ratios