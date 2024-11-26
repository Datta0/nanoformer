from math import e
import torch
from datasets import load_dataset
from argparse import ArgumentParser
from arch.config import Config
from arch.model import NanoFormerForCausalLM
from transformers import Trainer, TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import wandb
from tqdm.auto import tqdm
import os
import shutil

def get_param_count(model):
    total_count, grad_count = 0, 0
    unique_params = set()  # Track unique parameters by their `id`
    
    for _, param in model.named_parameters():
        # Check if this parameter is already counted
        if id(param) not in unique_params:
            unique_params.add(id(param))  # Mark this parameter as counted
            total_count += param.numel()
            if param.requires_grad:
                grad_count += param.numel()
                
    return total_count, grad_count


def tokenize_function(examples, tokenizer, max_length):
    # Tokenize without padding first
    tokenized = tokenizer(
        examples["text"], 
        max_length=max_length, 
        truncation=True,
        padding=False,  # Don't pad yet
        return_tensors=None  # Return lists instead of tensors
    )
    return tokenized  # Return without padding

def create_dataloaders(dataset, tokenizer, batch_size, max_length, num_workers=4):
    tokenize = partial(tokenize_function, tokenizer=tokenizer, max_length=max_length)
    
    train_dataset = dataset["train"].map(
        tokenize,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["test"].map(
        tokenize,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset["test"].column_names
    )
    
    def collate_fn(batch):
        # Find max length in batch
        max_len = max(len(x['input_ids']) for x in batch)
        # Round up to nearest multiple of 8
        max_len = ((max_len + 7) // 8) * 8
        max_len = min(max_len, max_length)
        
        # Pad each sequence to max_len
        padded_batch = tokenizer.pad(
            {'input_ids': [x['input_ids'] for x in batch],
             'attention_mask': [x['attention_mask'] for x in batch]},
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        return padded_batch
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader

def custom_training_loop(
    model,
    train_dataloader,
    val_dataloader,
    args,
    device="cuda:0",
):
    # Initialize wandb
    if args.run_name is None:
        run_name = f'{args.attention_type}_ep{args.num_epochs}_bs{args.batch_size}x{args.gradient_accumulation_steps}_lr{args.lr}_norm{args.max_grad_norm}'
    else:
        run_name = args.run_name
    output_dir = f'/home/datta0/models/nanoformer/{run_name}'
    
    if not args.no_wandb:
        wandb.init(
            project="nanoformer",
            name=run_name,
            config=vars(args)
        )
    
    # Track best validation loss for saving best model
    best_val_loss = float('inf')
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * args.num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
    )

    if args.compile:
        model = torch.compile(model)

    # Training loop
    model.train()
    model.set_train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with bfloat16 autocast
            with torch.amp.autocast("cuda:0", enabled=True, dtype=torch.bfloat16):
                outputs, loss = model(**batch)
                # Divide loss by gradient accumulation steps
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps  # Multiply back to get true loss
            
            # Only update weights after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if args.attention_type=="ngpt":
                    model.normalize_weights()
                
                # Logging
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                if not args.no_wandb:
                    wandb.log({
                        "train/loss": total_loss / args.gradient_accumulation_steps,  # Average loss over accumulation steps
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/gradient_norm": grad_norm.item(),
                        "train/epoch": epoch + (step / len(train_dataloader)),
                        "train/global_step": global_step,
                    }, step=global_step)
                    
                print(f"Epoch {epoch+1}/{args.num_epochs} | "
                      f"Step {step+1}/{len(train_dataloader)} | "
                      f"Loss: {total_loss / args.gradient_accumulation_steps:.4f}")
                total_loss = 0
                global_step += 1

                # if args.tie_word_embeddings and not torch.allclose(model.lm_head.weight, model.model.embed_tokens.weight):
                #     print(f'Unequal tied embeddings at step {(epoch,step)}')
                
                progress_bar.set_postfix(loss=f"{total_loss / args.gradient_accumulation_steps:.4f}")
        
            # Save checkpoint every save_steps
            if global_step % args.save_steps == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                model.save_pretrained(checkpoint_dir)
                
                # Keep only last 3 checkpoints
                checkpoints = sorted([
                    d for d in os.listdir(output_dir) 
                    if d.startswith('checkpoint-')
                ], key=lambda x: int(x.split('-')[1]))
                
                if len(checkpoints) > 3:
                    shutil.rmtree(f"{output_dir}/{checkpoints[0]}")
            
            
        # Validation loop
        val_progress = tqdm(val_dataloader, desc=f"Validation epoch {epoch+1}")
        val_loss = 0
        with torch.no_grad():
            for batch in val_progress:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = f"{output_dir}/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
        
        if not args.no_wandb:
            # Log validation metrics
            wandb.log({
                "val/loss": val_loss,
                "val/epoch": epoch + 1,
            }, step=global_step)
        
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
        model.train()
    
    # Close wandb run
    wandb.finish()

def count_tokens_in_dataset(dataset, tokenizer, max_len=4096, batch_size=1000, num_proc=16):
    # Tokenize the dataset in batches and parallelize the process
    dataset = dataset.map(
        lambda x: {
            "len": [
                tokenizer(txt, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')['input_ids'].shape[1]
                for txt in x['text']
            ]
        },
        batched=True, batch_size=batch_size, num_proc=num_proc
    )
    if 'train' in dataset:
        lens = dataset['train']["len"]
    else:
        lens = dataset["len"]
    total_tokens = np.sum(lens)
    avg_tokens = total_tokens / len(lens)
    max_tokens = np.max(lens)
    min_tokens = np.min(lens)
    
    return total_tokens, avg_tokens, max_tokens, min_tokens, lens

def main(args):
    dataset = load_dataset(args.dataset)
    train_data, val_data = dataset["train"], dataset["test"]
    print(train_data, val_data)

    tokenizer = AutoTokenizer.from_pretrained("imdatta0/nanoformer")
    tokenizer.pad_token = tokenizer.eos_token

    args.vocab_size = tokenizer.vocab_size
    config = Config(**vars(args))
    config.vocab_size = tokenizer.vocab_size
    print(f'Setting vocab size to {tokenizer.vocab_size} from tokenizer')

    model = NanoFormerForCausalLM(config)
    print(f'model is {model}')
    total_params, trainable_params = get_param_count(model)
    print(f'Total params: {total_params} aka {total_params/1e6:.2f}M, Trainable params: {trainable_params}')

    if args.estimate:
        total_tokens, avg_tokens, max_tokens, min_tokens, lens = count_tokens_in_dataset(dataset, tokenizer)
        print(f'Total tokens: {total_tokens} aka {total_tokens/1e6:.2f}M, Average tokens: {avg_tokens}, Max tokens: {max_tokens}, Min tokens: {min_tokens}')
        return

    model = model.to(torch.bfloat16)
    model.train()
    model.to("cuda:0")

    train_dataloader, val_dataloader = create_dataloaders(
        dataset,
        tokenizer,
        args.batch_size,
        args.max_position_embeddings,
    )
    
    custom_training_loop(
        model,
        train_dataloader,
        val_dataloader,
        args,
    )

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="imdatta0/wikipedia_en_sample")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-3)
    # parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action='store_true')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--estimate", action='store_true')



    # add everything in Config as argument
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--hidden_act", type=str, default="silu")
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)
    parser.add_argument("--input_layernorm", type=bool, default=True)
    parser.add_argument("--post_attention_layernorm", type=bool, default=False)
    parser.add_argument("--pre_ffnn_layernorm", type=bool, default=True)
    parser.add_argument("--post_ffnn_layernorm", type=bool, default=False)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--tie_word_embeddings", action='store_true')
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--rope_scaling", type=dict, default=None)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--attention_scale", type=float, default=0)
    parser.add_argument("--embedding_multiplier", type=float, default=1.0)
    parser.add_argument("--logits_scaling", type=float, default=1.0)
    parser.add_argument("--residual_multiplier", type=float, default=1.0)
    parser.add_argument("--attention_multiplier", type=float, default=1.0)
    parser.add_argument("--attention_cap", type=float, default=None)
    parser.add_argument("--logit_cap", type=float, default=None)
    parser.add_argument("--attention_type", type=str, default="gqa")

    args = parser.parse_args()

    main(args)
