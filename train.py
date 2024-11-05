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


def main(args):

    dataset = load_dataset(args.dataset)
    train_data, val_data = dataset["train"], dataset["test"]
    print(train_data, val_data)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    args.vocab_size = tokenizer.vocab_size
    config = Config(**vars(args))
    config.vocab_size = tokenizer.vocab_size
    print(f'Setting vocab size to {tokenizer.vocab_size} from GPT2 tokenizer')

    model = NanoFormerForCausalLM(config)
    model.train()
    model.to("cuda:0")

    run_name = f'{config.attention_type}_ep{args.num_train_epochs}_bs{args.batch_size}_lr{args.lr}_norm{args.max_grad_norm}'
    output_dir = f'/home/datta0/models/nanoformer/'
    logs_dir = f'/home/datta0/logs/nanoformer/'

    training_args = TrainingArguments(
        output_dir=output_dir,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=logs_dir,
        run_name=run_name,
        report_to="wandb",
    )


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

def create_dataloaders(dataset, tokenizer, batch_size, max_length, num_workers=1):
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
        collate_fn=collate_fn  # Add custom collate function
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn  # Add custom collate function
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
    wandb.init(
        project="nanoformer",  # You can customize this
        name=f'{args.attention_type}_ep{args.num_train_epochs}_bs{args.batch_size}_lr{args.lr}_norm{args.max_grad_norm}',
        config=vars(args)
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * args.num_train_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
    )
    
    # Training loop
    model.train()
    model.set_train()
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with gradient checkpointing
            with torch.amp.autocast("cuda:0", enabled=True, dtype=torch.bfloat16):
                outputs, loss = model(**batch)
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item()
            
            # Update weights after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log metrics asynchronously
                wandb.log({
                    "train/loss": total_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (step / len(train_dataloader)),
                    "train/global_step": global_step,
                }, step=global_step)
                
                print(f"Epoch {epoch+1}/{args.num_train_epochs} | "
                      f"Step {step+1}/{len(train_dataloader)} | "
                      f"Loss: {total_loss:.4f}")
                total_loss = 0
                global_step += 1
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_dataloader)
        # Log validation metrics
        wandb.log({
            "val/loss": val_loss,
            "val/epoch": epoch + 1,
        }, step=global_step)
        
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
        model.train()
    
    # Close wandb run
    wandb.finish()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="imdatta0/wikipedia_en_sample")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")


    # add everything in Config as argument
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--hidden_act", type=str, default="silu")
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6)
    parser.add_argument("--input_layernorm", type=bool, default=True)
    parser.add_argument("--post_attention_layernorm", type=bool, default=True)
    parser.add_argument("--pre_ffnn_layernorm", type=bool, default=False)
    parser.add_argument("--post_ffnn_layernorm", type=bool, default=False)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--tie_word_embeddings", type=bool, default=False)
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
