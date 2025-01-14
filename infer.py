import os
import json
from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer

from arch.config import Config
from arch.model import NanoFormerForCausalLM


def main(model_path):
    if not os.path.isdir(model_path):
        raise OSError(f"Path {model_path} does not exist")
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.isfile(config_path):
        raise OSError(f'Config file does not exist in {model_path}')

    with open(config_path, 'r') as f:
        config = Config(**json.load(f))
        config.gradient_checkpointing = False

    model = NanoFormerForCausalLM(config)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), weights_only=True),strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("imdatta0/nanoformer")
    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    print('Model loaded successfully')

    with torch.no_grad():
        while True:
            text = input('Enter text: ')
            if text == "":
                print(f'empty text detected, quitting')
                break
            for i in range(10):
                tokens = tokenizer([text], return_tensors='pt')
                output = model(tokens['input_ids'], tokens['attention_mask'])
                print(f'output of the model {output} shape {output[0].shape}')
                out_text = tokenizer.decode(output[0][-1].argmax(dim=-1))
                print(out_text[len(text)], end='')
                text = out_text


    return

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_path",type=str, required=True, help="Enter the path where you stored the model weights and config")
    
    args = parser.parse_args()

    main(args.model_path)
