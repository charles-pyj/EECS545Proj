import argparse
import sys
import torch
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, AutoModelWithLMHead

device = torch.device("cuda:5")

def load_gpt2_model(model_name_or_path):
    print("Loading GPT2 model configuration...")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)

    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(
       "gpt2",
        use_fast=True,
        trust_remote_code=False,
        padding_side = 'left'
    )
    if tokenizer.pad_token is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading GPT2 model...")
    # Assuming 'model_path' is the path to your .pt file
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # Initialize model architecture
    model_state_dict = torch.load(model_name_or_path+"policy.pt")
    model.load_state_dict(model_state_dict['state'])
    model.eval()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing token embeddings to match tokenizer size...")
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    print("GPT-2 model and tokenizer loaded successfully!")
    return model, tokenizer

def load_model(model_name_or_path):
    print("Loading Flan-T5 model configuration...")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)

    print("Loading Flan-T5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        use_fast=True,
        trust_remote_code=False
    )

    print("Loading Flan-T5 model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        trust_remote_code=False,
    )
    model.eval()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing token embeddings to match tokenizer size...")
        model.resize_token_embeddings(len(tokenizer))

    print("Model and tokenizer loaded successfully!")

    # Use GPU if available
    model.to(device)
    return model, tokenizer


def generate_text(model, tokenizer, input_text):
    inputs = tokenizer(input_text, 
                       return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, 
                             max_new_tokens=10000,
                             num_beams=5,
                             num_return_sequences=3,
                             temperature=0.7,
                             do_sample=True)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, decoded_output in enumerate(decoded_outputs):
        print(f"Output {i+1}: {decoded_output}")
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Predictions for Spatial Understanding')
    parser.add_argument('--model_path', type=str, default="/shared/3/projects/spatial-understanding/checkpoints_1109/",
                        help='Path to the model directory')
    parser.add_argument('--test_file', type=str,
                        default='/shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl_full/test_spatial_understanding.csv',
                        help='Path to the test input file')
    parser.add_argument('--predictions_out', type=str, default='../../data/test_predictions.csv',
                        help='Path to save the predictions output')

    args = parser.parse_args()
    inputText = "You are driving from dayalu, praveen, md located at 4260 Plymouth Rd Ann Arbor, MI 48109 to dubin law p located at 2723 S State St Ann Arbor, MI 48104. What directions do you take?"
    model, tokenizer = load_model(args.model_path)
    generate_text(model,tokenizer,inputText)
