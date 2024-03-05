import re
import sys
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append('../..')

from src.utils.model_utils import load_model, load_gpt2_model, load_opt_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_direction(input_text, model, tokenizer, model_name="flan-t5", max_new_tokens=1000):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(device)

    if re.search(r"opt", args.model_name):
        if inputs.input_ids.size(1) > model.config.max_position_embeddings:
            inputs.input_ids = inputs.input_ids[:, :model.config.max_position_embeddings]
        outputs = model.generate(**inputs, max_length=min(max_new_tokens, model.config.max_position_embeddings))
    elif re.search(r"gpt", args.model_name):
        if inputs.input_ids.size(1) > model.config.max_position_embeddings:
            inputs.input_ids = inputs.input_ids[:, :model.config.max_position_embeddings]
        outputs = model.generate(**inputs, max_length=min(max_new_tokens, model.config.max_position_embeddings),  
        do_sample=True, 
        temperature=0.8)
    elif re.search(r"flan", args.model_name):
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def main(args):
    if re.search(r"flan", args.model_name):
        model, tokenizer = load_model(args.model_path)
    elif re.search(r"gpt", args.model_name):
        model, tokenizer = load_gpt2_model(args.model_path)
    elif re.search(r"opt", args.model_name):
        model, tokenizer = load_opt_model(args.model_path)

    if not os.path.exists(args.prediction_output_path):
        os.makedirs(args.prediction_output_path)

    test_df = pd.read_csv(args.data_path)
    predictions = []
    for question in tqdm(test_df['question']):
        output_text = generate_direction(question, model, tokenizer, args.model_name)
        predictions.append(output_text)

    # Save the results for easy loading
    test_df['predictions'] = predictions
    # Saved the prediction results under the model_path/evaluations folder.
    test_df.to_csv(os.path.join(args.prediction_output_path,
                                f"test_predictions_{args.model_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Predictions')
    parser.add_argument('--model_path', type=str, default="/shared/3/projects/spatial-understanding/checkpoints_1109/",
                        help='Path to the saved model checkpoint')
    parser.add_argument('--model_name', type=str,
                        default="flan-t5", help='Name the saved model')
    parser.add_argument('--data_path', type=str,
                        default="/shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl_full/test_spatial_understanding.csv",
                        help='Path to the test dataset')
    parser.add_argument('--prediction_output_path', type=str,
                        default="/shared/3/projects/spatial-understanding/checkpoints_1109/evaluations",
                        help='Path to the predictions CSV file')
    args = parser.parse_args()
    main(args)
