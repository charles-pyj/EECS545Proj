import argparse
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, AutoModelWithLMHead


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

    model = model.to("cuda")
    # Use GPU if available

    return model, tokenizer


def generate_text(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=1000)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def generate_predictions(model_path, test_file, predictions_out):
    model , tokenizer = load_model(model_path)
    model = model.to('cuda')
    test_df = pd.read_csv(test_file)
    predictions = [None] * len(test_df)
    for i, question in enumerate(tqdm(test_df['question'])):
        output_text = generate_text(model, tokenizer, question)
        predictions[i] = output_text

        # Save every 500 questions so you don't lose progress
        if (i + 1) % 500 == 0:
            test_df['predictions'] = predictions
            test_df.to_csv(predictions_out, index=False)

    # Save the final results
    test_df['predictions'] = predictions
    test_df.to_csv(predictions_out, index=False)
    return test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Predictions for Spatial Understanding')
    parser.add_argument('--model_path', type=str, default="/shared/3/projects/spatial-understanding/checkpoints_1109/step_18000",
                        help='Path to the model directory')
    parser.add_argument('--test_file', type=str,
                        default='./testData.csv',
                        help='Path to the test input file')
    parser.add_argument('--predictions_out', type=str, default='./result_temp.csv',
                        help='Path to save the predictions output')

    args = parser.parse_args()
    print(generate_predictions(args.model_path, args.test_file, args.predictions_out))