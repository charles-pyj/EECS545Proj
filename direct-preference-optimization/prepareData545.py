import argparse
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, AutoModelWithLMHead

torch.cuda.set_device(6)

def processGT(row):
    ans = f"A: Let's think step by step.\n{row['rationale']} So the answer is {row['target']}."
    return ans
def main(args):
    file = pd.read_csv("/home/panyijun/EECS545Proj/testT5.csv")
    file['ground_truth'] = file.apply(processGT,axis=1)
    print(len(file))
    print(file.columns)
    file.to_csv("./EECS545SFT_T5.csv",index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Predictions for Spatial Understanding')

    args = parser.parse_args()

    main(args)