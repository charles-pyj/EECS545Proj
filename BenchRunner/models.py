import torch
import torch.nn as nn
from transformers import OPTForCausalLM, GPT2Tokenizer
from openai import OpenAI

# You can replace with your api key
API_KEY = "api key"


class OPTModel(nn.Module):
    def __init__(self, model_version):
        super().__init__()

        self.model = OPTForCausalLM.from_pretrained(model_version)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)

    def __call__(self, prompt, max_length=2048):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device=self.model.device)
        input_len = input_ids.shape[1]  # Length of the inputs
        max_length = min(max_length, int(1.4 * input_len))  # Define the max output length

        model_output = self.model.generate(input_ids,
                                           max_length=max_length,
                                           num_return_sequences=1,
                                           do_sample=True)

        output_ids = model_output[0, input_len:]  # Remove the inputs
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Enforce a stop sign to truncate the output
        answers = output.split('\n\n', 1)
        return answers[0] if len(answers) > 1 else output


class ChatGPTModel:
    def __init__(self, model_version):
        self.client = OpenAI(api_key=API_KEY)
        self.model = model_version

    def __call__(self, prompt, max_tokens=1000):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )

        output = response.choices[0].message.content
        return output
