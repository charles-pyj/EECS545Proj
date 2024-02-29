import torch
import torch.nn as nn
from transformers import OPTForCausalLM, GPT2Tokenizer
from openai import OpenAI

# You can replace with your api key
API_KEY = "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"


class OPTModel(nn.Module):
    def __init__(self, model_version):
        super().__init__()

        self.model = OPTForCausalLM.from_pretrained(model_version)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)

    def __call__(self, prompt, max_tokens=500):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device=self.model.device)
        model_max_len = self.model.config.max_position_embeddings
        chunk_size = 100  # Tokens to generate each iteration

        generated_text = ""
        tokens_generated = 0
        while tokens_generated < max_tokens:
            if input_ids.shape[1] + chunk_size > model_max_len:
                input_ids = input_ids[:, -(model_max_len - chunk_size):]

            output = self.model.generate(input_ids,
                                         max_length=input_ids.shape[1] + chunk_size,
                                         num_return_sequences=1,
                                         do_sample=False,
                                         repetition_penalty=1.3)
            generated_text += self.tokenizer.decode(output[0, -100:], skip_special_tokens=True)
            input_ids = output
            tokens_generated += chunk_size

        # Enforce a stop sign to truncate the output
        answers = generated_text.split('\n\n', 1)
        return answers[0] if len(answers) > 1 else generated_text


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
