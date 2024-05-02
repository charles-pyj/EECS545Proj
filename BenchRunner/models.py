import torch
import torch.nn as nn
from transformers import OPTForCausalLM, GPT2Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from typing import Literal, List
from openai import OpenAI
from huggingface_hub import login

# You can replace with your api key
API_KEY = "your-api-key"


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

class T5Model(nn.Module):
    def __init__(self, model_version):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(model_version)
    
    def __call__(self, prompt, max_tokens=500):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device=self.model.device)
        model_max_len = self.model.config.max_position_embeddings
        chunk_size = 100  # Tokens to generate each iteration
        outputs = self.model.generate(**inputs, 
                                      max_new_tokens=max_tokens,
                                      do_sample=False)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

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

class GemmaInstructModel:
    def __init__(self, model_version: Literal["google/gemma-2b-it", "google/gemma-7b-it"], device: torch.device='cpu') -> None:
        # login to hf
        login('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')
        
        self.model = AutoModelForCausalLM.from_pretrained(model_version).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_version)
        self.chat = []
    
    def __call__(self, prompt: str, max_tokens: int=1000) -> List[str]:
        self.chat.append({
            "role": "user",
            "content": prompt
        })
        templated_prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False, add_generation_prompt=True)
        print(templated_prompt)
        inputs = self.tokenizer.encode(templated_prompt, add_special_tokens=False, return_tensors='pt')

        with torch.inference_mode():
            outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=max_tokens)
       
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return decoded_output

class GemmaModel:
    def __init__(self, model_version: Literal["google/gemma-2b", "google/gemma-7b"], device: torch.device='cpu') -> None:
        # login to hf
        login('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')
        
        self.model = AutoModelForCausalLM.from_pretrained(model_version).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_version)

    def __call__(self, prompt: str, max_tokens: int=1000) -> torch.Any:
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        
        with torch.inference_mode():
            # need to tune this, generate things badly now.
            outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=max_tokens)
        
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return decoded_output