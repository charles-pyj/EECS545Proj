import os
from BenchRunner.utils import *


class FewShotTemplate:
    def __init__(self, task_name, prompt_dir, mode):
        """Load and parse prefixes and fewshot examples"""
        example_path = os.path.join(prompt_dir, f'{task_name}.txt')
        with open(example_path, encoding="utf8") as file:
            content = file.read()

        content = content.split('-----\n')[1]
        content = content.split("\n\n")

        # The input prefix of Few shot prompting
        self.prefix = '\n\n'.join(content[: -3])

        # The examples in the form of a list of dicts
        self.examples = []
        self.mode = mode
        for example in content[-3:]:
            question, answer = example.split('\nA: ', 1)

            question = question.replace('Q: ', '')
            if mode == 'cot':
                answer = answer.replace('A: ', '')
            elif mode == 'direct':
                answer = extract_answer(answer)
            else:
                raise NotImplementedError()
            self.examples.append({'question': question, 'answer': answer})

    def get_examples(self):
        return self.examples

    @staticmethod
    def get_example_template():
        template = 'Q: {question}\nA: {answer}'

        return template

    def get_prefix(self):
        return self.prefix

    def get_suffix(self):
        if self.mode == 'cot':
            return "Q: {question}\nA: Let's think step by step."
        elif self.mode == 'direct':
            return "Q: {question}\nA: "
        else:
            raise NotImplementedError()

    def get_prompt(self, text):
        prompt_template = self.prefix + '\n' * 2
        for example in self.examples:
            prompt_template += self.get_example_template().format(**example) + '\n' * 2

        prompt_template += self.get_suffix().format(question=text)

        return prompt_template
