import os
import json
from BenchRunner.template import FewShotTemplate
from BenchRunner.utils import *

full_tasks = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
              'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
              'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
              'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting',
              'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names',
              'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences',
              'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
              'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']


class BBHRunner:
    def __init__(self, model, data_base_dir, prompt_base_dir, log_base_dir, tasks=None):
        self.model = model
        self.data_base_dir = data_base_dir
        self.prompt_base_dir = prompt_base_dir

        if tasks is None:
            tasks = full_tasks
        self.tasks = tasks

        self.log_base_dir = log_base_dir
        if not os.path.exists(self.log_base_dir):
            os.makedirs(self.log_base_dir)
        self.logs = []

    def load_data(self, task):
        """Load input questions and prompts from corresponding directories"""
        template = FewShotTemplate(task, self.prompt_base_dir)
        with open(os.path.join(self.data_base_dir, f'{task}.json'), encoding="utf8") as file:
            questions = json.load(file)['examples']

        return template, questions

    def log_output(self, prompt, prediction, answer, target):
        self.logs.append(
            {
                'input': prompt,
                'prediction': prediction,
                'answer': answer,
                'target': target
            }
        )

    def flush(self, task, accuracy=None):
        file_path = os.path.join(self.log_base_dir, f'{task}.json')
        data = {'outputs': self.logs, 'accuracy': accuracy}

        # Write JSON data to file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

        self.logs = []

    def run_bbh_test(self):
        """Test the function of bbh runner or models, running part of the bench"""
        total_score, total_missing, total_count = 0, 0, 0
        for task in self.tasks:
            score, missing, count = 0, 0, 0
            template, questions = self.load_data(task)

            for i, question in enumerate(questions[0: 1]):
                count += 1
                prompt = template.get_prompt(question['input'])
                output = self.model(prompt)

                print('-' * 25, f'{task} Question {i}', '-' * 25)
                print(question['input'])
                print('*' * 15, 'Model Output', '*' * 15)
                print(output)
                print('*' * 15, 'Correct Answer', '*' * 15)
                print(f'Correct answer "{question["target"]}"')

                answer = extract_answer(output)
                if answer is None:
                    print('No answer found')
                    missing += 1
                elif answer == question['target']:
                    print(f'Extracted answer "{answer}": Hit.')
                    score += 1
                else:
                    print(f'Extracted answer "{answer}": Miss.')

                # Log the model output
                self.log_output(prompt, output, answer, question['target'])

            self.flush(task, accuracy=score/count)
            total_score += score
            total_missing += missing
            total_count += count

        print(f'Total score: {total_score}/{total_count}')
        print(f'Missing outputs: {total_missing}/{total_count}')

    def run_bbh(self):
        """Run the whole bbh bench"""
        for task in self.tasks:
            template, questions = self.load_data(task)

            for i, question in enumerate(questions):
                prompt = template.get_prompt(question['input'])
                answer = self.model(prompt)

                print('-' * 25, f'{task} Question {i}', '-' * 25)
                print(question['input'])
                print('*' * 15, 'Model Output', '*' * 15)
                print(answer)
                print('*' * 15, 'Correct Answer', '*' * 15)
                print(question['target'])



