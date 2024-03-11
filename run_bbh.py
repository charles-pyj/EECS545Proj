from BenchRunner.bbhrunner import BBHRunner
from BenchRunner.models import OPTModel, ChatGPTModel

# Use "model = OPTModel('facebook/opt-1.3b').to('cuda')" to call facebook opt models
# Fine-tuning is required for fewshot performance
model = ChatGPTModel('gpt-3.5-turbo')

# Test the functionality
bbh = BBHRunner(model,
                './bbh',
                './cot-prompts',
                './gpt-3.5-turbo-partial-outputs',
                )
bbh.run_bbh_test()

# An example of using direct answer prompt.
bbh = BBHRunner(model,
                './bbh',
                './cot-prompts',
                './gpt-3.5-turbo-partial-outputs',
                mode='direct')
bbh.run_bbh_test()
