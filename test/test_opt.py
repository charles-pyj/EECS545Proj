"""A script to test the funtionality of the bbh"""

from BenchRunner.bbhrunner import BBHRunner
from BenchRunner.models import ChatGPTModel, OPTModel

# Use "model = OPTModel('facebook/opt-1.3b').to('cuda')" to call facebook opt models
# Fine-tuning is required for fewshot performance
model = OPTModel('facebook/opt-1.3b').to('cuda')

bbh = BBHRunner(model,
                './bbh',
                './cot-prompts',
                './opt-1.3b-partial-outputs')
bbh.run_bbh_test()
