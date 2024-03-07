from BenchRunner.models import GemmaModel, GemmaInstructModel
from BenchRunner.bbhrunner import BBHRunner

model = GemmaInstructModel(model_version='google/gemma-2b-it')

# Test the functionality
bbh = BBHRunner(model,
                './bbh',
                './cot-prompts',
                './gemma-2b-it-partial-outputs')
bbh.run_bbh_test()