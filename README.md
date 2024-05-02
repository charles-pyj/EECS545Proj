# EECS545Proj
Course Project for EECS545

## Evaluation on BBH

The bench is availabe at [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard). You can run the full evaluation by directly running our provided script after retrieving `bbh` and `cot-prompts` folders.
```python
python run_bbh.py
```
You can replace the following command
```python
model = ChatGPTModel('gpt-3.5-turbo')
```
with you desired model.

We also provide all the existing results in JSON format in `./results` folder. The results are visualized in `parser.ipynb`


## Data augmentation

## Fintune T5 and OPT models

## DPO
