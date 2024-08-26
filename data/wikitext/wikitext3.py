from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import tiktoken
from ipdb import set_trace as st


enc = tiktoken.get_encoding("gpt2")

## https://huggingface.co/docs/datasets/en/use_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
print(ds['test']['text'][:4])

tokenizers = AutoTokenizer.from_pretrained("gpt2")
print(tokenizers(ds['test']['text'][:4]))



def tokenization(example):
    return tokenizers(example['text'])

## fastest way to tokenize entire dataset https://huggingface.co/docs/datasets/en/nlp_process
# dataset = ds.map(tokenization, batched=True)
# format for capatible with torch
# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])


x = enc.encode_ordinary(ds['test']['text'])
txt = enc.decode(x)


## evaluate https://huggingface.co/spaces/evaluate-metric/perplexity
perplexity = evaluate.load("perplexity", module_type="metric")

input_texts = ds['test']['text']
## "Each input text must be at least one token long."
input_texts = [s for s in input_texts if s!='']

results = perplexity.compute(model_id='gpt2',
                             predictions=input_texts)
print(list(results.keys()))
print(round(results["mean_perplexity"], 2))
print(round(results["perplexities"][0], 2))
st()