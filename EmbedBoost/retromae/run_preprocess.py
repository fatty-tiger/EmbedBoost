import argparse
import os
import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from EmbedBoost.common import log_util

logger = logging.getLogger(__name__)

def load_datas(dataset_name, input_dir):
    if dataset_name == 'multicpr_ecom':    
        datas = []
        with open(os.path.join(input_dir, 'ecom_corpus.tsv')) as f:
            for line in f:
                splits = line.strip().split('\t')
                if len(splits) != 2:
                    continue
                datas.append({'text': splits[1]})
        
        with open(os.path.join(input_dir, 'train_hard_neg.jsonl')) as f:
            for line in f:
                d = json.loads(line.strip())
                if 'query' not in d or 'positive' not in d:
                    continue
                text = d['query'] + ' ' + d['positive']
                datas.append({'text': text})
        logger.info(f"dataset-{dataset_name} loaded, {len(datas)} documents in total.")
        return datas

    raise ValueError('Unknown dataset name: {}'.format(dataset_name))


def build_dataset(dataset_name: str, input_dir: str, tokenizer_name: str, max_seq_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], 
                         add_special_tokens=False, 
                         truncation=True,
                         max_length=max_seq_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)

    def rename_to_tokenids(examples):
        new_examples = [sent[:] for sent in examples['input_ids']]
        return {'token_ids': new_examples}

    datas = load_datas(dataset_name, input_dir)
    raw_dataset = Dataset.from_list(datas)
    tokenized_dataset = raw_dataset.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_dataset = tokenized_dataset.map(rename_to_tokenids, num_proc=8, batched=True, remove_columns=["input_ids"])
    return processed_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    dataset = build_dataset(args.dataset_name, args.input_dir, args.tokenizer_name, args.max_seq_length)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    log_util.simple_init()
    main()
