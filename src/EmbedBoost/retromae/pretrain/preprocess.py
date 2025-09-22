import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def load_datas(input_data_fpath):
    datas = []
    with open(input_data_fpath) as f:
        for line in f:
            datas.append({'text': line.strip()})
    return datas


def build_dataset(input_data_fpath: str, tokenizer_name: str, max_seq_length: int = 256):
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

    datas = load_datas(input_data_fpath)
    raw_dataset = Dataset.from_list(datas)
    tokenized_dataset = raw_dataset.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_dataset = tokenized_dataset.map(rename_to_tokenids, num_proc=8, batched=True, remove_columns=["input_ids"])
    return processed_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_fpath", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    dataset = build_dataset(args.input_data_fpath, args.tokenizer_name, args.max_seq_length)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()