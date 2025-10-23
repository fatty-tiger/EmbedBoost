import argparse
import collections
import sys
import os
import logging
import shutil
import json
import torch
import torch.nn as nn

import sentencepiece as spm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors


LOG_DATE_FMT = '%Y‐%m‐%d %H:%M:%S'
LOG_FMT = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stderr,
                    datefmt=LOG_DATE_FMT,
                    format=LOG_FMT)

logger = logging.getLogger(__name__)


class CustomSpmConverter(SpmConverter):
    def post_processor(self):
        bos_piece = self.proto.trainer_spec.bos_piece
        eos_piece = self.proto.trainer_spec.eos_piece
        return processors.TemplateProcessing(
            single=f"{bos_piece}:0 $A:0 {eos_piece}:0",
            pair=f"{bos_piece}:0 $A:0 {eos_piece}:0 $B:1 {eos_piece}:1",
            special_tokens=[
                (bos_piece, self.original_tokenizer.PieceToId(bos_piece)),
                (eos_piece, self.original_tokenizer.PieceToId(eos_piece)),
            ]
        )


class EmbeddingTransporter(object):
    def __init__(self, args) -> None:
        self.src_model_id_or_path = args.src_model_id_or_path
        self.src_tokenizer = AutoTokenizer.from_pretrained(args.src_model_id_or_path)
        self.src_vocab = self._load_vocab_dict(os.path.join(self.src_model_id_or_path, 'vocab.txt'))
        self.src_config = AutoConfig.from_pretrained(args.src_model_id_or_path)
        self.src_model = AutoModel.from_pretrained(args.src_model_id_or_path)
        self.src_embedding = self.src_model.embeddings.word_embeddings

        self.dest_model_id_or_path = args.output_dir
        self.dest_tokenizer = AutoTokenizer.from_pretrained(self.dest_model_id_or_path)
        self.dest_vocab = self._load_vocab_dict(os.path.join(self.dest_model_id_or_path, 'vocab.txt'))
        self.dest_embedding = nn.Embedding(self.dest_tokenizer.vocab_size, self.src_config.hidden_size, padding_idx=self.src_config.pad_token_id)
        
        self.do_lower_case = args.do_lower_case

    def _load_vocab_dict(self, vocab_fpath):
        vocab_dict = {}
        with open(vocab_fpath) as f:
            for i, line in enumerate(f):
                word = line.strip()
                vocab_dict[word] = i
        return vocab_dict
    
    def _get_backend_model(self, tokenizer: PreTrainedTokenizerBase):
        return tokenizer.backend_tokenizer.model.__class__.__name__
    
    def get_word_mapping(self):
        src_tokenizer = self.src_tokenizer
        src_vocab = self.src_vocab
        dest_tokenizer = self.dest_tokenizer
        dest_vocab = self.dest_vocab

        src_tokenizer_backend = self._get_backend_model(src_tokenizer)
        dest_tokenizer_backend = self._get_backend_model(dest_tokenizer)
        logger.info(f"src_tokenizer_backend: {src_tokenizer_backend}, dest_tokenizer_backend: {dest_tokenizer_backend}")

        word_mapping = {}
        token_mapping = {}
        oov_words = {}
        for k, v in dest_vocab.items():
            # special tokens
            if k == src_tokenizer.unk_token:
                token_mapping[k] = [src_tokenizer.unk_token]
                word_mapping[v] = [src_vocab[src_tokenizer.unk_token]]
                continue
            if k == src_tokenizer.cls_token:
                token_mapping[k] = [src_tokenizer.cls_token]
                word_mapping[v] = [src_vocab[src_tokenizer.cls_token]]
                continue
            if k == src_tokenizer.sep_token:
                token_mapping[k] = [src_tokenizer.sep_token]
                word_mapping[v] = [src_vocab[src_tokenizer.sep_token]]
                continue
            # if k == dest_tokenizer.bos_token:
            #     word_mapping[v] = [src_vocab[src_tokenizer.cls_token]]
            #     continue
            # if k == dest_tokenizer.eos_token:
            #     word_mapping[v] = [src_vocab[src_tokenizer.sep_token]]
            #     continue
            if k == src_tokenizer.pad_token:
                token_mapping[k] = [src_tokenizer.pad_token]
                word_mapping[v] = [src_vocab[src_tokenizer.pad_token]]
                continue
            if k == src_tokenizer.mask_token:
                token_mapping[k] = [src_tokenizer.mask_token]
                word_mapping[v] = [src_vocab[src_tokenizer.mask_token]]
                continue

            # normalize
            if self.do_lower_case:
                k = k.lower()
            
            # 可以在原词表中直接找到的词(对前缀进行处理)
            if k in src_vocab:
                token_mapping[k] = [k]
                word_mapping[v] = [src_vocab[k]]
                continue
            
            # 尝试去除或者添加前缀
            if k.startswith('▁') and k != '▁' and k.lstrip('▁') in src_vocab:
                token_mapping[k] = [k.lstrip('▁')]
                word_mapping[v] = [src_vocab[k.lstrip('▁')]]
                continue
            if k.startswith('##') and k != '##' and k.lstrip('##') in src_vocab:
                token_mapping[k] = [k.lstrip('##')]
                word_mapping[v] = [src_vocab[k.lstrip('##')]]
                continue
            if not k.startswith('▁') and f'▁{k}' in src_vocab:
                token_mapping[k] = [f'▁{k}']
                word_mapping[v] = [src_vocab[f'▁{k}']]
                continue
            if not k.startswith('##') and f'##{k}' in src_vocab:
                token_mapping[k] = [f'##{k}']
                word_mapping[v] = [src_vocab[f'##{k}']]
                continue
            
            k2 = k.lstrip('##').lstrip('▁')
            if not k2:
                continue
                
            # 按src_tokenizer切分后取平均池化（切分前，前缀均去除）
            tokens = src_tokenizer.tokenize(k2)
            if src_tokenizer.unk_token not in tokens:
                ids = src_tokenizer.convert_tokens_to_ids(tokens)
                token_mapping[k] = tokens
                word_mapping[v] = ids[:]
                continue
            
            # 按char_level切分后取平均池化
            tokens = []
            for tk in list(k2):
                if tk in src_vocab:
                    tokens.append(tk)
                    continue
                if tk.startswith('▁') and tk != '▁' and tk.lstrip('▁') in src_vocab:
                    tokens.append(tk.lstrip('▁'))
                    continue
                if tk.startswith('##') and tk != '##' and tk.lstrip('##') in src_vocab:
                    tokens.append(tk.lstrip('##'))
                    continue
                if not tk.startswith('▁') and f'▁{tk}' in src_vocab:
                    tokens.append(f'▁{tk}')
                    continue
                if not tk.startswith('##') and f'##{tk}' in src_vocab:
                    tokens.append(f'##{tk}')
                    continue
                tokens.append(src_tokenizer.unk_token)
            if src_tokenizer.unk_token not in tokens:
                ids = src_tokenizer.convert_tokens_to_ids(tokens)
                token_mapping[k] = tokens
                word_mapping[v] = ids
                continue

            oov_words[k] = v

        return word_mapping, oov_words, token_mapping

    def run(self):
        src_weights = self.src_embedding.weight.data
        nn.init.uniform_(self.dest_embedding.weight, -1.0, 1.0)
        dest_weights = self.dest_embedding.weight.data
        counter = collections.defaultdict(int)
        
        word_mapping, oov_words, token_mapping = self.get_word_mapping()
        for k, vlist in token_mapping.items():
            logger.info(f'word: {k} is mapped to {vlist}')
        for k in oov_words:
            logger.info(f'word: {k} cant be mapped to any tokens')
        
        for i in range(self.dest_tokenizer.vocab_size):
            if i in word_mapping:
                if len(word_mapping[i]) == 1:
                    dest_weights[i, :] = src_weights[word_mapping[i][0], :]
                    counter['identity'] += 1
                else:
                    dest_weights[i, :] = src_weights[word_mapping[i], :].mean(dim=0)
                    counter['mean_pooling'] += 1
        counter['random'] = len(oov_words)
        self.src_model.embeddings.word_embeddings = self.dest_embedding
        torch.save(self.src_model.state_dict(), os.path.join(self.dest_model_id_or_path, 'pytorch_model.bin'))

        src_model_config_fpath = os.path.join(self.src_model_id_or_path, 'config.json')
        dest_model_config_fpath = os.path.join(self.dest_model_id_or_path, 'config.json')
        # 更新模型config
        with open(src_model_config_fpath) as f, \
                open(dest_model_config_fpath, 'w') as wr:
            config = json.loads(f.read())
            config['vocab_size'] = self.dest_tokenizer.vocab_size
            wr.write(json.dumps(config, ensure_ascii=False, indent=4))
        
        logger.info("rebuild vocab summary:\n %s" % json.dumps(counter, ensure_ascii=False, indent=4))


def load_training_corpus(train_corpus_fpath):
    samples = []
    with open(train_corpus_fpath, 'r') as f:
        for line in f:
            samples.append(line.strip())
            if len(samples) == 2000:
                yield samples[:]
                samples = []
    if samples:
        yield samples[:]


def train_new_tokenizer(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.trainer == 'spm':
        normalization_rule_name = args.normalization_rule_name + "_cf" if args.do_lower_case else args.normalization_rule_name
        spm.SentencePieceTrainer.train(
            input=args.train_corpus_fpath,
            model_prefix=args.model_prefix,
            model_type=args.model_type,
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
            max_sentencepiece_length=args.max_sentencepiece_length,
            normalization_rule_name=normalization_rule_name,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece=args.unk_piece,
            bos_piece=args.bos_piece,
            eos_piece=args.eos_piece,
            pad_piece=args.pad_piece,
            unk_surface=args.unk_surface
        )

        shutil.move(f"{args.model_prefix}.model", f"{args.output_dir}/spiece.model")
        shutil.move(f"{args.model_prefix}.vocab", f"{args.output_dir}/vocab.txt")

        spm_tokenizer = spm.SentencePieceProcessor(
            model_file=f"{args.output_dir}/spiece.model",
            add_bos=True,
            add_eos=True
        )
        spm_tokenizer.vocab_file = f"{args.output_dir}/spiece.model"
        spm_converter = CustomSpmConverter(spm_tokenizer)
        converted = spm_converter.converted()
        converted.save(f"{args.output_dir}/tokenizer.json")
    
    elif args.trainer == 'hft':
        old_tokenizer = AutoTokenizer.from_pretrained(args.src_model_id_or_path, do_lower_case=args.do_lower_case)
        training_corpus = load_training_corpus(args.train_corpus_fpath)
        new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, args.vocab_size)
        new_tokenizer.save_pretrained(args.output_dir)

    # 权重转化
    transporter = EmbeddingTransporter(args)
    transporter.run()


def main():
    parser = argparse.ArgumentParser(description="主程序")

    subparsers = parser.add_subparsers(dest='trainer', required=True)

    # 使用huggingface transformers 训练
    hft_parser = subparsers.add_parser('hft')
    hft_parser.add_argument('--output_dir', type=str, help='输出目录')
    hft_parser.add_argument('--train_corpus_fpath', type=str, help='训练数据集路径')
    hft_parser.add_argument('--src_model_id_or_path', type=str, required=True, help='源模型')
    hft_parser.add_argument('--do_lower_case', default=False, action='store_true', help='是否转小写')
    hft_parser.add_argument('--vocab_size', type=int, default=20000, help='词表大小')

    # 使用sentencepiece训练
    spm_parser = subparsers.add_parser('spm')
    spm_parser.add_argument('--output_dir', type=str, help='输出目录')
    spm_parser.add_argument('--train_corpus_fpath', type=str, help='训练数据集路径')
    spm_parser.add_argument('--src_model_id_or_path', type=str, required=True, help='源模型')
    spm_parser.add_argument('--do_lower_case', default=False, action='store_true', help='是否转小写')
    spm_parser.add_argument('--vocab_size', type=int, default=20000, help='词表大小')
    spm_parser.add_argument('--model-path', required=True)
    spm_parser.add_argument('--model_prefix', type=str, default='spiece', help='模型输出名称')
    spm_parser.add_argument('--model_type', type=str, default='bpe', help='分词算法')
    spm_parser.add_argument('--normalization_rule_name', type=str, default='nmt_nfkc', help='标准化算法')
    spm_parser.add_argument('--input_sentence_size', type=int, default=0, help='最大输入训练语句数量')
    spm_parser.add_argument('--max_sentencepiece_length', type=int, default=16, help='最大piece长度')
    spm_parser.add_argument('--unk_piece', type=str, default='<unk>', help='unk_piece')
    spm_parser.add_argument('--bos_piece', type=str, default='<s>', help='bos_piece')
    spm_parser.add_argument('--eos_piece', type=str, default='</s>', help='eos_piece')
    spm_parser.add_argument('--pad_piece', type=str, default='<pad>', help='pad_piece')
    spm_parser.add_argument('--unk_surface', type=str, default='<unk>', help='unk_surface')

    args = parser.parse_args()
    print(args)
    train_new_tokenizer(args)


if __name__ == "__main__":
    main()