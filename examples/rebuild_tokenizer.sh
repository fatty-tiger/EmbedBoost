python src/tokenizer_boost/run.py \
    --output_dir output/bert-base-chinese-bpe \
    --trainer_type spm \
    --train_corpus_fpath your/tokenizer/corpus.txt \
    --model_prefix spiece \
    --model_type bpe \
    --vocab_size 40000 \
    --do_lower_case \
    --input_sentence_size 200000 \
    --max_sentencepiece_length 5 \
    --normalization_rule_name nmt_nfkc \
    --unk_piece "[UNK]" \
    --bos_piece "[CLS]" \
    --eos_piece "[SEP]" \
    --pad_piece "[PAD]" \
    --unk_surface "<UNK>"
    --src_model_id_or_path google/bert-base-chinese

