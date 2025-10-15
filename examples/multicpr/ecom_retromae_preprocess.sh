/data/miniconda3/envs/deepnlp/bin/python EmbedBoost/retromae/pretrain/preprocess.py \
    --input_data_fpath /data/jiangjie/FattyEmbedding/data/multicpr/ecom/corpus.txt.sampled \
    --tokenizer_name /data/pretrained_models/ernie-3.0-medium-zh \
    --max_seq_length 256 \
    --output_dir output/multicpr/ecom/v1/datasets/retromae_demo