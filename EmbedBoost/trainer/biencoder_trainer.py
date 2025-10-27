import os
import sys
import logging
import argparse
import json

import torch

from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader
from datetime import datetime

from EmbedBoost.model.bgem3 import BGEM3Embedder
from EmbedBoost.model.biencoder_v2 import BiEncoder, BiEncoderWithGradCache
from EmbedBoost.loss.biencoder_loss import MultiInfoNCELoss
from EmbedBoost.dataset.biencoder_dataset import BiEncoderDataset


LOG_DATE_FMT = '%Y‐%m‐%d %H:%M:%S'
LOG_FMT = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stderr,
                    datefmt=LOG_DATE_FMT,
                    format=LOG_FMT)
logger = logging.getLogger(__name__)


def report_loss(epoch, step, step_loss_dict):
    total_loss = step_loss_dict['loss']
    message = f"Losses in epoch-{epoch}, step-{step}, total_loss: {total_loss:.4f}"
    if 'dense_loss' in step_loss_dict:
        message += f", dense_loss: {step_loss_dict['dense_loss']:.4f}"
    if 'sparse_loss' in step_loss_dict:
        message += f", sparse_loss: {step_loss_dict['sparse_loss']:.4f}"
    if 'dense_self_distill_loss' in step_loss_dict:
        message += f", dense_self_distill_loss: {step_loss_dict['dense_self_distill_loss']:.4f}"
    if 'sparse_self_distill_loss' in step_loss_dict:
        message += f", sparse_self_distill_loss: {step_loss_dict['sparse_self_distill_loss']:.4f}"
    logger.info(message)


def report_epoch_loss(epoch, step, avg_loss_dict):
    total_loss = avg_loss_dict['loss']
    message = f"Avg Losses in epoch-{epoch}(step-{step}), total_loss: {total_loss:.4f}"
    if 'dense_loss' in avg_loss_dict:
        message += f", dense_loss: {avg_loss_dict['dense_loss']:.4f}"
    if 'sparse_loss' in avg_loss_dict:
        message += f", sparse_loss: {avg_loss_dict['sparse_loss']:.4f}"
    if 'dense_self_distill_loss' in avg_loss_dict:
        message += f", dense_self_distill_loss: {avg_loss_dict['dense_self_distill_loss']:.4f}"
    if 'sparse_self_distill_loss' in avg_loss_dict:
        message += f", sparse_self_distill_loss: {avg_loss_dict['sparse_self_distill_loss']:.4f}"
    logger.info(message)


def train(args):
    device = torch.device(args.device)
    q_model = BGEM3Embedder(
        args.q_model_name_or_path, 
        use_dense=args.use_dense,
        dense_pooling=args.dense_pooling,
        dense_dim=args.dense_dim,
        use_sparse=args.use_sparse
    )
    q_model.to(device)
    q_tokenizer = q_model.tokenizer

    if args.p_model_name_or_path is not None:
        p_model = BGEM3Embedder(
            args.p_model_name_or_path, 
            use_dense=args.use_dense,
            dense_pooling=args.dense_pooling,
            dense_dim=args.dense_dim,
            use_sparse=args.use_sparse
        )
        p_model.to(device)
        p_tokenizer = p_model.tokenizer
        optimizer = torch.optim.AdamW(chain(q_model.parameters(), p_model.parameters()), lr=args.learning_rate)
    else:
        p_model = q_model
        p_tokenizer = q_tokenizer
        optimizer = torch.optim.AdamW(q_model.parameters(), lr=args.learning_rate)

    if args.resume_from_checkpoint:
        training_state_fpath = os.path.join(args.q_model_name_or_path, "../training-state-ckp.pt")
        if not os.path.exists(training_state_fpath):
            raise ValueError(f"training state checkpoint not found: {training_state_fpath}")
        training_state_checkpoint = torch.load(training_state_fpath)
        optimizer.load_state_dict(training_state_checkpoint['optimizer_state_dict'])
        logger.info("optimizer state loaded.")
    
    loss_func = MultiInfoNCELoss()

    train_files = args.train_data_files.split(",")
    train_dataset = BiEncoderDataset(train_files, q_tokenizer, p_tokenizer, args.max_query_length, args.max_doc_length, args.negative_mode, group_size=args.group_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, drop_last=True, shuffle=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # 将参数转为字典
    args_dict = vars(args)
    args_dict['save_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as wr:
        json.dump(args_dict, wr, indent=4, ensure_ascii=False)

    q_model.train()
    p_model.train()
    if args.gradient_checkpointing:
        q_model.gradient_checkpointing_enable()
        p_model.gradient_checkpointing_enable()

    if args.gradient_cache:
        biencoder = BiEncoderWithGradCache(
            q_model=q_model,
            p_model=p_model,
            chunk_size=args.cache_chunk_size,
            loss_fn=loss_func,
            split_input_fn=None,
            get_rep_fn=None
        )
    else:
        biencoder = BiEncoder(
            q_model=q_model,
            p_model=p_model,
            loss_fn=loss_func,
            get_rep_fn=None
        )
    
    last_epoch = training_state_checkpoint['epoch'] if args.resume_from_checkpoint else 0
    step = training_state_checkpoint['step'] if args.resume_from_checkpoint else 0

    model_kwargs = {
    }
    
    loss_kwargs = {
        'temperature': args.temperature,
        'self_distill': False
        #'use_mrl': args.use_mrl,
        #'mrl_dims': args.mrl_dims,
        #'use_mrl_distill': args.use_mrl_distill,
        #'mrl_distill_weight': args.mrl_distill_weight
    }
    for epoch in range(last_epoch+1, args.train_epochs+1):
        for batch_idx, (q_inputs, p_inputs, n_inputs) in enumerate(tqdm(train_dataloader, disable=args.disable_tqdm)):
            q_inputs = {key: val.to(device) for key, val in q_inputs.items()}
            p_inputs = {key: val.to(device) for key, val in p_inputs.items()}
            if n_inputs:
                n_inputs = {key: val.to(device) for key, val in n_inputs.items()}
            optimizer.zero_grad()
            loss = biencoder(q_inputs, p_inputs, n_inputs, model_kwargs, loss_kwargs)
            optimizer.step()
            step += 1
            if step % args.log_steps == 0:
                logger.info(f"Losses in epoch-{epoch}, step-{step}, total_loss: {loss.item():.4f}")
            
            if args.save_steps > 0 and step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f'ckp_epoch_{epoch}_step_{step}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                q_save_dir = os.path.join(save_dir, "q_model")
                if not os.path.exists(q_save_dir):
                    os.makedirs(q_save_dir)
                q_model.save(q_save_dir)
                p_save_dir = os.path.join(save_dir, "p_model")
                if not os.path.exists(p_save_dir):
                    os.makedirs(p_save_dir)
                p_model.save(p_save_dir)
        
        if args.save_epochs > 0 and epoch % args.save_epochs == 0:
            save_dir = os.path.join(args.output_dir, f'ckp_epoch_{epoch}_step_{step}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            q_save_dir = os.path.join(save_dir, "q_model")
            if not os.path.exists(q_save_dir):
                os.makedirs(q_save_dir)
            q_model.save(q_save_dir)
            p_save_dir = os.path.join(save_dir, "p_model")
            if not os.path.exists(p_save_dir):
                os.makedirs(p_save_dir)
            p_model.save(p_save_dir)
        
            # save optimizer ckp
            training_state = {
                'epoch': epoch,
                'step': step,
                'optimizer_state_dict': optimizer.state_dict()
            }
            ckp_fpath =  os.path.join(save_dir, "training-state-ckp.pt")
            logger.info(f"training_state saved to: {ckp_fpath}")
            torch.save(training_state, ckp_fpath)


def main():
    parser = argparse.ArgumentParser(description="训练句子嵌入模型的命令行脚本")

    parser.add_argument(
        "--train_data_files",
        type=str,
        help="训练数据文件路径, 可以输入多个文件, 用逗号分隔"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="模型输出目录"
    )

    # ------------------
    # 模型参数
    # ------------------
    parser.add_argument(
        "--q_model_name_or_path",
        type=str,
        help="预训练模型路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--p_model_name_or_path",
        type=str,
        default=None,
        help="预训练模型路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="是否继续训练"
    )
    parser.add_argument(
        "--use_dense",
        action="store_true",
        help="使用稀疏向量 (默认: false)"
    )
    parser.add_argument(
        "--dense_pooling",
        type=str,
        default="cls",
        choices=["cls", "mean", "max"],
        help="池化方法 (默认: cls)"
    )
    parser.add_argument(
        "--dense_dim",
        type=int,
        default=128,
        help="稠密向量维度 (默认: 128)"
    )
    parser.add_argument(
        "--use_mrl",
        action="store_true",
        help="使用MRL (默认: false)"
    )
    parser.add_argument(
        "--mrl_dims",
        type=str,
        default="AUTO",
        help="mrl_dims"
    )
    parser.add_argument(
        "--use_mrl_distill",
        action="store_true",
        help="MRL自蒸馏 (默认: false)"
    )
    parser.add_argument(
        "--mrl_distill_weight",
        type=float,
        default=0.2,
        help="mrl_distill_weight (默认: 0.2)"
    )
    parser.add_argument(
        "--use_sparse",
        action="store_true",
        help="使用稀疏向量 (默认: false)"
    )
    parser.add_argument(
        "--max_query_length",
        type=int,
        default=128,
        help="最大Query长度 (默认: 128)"
    )
    parser.add_argument(
        "--max_doc_length",
        type=int,
        default=128,
        help="最大Document长度 (默认: 128)"
    )
    # ------------------
    # 训练参数
    # ------------------
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="梯度检查点"
    )
    parser.add_argument(
        "--gradient_cache",
        action="store_true",
        help="梯度缓存"
    )
    parser.add_argument(
        "--cache_chunk_size",
        type=int,
        default=128,
        help="梯度缓存"
    )

    parser.add_argument(
        "--negative_mode",
        type=str,
        help="使用自蒸馏 (默认: false)"
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="训练轮数 (默认: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="每批次样本数 (默认: 1024)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=0,
        help="显示负样本模式下的组大小 (默认: 0)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="学习率 (默认: 5e-5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="对比学习中的温度系数 (默认: 0.05)"
    )
    parser.add_argument(
        "--self_distill",
        action="store_true",
        help="使用自蒸馏 (默认: false)"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="每训练多少步打印一次日志 (默认: 50)"
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=5,
        help="每多少个epoch保存一次模型 (默认: 5)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="每多少个step保存一次模型 (默认: 10000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备编号 (默认: cuda:0)"
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="禁用训练进度条"
    )

    # 解析参数
    args = parser.parse_args()

    print("🚀 启动训练任务，参数配置如下：")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train(args)


if __name__ == '__main__':
    main()
