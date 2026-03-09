import os
import sys
import time
import logging
import argparse
import json
import collections

import torch
import torch.nn as nn

from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader
from datetime import datetime

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import LinearLR, SequentialLR

# from EmbedBoost.model.bgem3 import BGEM3Embedder
# from EmbedBoost.model.biencoder import BiEncoder, BiEncoderWithGradCache
from EmbedBoost.model.biencoder import BGEM3Biencoder
from EmbedBoost.loss.biencoder_loss import MultiInfoNCELoss
from EmbedBoost.dataset.biencoder_dataset import BiEncoderDataset

from EmbedBoost.evaluate.biencoder_evaluator import run_evaluate


LOG_DATE_FMT = '%Y‐%m‐%d %H:%M:%S'
LOG_FMT = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stderr,
                    datefmt=LOG_DATE_FMT,
                    format=LOG_FMT)
logger = logging.getLogger(__name__)



def accum_step_loss(loss_dict, accum_loss_dict):
    accum_loss_dict['total_loss'] += loss_dict['total_loss']
    if 'dense_loss' in loss_dict:
        accum_loss_dict['dense_loss'] += loss_dict['dense_loss']
    if 'sparse_loss' in loss_dict:
        accum_loss_dict['sparse_loss'] += loss_dict['sparse_loss']
    if 'colbert_loss' in loss_dict:
        accum_loss_dict['colbert_loss'] += loss_dict['colbert_loss']
    if 'ensemble_loss' in loss_dict:
        accum_loss_dict['ensemble_loss'] += loss_dict['ensemble_loss']


def report_step_info(epoch, step, lr, loss_dict):
    total_loss = loss_dict['total_loss']
    logging.info(f"Losses in epoch-{epoch}(step-{step} lr-{lr:.8f}):")
    message = f"total_loss: {total_loss:.2f}"
    if 'dense_loss' in loss_dict:
        message += f", dense_loss: {loss_dict['dense_loss']:.2f}"
    if 'sparse_loss' in loss_dict:
        message += f", sparse_loss: {loss_dict['sparse_loss']:.2f}"
    if 'splade_reg' in loss_dict:
        message += f", splade_reg: {loss_dict['splade_reg']:.2f}"
    if 'colbert_loss' in loss_dict:
        message += f", colbert_loss: {loss_dict['colbert_loss']:.2f}"
    if 'ensemble_loss' in loss_dict:
        message += f", ensemble_loss: {loss_dict['ensemble_loss']:.2f}"
    # if 'dense_self_distill_loss' in loss_dict:
    #     message += f", dense_self_distill_loss: {loss_dict['dense_self_distill_loss']:.4f}"
    # if 'sparse_self_distill_loss' in loss_dict:
    #     message += f", sparse_self_distill_loss: {loss_dict['sparse_self_distill_loss']:.4f}"
    logger.info(message)

def report_avg_loss(epoch, accum_steps, loss_dict):
    total_loss = loss_dict['total_loss'] / accum_steps
    message = f"Avg Losses in epoch-{epoch}, accum_steps:{accum_steps}, total_loss: {total_loss:.2f}"
    if 'dense_loss' in loss_dict:
        dense_loss = loss_dict['dense_loss'] / accum_steps
        message += f", dense_loss: {dense_loss:.2f}"
    if 'sparse_loss' in loss_dict:
        sparse_loss = loss_dict['sparse_loss'] / accum_steps
        message += f", sparse_loss: {sparse_loss:.2f}"
    if 'colbert_loss' in loss_dict:
        colbert_loss = loss_dict['colbert_loss'] / accum_steps
        message += f", colbert_loss: {colbert_loss:.2f}"
    if 'ensemble_loss' in loss_dict:
        ensemble_loss = loss_dict['ensemble_loss'] / accum_steps
        message += f", ensemble_loss: {ensemble_loss:.2f}"
    # if 'dense_self_distill_loss' in loss_dict:
    #     message += f", dense_self_distill_loss: {loss_dict['dense_self_distill_loss']:.4f}"
    # if 'sparse_self_distill_loss' in loss_dict:
    #     message += f", sparse_self_distill_loss: {loss_dict['sparse_self_distill_loss']:.4f}"
    logger.info(message)


def save_model(epoch, step, args, model_to_save, optimizer=None):
    save_dir = os.path.join(args.output_dir, f'ckp_epoch_{epoch}_step_{step}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, "q_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save.save(save_dir)

    if optimizer is not None:
        training_state = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict()
        }
        opt_ckp_fpath =  os.path.join(save_dir, "training-state-ckp.pt")
        logger.info(f"training_state saved to: {opt_ckp_fpath}")
        torch.save(training_state, opt_ckp_fpath)

def train(args):
    # torch.autograd.set_detect_anomaly(True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

        # 将参数转为字典
        args_dict = vars(args)
        args_dict['save_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as wr:
            json.dump(args_dict, wr, indent=4, ensure_ascii=False)
        
        print("🚀 启动训练任务，参数配置如下：")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")

    if world_size > 1:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=world_size, rank=rank)
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)            

    # 初始化q_model
    biencoder = BGEM3Biencoder(
        args.q_model_name_or_path, 
        use_dense=args.use_dense,
        dense_pooling=args.dense_pooling,
        dense_dim=args.dense_dim,
        dense_normalize=args.dense_normalize,
        use_sparse=args.use_sparse,
        sparse_mode=args.sparse_mode,
        sparse_normalize=args.sparse_normalize,
        use_colbert=args.use_colbert,
        colbert_dim=args.colbert_dim,
        colbert_normalize=args.colbert_normalize,
    )
    biencoder.to(device)
    q_tokenizer = biencoder.tokenizer
    p_tokenizer = q_tokenizer

    if world_size > 1:
        biencoder = DDP(biencoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.find_unused_parameters)
    
    optimizer = torch.optim.AdamW(biencoder.parameters(), lr=args.learning_rate)

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
    
    # Use DistributedSampler if DDP is enabled
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            collate_fn=train_dataset.collate_fn, 
            drop_last=True,
            # num_workers=args.num_workers
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            collate_fn=train_dataset.collate_fn,
            drop_last=True, 
            shuffle=True
        )


    # from transformers import get_linear_schedule_with_warmup
    # total_training_steps = len(train_dataloader) * args.train_epochs
    # warmup_steps = int(total_training_steps * args.warmup_ratio)
    # if rank in [-1, 0]:
    #     logger.info(f"Total training steps: {total_training_steps}, warmup steps: {warmup_steps}")
    
    # # Create learning rate scheduler
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_training_steps
    # )
    
    total_training_steps = len(train_dataloader) * args.train_epochs
    
    # TODO: 学习率调度优化
    # schedulers = []
    # if args.warmup_ratio > 0:
    #     # 2. 定义第一个调度器：Warmup (从 initial_lr * 0.1 增长到 initial_lr)
    #     warmup_steps = int(total_training_steps * args.warmup_ratio)
    #     scheduler_warmup = LinearLR(optimizer, 
    #                                 start_factor=1e-6, 
    #                                 end_factor=args.learning_rate, 
    #                                 total_iters=warmup_steps)
    #     schedulers.append(scheduler_warmup)
    
    # # 3. 定义第二个调度器：Linear Decay (从 initial_lr 衰减到 end_lr)
    # # 注意：此时的基准 lr 已经是 initial_lr 了
    # scheduler_decay = LinearLR(optimizer, 
    #                         start_factor=1.0, 
    #                         end_factor=end_lr/initial_lr, 
    #                         total_iters=decay_epochs)

    # # 4. 使用 SequentialLR 将两者串联
    # # milestones 列表表示在哪个时间点切换到下一个调度器
    # scheduler = SequentialLR(optimizer, 
    #                         schedulers=[scheduler_warmup, scheduler_decay], 
    #                         milestones=[warmup_epochs])

    # 1e-5 -> 1e-6
    scheduler = LinearLR(optimizer, 
                        start_factor=1.0, 
                        end_factor=args.end_learning_rate/args.learning_rate, 
                        total_iters=100)

    biencoder.train()
    
    # gradient checkpointing
    if args.gradient_checkpointing:
        if world_size > 1:
            biencoder.module.gradient_checkpointing_enable()
        else:
            biencoder.gradient_checkpointing_enable()
    
    last_epoch = training_state_checkpoint['epoch'] if args.resume_from_checkpoint else 0
    step = training_state_checkpoint['step'] if args.resume_from_checkpoint else 0
    step_accum_loss_dict = collections.defaultdict(float)

    model_kwargs = {
    }
    
    loss_kwargs = {
        'temperature': args.temperature,
        'circle_mp': args.circle_mp,
        'circle_mn': args.circle_mn,
        'circle_gamma_p': args.circle_gamma_p,
        'circle_gamma_n': args.circle_gamma_n,
        'colbert_chunk_size': args.colbert_chunk_size,
        'self_distill': args.self_distill,
        'self_distill_steps': args.self_distill_steps,
    }
    for epoch in range(last_epoch+1, args.train_epochs+1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)  # Shuffle data differently for each epoch across GPUs
        
        epoch_accum_steps = 0
        epoch_accum_loss_dict = collections.defaultdict(float)

        for batch_idx, (q_inputs, p_inputs, n_inputs) in enumerate(tqdm(train_dataloader, disable=args.disable_tqdm or rank not in [0, -1])):
            q_inputs = {f"q_{key}": val.to(device) for key, val in q_inputs.items()}
            p_inputs = {f"p_{key}": val.to(device) for key, val in p_inputs.items()}
            if n_inputs:
                n_inputs = {f"n_{key}": val.to(device) for key, val in n_inputs.items()}
            else:
                n_inputs = {
                    'n_input_ids': None,
                    'n_attention_mask': None,
                    'n_token_type_ids': None
                }
            
            optimizer.zero_grad()
            loss_kwargs['step'] = step
            
            bsz = q_inputs['q_input_ids'].shape[0]
            targets = torch.arange(bsz).to(device)
            q_encoded, p_encoded, n_encoded = biencoder(**q_inputs, **p_inputs, **n_inputs)
            loss, loss_dict = loss_func(q_encoded, p_encoded, n_encoded, targets, **loss_kwargs)
            loss.backward()
            
            nn.utils.clip_grad_norm_(biencoder.parameters(), max_norm=1.0)

            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            scheduler.step()
            
            if rank in [0, -1]:
                step += 1
                epoch_accum_steps += 1

            if step % args.log_steps == 0 and rank in [0, -1]:
                report_step_info(epoch, step, current_lr, loss_dict)
                accum_step_loss(loss_dict, epoch_accum_loss_dict)
                accum_step_loss(loss_dict, step_accum_loss_dict)
            
            if rank in [0, -1] and step % args.log_avg_steps == 0:
                report_avg_loss(epoch, args.log_avg_steps, step_accum_loss_dict)
                step_accum_loss_dict.clear()
            
            if args.save_steps > 0 and step % args.save_steps == 0 and rank in [0, -1]:
                biencoder_to_save = biencoder.module if world_size > 1 else biencoder
                save_model(epoch, step, args, biencoder_to_save, optimizer=None)

            if rank in [0, -1] and step % args.eval_steps == 0:
                ts = time.time()
                biencoder.eval()
                biencoder_to_save = biencoder.module if world_size > 1 else biencoder
                run_evaluate(biencoder_to_save, recall_topK=200, limit=2000)
                biencoder.train()
                cost = int(time.time() - ts)
                logger.info(f"Evaluate cost: {cost} secs")
                # logger.info(f"Eval metric on Epoch-{epoch} Step-{step}:\n {json.dumps(eval_metric_dict, ensure_ascii=False, indent=4)}")

        if rank in [0, -1]:
            report_avg_loss(epoch, epoch_accum_steps, epoch_accum_loss_dict)

        if args.save_epochs > 0 and epoch % args.save_epochs == 0 and rank in [0, -1]:
            biencoder_to_save = biencoder.module if world_size > 1 else biencoder
            save_model(epoch, step, args, biencoder_to_save, optimizer=optimizer)
    
    if world_size > 1:
        dist.destroy_process_group()


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
        "--dense_normalize",
        action="store_true",
        help="稠密向量归一化"
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
        "--sparse_mode",
        type=str,
        default="splade",
        help="sparse_mode"
    )
    parser.add_argument(
        "--sparse_normalize",
        action="store_true",
        help="稀疏向量归一化"
    )

    parser.add_argument(
        "--use_colbert",
        action="store_true",
        help="use_colbert (默认: false)"
    )
    parser.add_argument(
        "--colbert_dim",
        type=int,
        default=64,
        help="colbert_dim"
    )
    parser.add_argument(
        "--colbert_normalize",
        action="store_true",
        help="colbert向量归一化"
    )
    parser.add_argument(
        "--colbert_chunk_size",
        type=int,
        default=0,
        help="colbert_chunk_size"
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
        "--end_learning_rate",
        type=float,
        default=5e-5,
        help="学习率 (默认: 5e-5)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup比例，表示前百分之多少的训练步骤用于线性warmup (默认: 0.1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="对比学习中的温度系数 (默认: 0.05)"
    )
    parser.add_argument(
        "--circle_mp",
        type=float,
        default=0.2,
        help="circle_mp"
    )
    parser.add_argument(
        "--circle_gamma_p",
        type=float,
        default=20.0,
        help="circle_gamma_p"
    )
    parser.add_argument(
        "--circle_mn",
        type=float,
        default=0.2,
        help="circle_mn"
    )
    parser.add_argument(
        "--circle_gamma_n",
        type=float,
        default=20.0,
        help="circle_gamma_n"
    )
    parser.add_argument(
        "--self_distill",
        action="store_true",
        help="使用自蒸馏 (默认: false)"
    )
    parser.add_argument(
        "--self_distill_steps",
        type=int,
        default=-1,
        help="self_distill_steps"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="每训练多少步打印一次日志 (默认: 50)"
    )
    parser.add_argument(
        "--log_avg_steps",
        type=int,
        default=100,
        help="每训练多少步打印一次平均损失"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="eval_steps"
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
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="Backend for distributed training (default: nccl)"
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="env://",
        help="Url for distributed training (default: env://)"
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Whether to find unused parameters in DDP"
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

    train(args)


if __name__ == '__main__':
    main()
