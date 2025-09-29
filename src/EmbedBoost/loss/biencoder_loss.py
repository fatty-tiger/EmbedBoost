import logging
import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, q_vectors, p_vectors, n_vectors=None, temperature=0.05):
        bsz = q_vectors.shape[0]
        embed_dim = q_vectors.shape[1]
        device = q_vectors.device

        # inbatch negative
        if n_vectors is None:
            targets = torch.arange(bsz).to(device)
            scores = torch.mm(q_vectors, p_vectors.transpose(1, 0))
            scores = scores / temperature
            loss = F.cross_entropy(scores, targets, reduction='mean')
            return loss
        # excitive negative
        else:
            n_vectors = n_vectors.reshape(bsz, -1, embed_dim)
            num_negatives = n_vectors.size(1)

            # 扩展查询向量以匹配负样本的维度 [batch_size, num_negatives, embedding_dim]
            q_vectors_expanded = q_vectors.unsqueeze(1).expand(-1, num_negatives, -1)
            
            q_pos_scores = (torch.sum(q_vectors * p_vectors, dim=1) / temperature).unsqueeze(1)
            q_neg_scores = torch.sum(q_vectors_expanded * n_vectors, dim=2)  / temperature

            # 合并正负样本相似度
            # [batch_size, 1 + num_negatives]
            scores = torch.cat([q_pos_scores, q_neg_scores], dim=1)
            # 标签：正样本始终在位置0
            targets = torch.zeros(bsz, dtype=torch.long, device=device)
            # logger.info(f"scores: {scores.shape}, targets: {targets.shape}")
            loss = F.cross_entropy(scores, targets, reduction='mean')
            return loss