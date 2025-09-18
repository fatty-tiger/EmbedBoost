import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


class InbatchNegInfoNCELoss(nn.Module):
    def __init__(self):
        super(InbatchNegInfoNCELoss, self).__init__()

    def forward(self, q_vectors, p_vectors, temperature=0.05, offset=0):
        bsz = q_vectors.shape[0]
        targets = torch.arange(bsz).to(q_vectors.device)
        if offset > 0:
            targets = targets.add(offset)

        scores = torch.mm(q_vectors, p_vectors.transpose(1, 0))
        scores = scores / temperature
        loss = F.cross_entropy(scores, targets, reduction='mean')
        return loss


class CommonInfoNCELoss(nn.Module):
    def __init__(self):
        super(CommonInfoNCELoss, self).__init__()

    def forward(self, q_encoded, pos_encoded, neg_encoded, temperature=0.05, use_sparse=False, sparse_weight=0.3, self_distill=False, distill_weight=0.2):
        ret_dict = {}

        q_dense_vecs, pos_dense_vecs, neg_dense_vecs = q_encoded['dense_vectors'], pos_encoded['dense_vectors'], neg_encoded['dense_vectors']
        
        batch_size = q_dense_vecs.size(0)
        embed_dim = q_dense_vecs.size(1)
        neg_dense_vecs = neg_dense_vecs.reshape(batch_size, -1, embed_dim)
        num_negatives = neg_dense_vecs.size(1)

        # 扩展查询向量以匹配负样本的维度 [batch_size, num_negatives, embedding_dim]
        q_dense_vecs_expanded = q_dense_vecs.unsqueeze(1).expand(-1, num_negatives, -1)
        
        q_pos_dense_scores = (torch.sum(q_dense_vecs * pos_dense_vecs, dim=1) / temperature).unsqueeze(1)
        q_neg_dense_scores = torch.sum(q_dense_vecs_expanded * neg_dense_vecs, dim=2)  / temperature

        # 合并正负样本相似度
        # [batch_size, 1 + num_negatives]
        dense_scores = torch.cat([q_pos_dense_scores, q_neg_dense_scores], dim=1)
        # 标签：正样本始终在位置0
        targets = torch.zeros(batch_size, dtype=torch.long, device=q_dense_vecs.device)
        dense_loss = F.cross_entropy(dense_scores, targets, reduction='mean')
        ret_dict['dense_loss'] = dense_loss
        loss = dense_loss
        
        if use_sparse:
            q_sparse_vecs, pos_sparse_vecs, neg_sparse_vecs = q_encoded['sparse_vectors'], pos_encoded['sparse_vectors'], neg_encoded['sparse_vectors']
            neg_embed_dim = q_sparse_vecs.size(1)
            neg_sparse_vecs = neg_sparse_vecs.reshape(batch_size, -1, neg_embed_dim)
            neg_num_negatives = neg_sparse_vecs.size(1)
            
            q_sparse_vecs_expanded = q_sparse_vecs.unsqueeze(1).expand(-1, neg_num_negatives, -1)

            q_pos_sparse_scores = (torch.sum(q_sparse_vecs * pos_sparse_vecs, dim=1) / temperature).unsqueeze(1)
            q_neg_sparse_scores = torch.sum(q_sparse_vecs_expanded * neg_sparse_vecs, dim=2)  / temperature

            sparse_scores = torch.cat([q_pos_sparse_scores, q_neg_sparse_scores], dim=1)
            # 标签：正样本始终在位置0
            sparse_loss = F.cross_entropy(sparse_scores, targets, reduction='mean')
            loss += sparse_loss * sparse_weight
            ret_dict['sparse_loss'] = sparse_loss

        if self_distill:
            ensemble_scores = dense_scores + sparse_scores
            teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)            
            
            dense_self_distill_loss = F.cross_entropy(dense_scores, teacher_targets, reduction='mean')
            loss += distill_weight * dense_self_distill_loss
            ret_dict['dense_self_distill_loss'] = dense_self_distill_loss
            
            sparse_self_distill_loss = F.cross_entropy(sparse_scores, teacher_targets, reduction='mean')
            loss += distill_weight * sparse_self_distill_loss
            ret_dict['sparse_self_distill_loss'] = sparse_self_distill_loss
        
        ret_dict['loss'] = loss
        return ret_dict
