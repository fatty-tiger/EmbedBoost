import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


class InbatchNegInfoNCELoss(nn.Module):
    def __init__(self):
        super(InbatchNegInfoNCELoss, self).__init__()

    def forward(self, q_encoded, p_encoded, temperature=0.05, offset=0, use_sparse=False, sparse_weight=0.3, self_distill=False, distill_weight=0.2):
        q_dense_vecs, p_dense_vecs = q_encoded['dense_vectors'], p_encoded['dense_vectors']
        bsz = q_dense_vecs.shape[0]
        targets = torch.arange(bsz).to(q_dense_vecs.device)
        if offset > 0:
            targets = targets.add(offset)

        ret_dict = {}

        dense_scores = torch.mm(q_dense_vecs, p_dense_vecs.transpose(1, 0))
        dense_scores = dense_scores / temperature
        dense_loss = F.cross_entropy(dense_scores, targets, reduction='mean')
        ret_dict['dense_loss'] = dense_loss
        loss = dense_loss
        
        if use_sparse:
            q_sparse_vecs, p_sparse_vecs = q_encoded['sparse_vectors'], p_encoded['sparse_vectors']
            sparse_scores = torch.mm(q_sparse_vecs, p_sparse_vecs.transpose(1,0))
            sparse_scores = sparse_scores / temperature
            sparse_loss = F.cross_entropy(sparse_scores, targets, reduction='mean')
            ret_dict['sparse_loss'] = sparse_loss
            loss += sparse_loss * sparse_weight

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


class CommonInfoNCELoss(nn.Module):
    """监督模式In-Batch损失"""
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
