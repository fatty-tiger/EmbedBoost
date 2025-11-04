import logging
import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# class InfoNCELoss(nn.Module):
#     def __init__(self):
#         super(InfoNCELoss, self).__init__()

#     def forward(self, q_vectors, p_vectors, n_vectors=None, temperature=0.05, use_mrl=False, mrl_dims=None, use_mrl_distill=False, mrl_distill_weight=0.2):
#         bsz = q_vectors.shape[0]
#         embed_dim = q_vectors.shape[1]
#         device = q_vectors.device

#         # inbatch negative
#         if n_vectors is None:
#             targets = torch.arange(bsz).to(device)

#             if use_mrl:
#                 ensemble_scores = None
#                 loss = None                
                
#                 # 优先操作大的dim，便于蒸馏小的dim
#                 mrl_dims = [int(x) for x in mrl_dims.split(",")]
#                 for i, dim in enumerate(mrl_dims[::-1]):
#                     sub_q_vectors = F.normalize(q_vectors[:, :dim])
#                     sub_p_vectors = F.normalize(p_vectors[:, :dim])
#                     sub_scores = torch.mm(sub_q_vectors, sub_p_vectors.transpose(1, 0)) / temperature
#                     if i == 0:
#                         loss = F.cross_entropy(sub_scores, targets, reduction='mean')
#                     else:
#                         loss += F.cross_entropy(sub_scores, targets, reduction='mean')
                    
#                     if use_mrl_distill:
#                         if i == 0:
#                             ensemble_scores = sub_scores
#                         else:
#                             # 对于较小的dim，添加distill loss
#                             ensemble_scores += sub_scores
#                             teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
#                             mrl_distill_loss = F.cross_entropy(sub_scores, teacher_targets, reduction='mean')
#                             loss += mrl_distill_weight * mrl_distill_loss
#                 loss = loss / len(mrl_dims)
#                 return loss
#             else:
#                 q_vectors = F.normalize(q_vectors, dim=-1)
#                 p_vectors = F.normalize(p_vectors, dim=-1)
#                 scores = torch.mm(q_vectors, p_vectors.transpose(1, 0)) / temperature
#                 loss = F.cross_entropy(scores, targets, reduction='mean')
#                 return loss
        
#         # excitive negative
#         else:
#             n_vectors = n_vectors.reshape(bsz, -1, embed_dim)
#             num_negatives = n_vectors.size(1)

#             # 扩展查询向量以匹配负样本的维度 [batch_size, num_negatives, embedding_dim]
#             q_vectors_expanded = q_vectors.unsqueeze(1).expand(-1, num_negatives, -1)
            
#             q_pos_scores = (torch.sum(q_vectors * p_vectors, dim=1) / temperature).unsqueeze(1)
#             q_neg_scores = torch.sum(q_vectors_expanded * n_vectors, dim=2)  / temperature

#             # 合并正负样本相似度
#             # [batch_size, 1 + num_negatives]
#             scores = torch.cat([q_pos_scores, q_neg_scores], dim=1)
#             # 标签：正样本始终在位置0
#             targets = torch.zeros(bsz, dtype=torch.long, device=device)
#             # logger.info(f"scores: {scores.shape}, targets: {targets.shape}")
#             loss = F.cross_entropy(scores, targets, reduction='mean')
#             return loss
        

class MultiInfoNCELoss(nn.Module):
    def __init__(self):
        super(MultiInfoNCELoss, self).__init__()

    def forward(self, q_vectors_dict, p_vectors_dict, n_vectors_dict, temperature=0.05, self_distill=False):
        bsz = 0
        device = None
        for k in ['dense_vectors', 'sparse_vectors', 'colbert_vectors']:
            if k in q_vectors_dict:
                bsz = q_vectors_dict[k].shape[0]
                device = q_vectors_dict[k].device
                break

        targets = torch.arange(bsz).to(device)
        loss = 0
        ensemble_score_list = []

        # inbatch negative
        if n_vectors_dict is None:
            if 'dense_vectors' in q_vectors_dict and 'dense_vectors' in p_vectors_dict:
                dense_scores = torch.mm(q_vectors_dict['dense_vectors'], p_vectors_dict['dense_vectors'].transpose(1, 0)) / temperature
                loss += F.cross_entropy(dense_scores, targets, reduction='mean')
                ensemble_score_list.append(dense_scores)
            
            if 'sparse_vectors' in q_vectors_dict and 'sparse_vectors' in p_vectors_dict:
                q_sparse_vectors = q_vectors_dict['sparse_vectors']
                p_sparse_vectors = p_vectors_dict['sparse_vectors']
                sparse_scores = torch.spmm(q_sparse_vectors, p_sparse_vectors.t()).to_dense()
                sparse_scores = 0.3 * sparse_scores
                sparse_loss = F.cross_entropy(sparse_scores, targets, reduction='mean')
                ensemble_score_list.append(sparse_scores)
                loss += 0.1 * sparse_loss
            
            if 'colbert_vectors' in q_vectors_dict and 'colbert_vectors' in p_vectors_dict:
                # colbert_vectors: [bsz, seq_len, colbert_dim]
                # token_scores: [bsz, seq_len, bsz, seq_len]
                token_scores = torch.einsum('qin,pjn->qipj', q_vectors_dict['colbert_vectors'], p_vectors_dict['colbert_vectors'])
                # colbert_scores: [bsz, seq_len, bsz]
                colbert_scores, _ = token_scores.max(-1)
                # colbert_scores: [bsz, bsz]
                colbert_scores = colbert_scores.sum(1) / q_vectors_dict['attention_mask'][:, 1:].sum(-1, keepdim=True)
                colbert_scores = colbert_scores / temperature

                ensemble_score_list.append(colbert_scores)
                colbert_loss = F.cross_entropy(colbert_scores, targets, reduction='mean')
                loss += colbert_loss
            
            if len(ensemble_score_list) > 1:
                ensemble_scores = sum(ensemble_score_list)
                ensemble_loss = F.cross_entropy(ensemble_scores, targets, reduction='mean')
                loss += ensemble_loss
                loss = loss / (len(ensemble_score_list) + 1)
            
            return loss
