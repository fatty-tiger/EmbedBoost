import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from EmbedBoost.loss.circle_loss import CircleLoss

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
        self.circle_loss_func = CircleLoss()

    def forward(self, q_vectors_dict, p_vectors_dict, n_vectors_dict, targets,
                loss_type='infonce',
                temperature=0.05,
                circle_mp=0.2, circle_gamma_p=20.0, 
                circle_mn=0.2, circle_gamma_n=20.0,
                colbert_chunk_size=0,
                step=0,
                self_distill=False, self_distill_steps=-1
            ):

        loss = 0
        extra_loss_dict = {}
        ensemble_score_list = []
        
        temperature = temperature if loss_type == 'infonce' else 1.0

        # inbatch negative
        if n_vectors_dict is None:
            dense_weight = 1.0
            sparse_weight = 1.0
            colbert_weight = 1.0
            splade_reg_weight = 0.01
            ensemble_weight = 0.5
                
            if 'dense_vectors' in q_vectors_dict and 'dense_vectors' in p_vectors_dict:
                dense_scores = torch.mm(q_vectors_dict['dense_vectors'], p_vectors_dict['dense_vectors'].transpose(1, 0)) / temperature
                dense_loss = dense_weight * F.cross_entropy(dense_scores, targets, reduction='mean')
                loss = loss + dense_loss
                ensemble_score_list.append(dense_scores)
                extra_loss_dict['dense_loss'] = dense_loss.detach().cpu().item()
            
            if 'sparse_vectors' in q_vectors_dict and 'sparse_vectors' in p_vectors_dict:
                q_sparse_vectors = q_vectors_dict['sparse_vectors']
                p_sparse_vectors = p_vectors_dict['sparse_vectors']
                if q_sparse_vectors.layout == torch.sparse_coo and p_sparse_vectors.layout == torch.sparse_coo:
                    sparse_scores = torch.spmm(q_sparse_vectors, p_sparse_vectors.t()).to_dense()
                else:
                    # 如果sparse_vectors进行了归一化，注意添加温度参数
                    sparse_scores = torch.mm(q_sparse_vectors, p_sparse_vectors.transpose(1, 0))
                sparse_loss = sparse_weight * F.cross_entropy(sparse_scores, targets, reduction='mean')
                sparse_loss = torch.clamp(sparse_loss, max=100.0)
                ensemble_score_list.append(sparse_scores)
                splade_reg = splade_reg_weight * (torch.sum(torch.mean(torch.abs(q_sparse_vectors), dim=0) ** 2) + torch.sum(torch.mean(torch.abs(p_sparse_vectors), dim=0) ** 2))
                loss = loss + sparse_loss + splade_reg
                extra_loss_dict['sparse_loss'] = sparse_loss.detach().cpu().item()
                extra_loss_dict['splade_reg'] = splade_reg.detach().cpu().item()
            
            if 'colbert_vectors' in q_vectors_dict and 'colbert_vectors' in p_vectors_dict:
                if colbert_chunk_size > 0:
                    # chunked colbert_scores
                    bsz = q_vectors_dict['colbert_vectors'].shape[0]
                    scores_list = []
                    for i in range(0, bsz, colbert_chunk_size):
                        end_i = min(i + colbert_chunk_size, bsz)
                        chunk_vector = q_vectors_dict['colbert_vectors'][i: end_i]  # [chunk_size, seq_len, dim]
                        chunk_mask = q_vectors_dict['attention_mask'][i: end_i]
                        # Compute scores for this chunk
                        # shape: chunk_size, seq_len, bsz, seq_len
                        scores = torch.einsum('qin,pjn->qipj', chunk_vector, p_vectors_dict['colbert_vectors']).max(-1)[0].sum(1)
                        scores = scores / chunk_mask[:, 1:].sum(-1, keepdim=True)
                        scores_list.append(scores)
                    colbert_scores = torch.cat(scores_list, dim=0)
                    colbert_scores = colbert_scores / temperature
                else:
                    # colbert_vectors: [bsz, seq_len, colbert_dim]
                    # token_scores: [bsz, seq_len, bsz, seq_len]
                    token_scores = torch.einsum('qin,pjn->qipj', q_vectors_dict['colbert_vectors'], p_vectors_dict['colbert_vectors'])
                    # colbert_scores: [bsz, seq_len, bsz]
                    colbert_scores, _ = token_scores.max(-1)
                    # colbert_scores: [bsz, bsz]
                    colbert_scores = colbert_scores.sum(1) / q_vectors_dict['attention_mask'][:, 1:].sum(-1, keepdim=True)
                    colbert_scores = colbert_scores / temperature
                
                ensemble_score_list.append(colbert_scores)
                colbert_loss = colbert_weight * F.cross_entropy(colbert_scores, targets, reduction='mean')
                colbert_loss = torch.clamp(colbert_loss, max=100.0)
                loss = loss + colbert_loss
                extra_loss_dict['colbert_loss'] = colbert_loss.detach().cpu().item()
            
            # 默认会使用ensemble_loss
            if len(ensemble_score_list) > 1:
                ensemble_scores = sum(ensemble_score_list)
                ensemble_loss = ensemble_weight * F.cross_entropy(ensemble_scores, targets, reduction='mean')
                ensemble_loss = torch.clamp(ensemble_loss, max=50.0)
                extra_loss_dict['ensemble_loss'] = ensemble_loss.detach().cpu().item()
                loss = loss + ensemble_loss
            
            # if self_distill and step >= self_distill_steps:
            #     teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
            #     if 'dense_vectors' in q_vectors_dict and 'dense_vectors' in p_vectors_dict:
            #         dense_distill_loss = F.cross_entropy(dense_scores, teacher_targets, reduction='mean')
            #         loss += 0.1 * dense_distill_loss
            #     if 'sparse_vectors' in q_vectors_dict and 'sparse_vectors' in p_vectors_dict:
            #         sparse_distill_loss = F.cross_entropy(sparse_scores, teacher_targets, reduction='mean')
            #         loss += 0.3 * sparse_distill_loss
            #     if 'colbert_vectors' in q_vectors_dict and 'colbert_vectors' in p_vectors_dict:
            #         colbert_distill_loss = F.cross_entropy(colbert_scores, teacher_targets, reduction='mean')
            #         loss += 0.1 * colbert_distill_loss
                # excitive negative
        
        else:
            dense_weight = 1.0
            sparse_weight = 1.0
            splade_reg_weight = 0.01

            if 'dense_vectors' in q_vectors_dict and 'dense_vectors' in p_vectors_dict and 'dense_vectors' in n_vectors_dict:
                # q_dense_vectors: [bsz, dense_dim]
                q_dense_vectors = q_vectors_dict['dense_vectors']
                # p_dense_vectors: [bsz, dense_dim]
                p_dense_vectors = p_vectors_dict['dense_vectors']
                
                bsz, dense_dim = q_dense_vectors.shape
                device = q_dense_vectors.device

                # n_dense_vectors: [bsz, n_negatives, dense_dim]
                n_dense_vectors = n_vectors_dict['dense_vectors'].view(bsz, -1, dense_dim)
                num_negatives = n_dense_vectors.shape[1]
                
                # 计算query和正样本的相似度(在dim=1上按位相乘并求和)
                # q_pos_scores: [batch_size, 1]
                q_pos_scores = (torch.sum(q_dense_vectors * p_dense_vectors, dim=1) / temperature).unsqueeze(1)

                # 计算query和负样本的相似度
                q_dense_vectors_expanded = q_dense_vectors.unsqueeze(1).expand(-1, num_negatives, -1)
                # logger.info(f"q_dense_vectors_expanded.shape: {q_dense_vectors_expanded.shape}")
                # logger.info(f"n_dense_vectors.shape: {n_dense_vectors.shape}")
                # q_neg_scores: [batch_size, num_negatives]
                q_neg_scores = torch.sum(q_dense_vectors_expanded * n_dense_vectors, dim=2)  / temperature


                # infoNCE
                if loss_type == 'infonce':
                    # 合并正负样本相似度
                    # scores: [batch_size, 1 + num_negatives]
                    scores = torch.cat([q_pos_scores, q_neg_scores], dim=1)
                    # 标签：正样本始终在位置0
                    targets = torch.zeros(bsz, dtype=torch.long, device=device)
                    dense_loss = dense_weight * F.cross_entropy(scores, targets, reduction='mean')
                    loss = loss + dense_loss

                if loss_type == 'circleloss':
                    dense_loss = dense_weight * self.circle_loss_func(
                        q_pos_scores, q_neg_scores, mp=circle_mp, mn=circle_mn, gamma_p=circle_gamma_p, gamma_n=circle_gamma_n)
                    loss = loss + dense_loss
                
                extra_loss_dict['dense_loss'] = dense_loss.detach().cpu().item()
                

            if 'sparse_vectors' in q_vectors_dict and 'sparse_vectors' in p_vectors_dict and 'sparse_vectors' in n_vectors_dict:
                # q_sparse_vectors: [bsz, sparse_dim]
                q_sparse_vectors = q_vectors_dict['sparse_vectors']
                # p_sparse_vectors: [bsz, sparse_dim]
                p_sparse_vectors = p_vectors_dict['sparse_vectors']
                bsz, sparse_dim = q_sparse_vectors.shape
                device = q_sparse_vectors.device

                # n_sparse_vectors: [bsz, n_negatives, sparse_dim]
                n_sparse_vectors = n_vectors_dict['sparse_vectors'].view(bsz, -1, sparse_dim)
                num_negatives = n_sparse_vectors.shape[1]
                
                # 计算query和正样本的相似度(在dim=1上按位相乘并求和)
                # q_pos_scores: [batch_size, 1]
                sparse_pos_scores = (torch.sum(q_sparse_vectors * p_sparse_vectors, dim=1) / temperature).unsqueeze(1)

                # 计算query和负样本的相似度
                q_sparse_vectors_expanded = q_sparse_vectors.unsqueeze(1).expand(-1, num_negatives, -1)
                # q_neg_scores: [batch_size, num_negatives]
                sparse_neg_scores = torch.sum(q_sparse_vectors_expanded * n_sparse_vectors, dim=2)  / temperature

                # InfoNCE
                if loss_type == 'infonce':
                    # 合并正负样本相似度
                    # scores: [batch_size, 1 + num_negatives]
                    scores = torch.cat([sparse_pos_scores, sparse_neg_scores], dim=1)
                    # 标签：正样本始终在位置0
                    targets = torch.zeros(bsz, dtype=torch.long, device=device)
                    sparse_loss = sparse_weight * F.cross_entropy(scores, targets, reduction='mean')

                if loss_type == 'circleloss':
                    # circleloss
                    sparse_loss = torch.clamp(
                        sparse_weight * \
                            self.circle_loss_func(sparse_pos_scores, sparse_neg_scores, mp=circle_mp, mn=circle_mn, gamma_p=circle_gamma_p, gamma_n=circle_gamma_n),
                        max=100.0
                    )

                splade_reg = torch.sum(torch.mean(torch.abs(q_sparse_vectors), dim=0) ** 2) + \
                                        torch.sum(torch.mean(torch.abs(p_sparse_vectors), dim=0) ** 2) + \
                                        torch.sum(torch.mean(torch.abs(n_sparse_vectors), dim=0) ** 2)
                splade_reg = splade_reg_weight * splade_reg    
                loss = loss + sparse_loss + splade_reg

                extra_loss_dict['sparse_loss'] = sparse_loss.detach().cpu().item()
                extra_loss_dict['splade_reg'] = splade_reg.detach().cpu().item()

            # if 'colbert_vectors' in q_vectors_dict and 'colbert_vectors' in p_vectors_dict:
            #     if colbert_chunk_size > 0:
            #         # chunked colbert_scores
            #         bsz = q_vectors_dict['colbert_vectors'].shape[0]
            #         scores_list = []
            #         for i in range(0, bsz, colbert_chunk_size):
            #             end_i = min(i + colbert_chunk_size, bsz)
            #             chunk_vector = q_vectors_dict['colbert_vectors'][i: end_i]  # [chunk_size, seq_len, dim]
            #             chunk_mask = q_vectors_dict['attention_mask'][i: end_i]
            #             # Compute scores for this chunk
            #             # shape: chunk_size, seq_len, bsz, seq_len
            #             scores = torch.einsum('qin,pjn->qipj', chunk_vector, p_vectors_dict['colbert_vectors']).max(-1)[0].sum(1)
            #             scores = scores / chunk_mask[:, 1:].sum(-1, keepdim=True)
            #             scores_list.append(scores)
            #         colbert_scores = torch.cat(scores_list, dim=0)
            #         colbert_scores = colbert_scores / temperature
                
            #     colbert_loss = colbert_weight * F.cross_entropy(colbert_scores, targets, reduction='mean')
            #     colbert_loss = torch.clamp(colbert_loss, max=100.0)
            #     loss = loss + colbert_loss
            #     extra_loss_dict['colbert_loss'] = colbert_loss.detach().cpu().item()

        extra_loss_dict['total_loss'] = loss.detach().cpu().item()
        return loss, extra_loss_dict
