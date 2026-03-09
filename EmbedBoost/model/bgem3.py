"""
BGE-M3 Embedder. 
Basically copied from https://github.com/FlagOpen/FlagEmbedding
Colbert part removed
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from typing import List, Dict, Union, Optional
from torch import Tensor
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm

from EmbedBoost.abc.embedder import BaseEmbedder
from EmbedBoost.common.file_util import batch_generator


logger = logging.getLogger(__name__)


class BGEM3Embedder(BaseEmbedder, nn.Module):
    def __init__(self, model_name_or_path: str, use_dense: bool = True, dense_pooling: str = 'cls', dense_dim: int = 512, infer_dense_dim: int = -1, 
                 use_sparse: bool = False, sparse_mode: str = 'splade_exp',
                 # sparse_agg: str = 'max', sparse_topk: int = 0, sparse_reg: str = 'none',
                 use_colbert: bool = False, colbert_dim: int = -1,
                 use_mrl: bool = False, mrl_dims: Optional[List[int]] = None):
        super(BGEM3Embedder, self).__init__()

        self.config = config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = config.vocab_size
        
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.dense_pooling = dense_pooling
        self.use_dense = use_dense
        self.dense_dim = dense_dim
        self.infer_dense_dim = infer_dense_dim if infer_dense_dim > 0 else dense_dim
        self.use_sparse = use_sparse
        self.sparse_mode = sparse_mode
        # self.sparse_agg = sparse_agg
        # self.splade_topk
        # self.splade_reg: none、l1、flops
        self.use_colbert = use_colbert
        self.colbert_dim = colbert_dim
        self.use_mrl = use_mrl
        self.mrl_dims = mrl_dims
        if mrl_dims is not None:
            if mrl_dims == 'AUTO':
                mrl_dims = [dense_dim]
            assert isinstance(mrl_dims, list)
            assert len(mrl_dims) > 0
            if dense_dim != mrl_dims[-1]:
                logger.warn(f"dense dim {dense_dim} does not match mrl dim {mrl_dims[-1]}, resetting dense dim to mrl dim.")
                self.dense_dim = mrl_dims[-1]
        logger.warn(f"use_mrl: {use_mrl}, mrl_dims: {mrl_dims}, infer_dense_dim: {infer_dense_dim}")
        
        # assert use_dense or use_sparse

        if use_dense:
            self.dense_linear = torch.nn.Linear(
                config.hidden_size,
                dense_dim if dense_dim > 0 else config.hidden_size
            )
            dense_state_fpath = os.path.join(model_name_or_path, 'dense_linear.pt')
            if os.path.exists(dense_state_fpath):    
                dense_state_dict = torch.load(dense_state_fpath, map_location='cpu', weights_only=True)
                dense_state_dict = {k.replace('dense.', ''): v for k, v in dense_state_dict.items()} # adhoc
                self.dense_linear.load_state_dict(dense_state_dict)
                logger.info("dense linear checkpoint loaded.")
            else:
                logger.warn("dense linear weights were not found, random initialized.")

        if use_sparse:
            if 'splade' in sparse_mode:
                from transformers.models.bert.modeling_bert import BertOnlyMLMHead
                self.sparse_linear = BertOnlyMLMHead(config)
                # self.sparse_linear = torch.nn.Linear(
                #     in_features=config.hidden_size,
                #     out_features=config.vocab_size
                # )
            else:
                self.sparse_linear = torch.nn.Linear(
                    in_features=config.hidden_size,
                    out_features=1
                )
            
            sparse_state_fpath = os.path.join(model_name_or_path, 'sparse_linear.pt')
            if os.path.exists(sparse_state_fpath):    
                sparse_state_dict = torch.load(sparse_state_fpath, map_location='cpu', weights_only=True)
                self.sparse_linear.load_state_dict(sparse_state_dict, strict=True)
                logger.info("sparse linear weights loaded.")
            else:
                logger.warn("sparse linear weights were not found, random initialized.")
            
            self.sparse_unused_tokens = torch.tensor([
                self.tokenizer.cls_token_id, 
                self.tokenizer.mask_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.sep_token_id,
            ])
        
        if use_colbert:
            self.colbert_linear = torch.nn.Linear(
                in_features=config.hidden_size,
                out_features=config.hidden_size if colbert_dim <= 0 else colbert_dim
            )
            colbert_state_fpath = os.path.join(model_name_or_path, 'colbert_linear.pt')
            if os.path.exists(colbert_state_fpath):    
                colbert_state_dict = torch.load(colbert_state_fpath, map_location='cpu', weights_only=True)
                self.colbert_linear.load_state_dict(colbert_state_dict)
                logger.info("colbert linear weights loaded.")
            else:
                logger.warn("colbert linear weights were not found, random initialized.")

    def dense_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the dense embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            List[torch.Tensor]: The dense embeddings.
        """
        if self.dense_pooling == "cls":
            logits = last_hidden_state[:, 0]
        elif self.dense_pooling == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            logits = s / d
        elif self.dense_pooling == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                logits = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                logits = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {self.dense_pooling} not implemented")
        
        logits = self.dense_linear(logits)
        return logits        
    
    def sparse_embedding(self, last_hidden_state, input_ids, to_coo=False):
        if 'splade' in self.sparse_mode:
            # use cls
            sparse_logits = self.sparse_linear(last_hidden_state[:, 0])
            # (bsz, vocab_size)
            # use log to avoid explosion
            sparse_logits = torch.log(1 + torch.relu(sparse_logits))
            # 输出侧，非正常token不计算。
            mask = torch.ones(sparse_logits.size(), device=input_ids.device)
            mask[:, [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.mask_token_id, self.tokenizer.sep_token_id, self.tokenizer.unk_token_id]] = 0.0
            sparse_logits = sparse_logits * mask
            return sparse_logits
            # mask = (
            #     (input_ids != self.tokenizer.cls_token_id) & \
            #     (input_ids != self.tokenizer.mask_token_id) & \
            #     (input_ids != self.tokenizer.pad_token_id) & \
            #     (input_ids != self.tokenizer.unk_token_id) & \
            #     (input_ids != self.tokenizer.sep_token_id)
            # ).long().unsqueeze(-1)
            # sparse_logits = self.sparse_linear(last_hidden_state) # shape of output tensor after this step (bs, seq_len, vocab_size)
            # # 此处的relu有一点问题，因为mlmhead会导致大部分的预测结果为负值，尤其是那个位置没有mask的情况
            # sparse_logits = torch.log(1 + torch.relu(sparse_logits)) * mask # shape of output tensor after this step (bs, seq_len, vocab_size)
            # sparse_logits = torch.max(sparse_logits, dim=1).values # shape of output tensor after this step (bs, vocab_size)
            # sparse_logits = F.normalize(sparse_logits, dim=1)
            # return sparse_logits
        else:
            mask = (
                (input_ids != self.tokenizer.cls_token_id) & \
                (input_ids != self.tokenizer.mask_token_id) & \
                (input_ids != self.tokenizer.pad_token_id) & \
                (input_ids != self.tokenizer.unk_token_id) & \
                (input_ids != self.tokenizer.sep_token_id)
            ).nonzero(as_tuple=True)
            sparse_logits = torch.relu(self.sparse_linear(last_hidden_state))
            sparse_logits = sparse_logits.squeeze(-1)
            indices = torch.stack([mask[0], input_ids[mask]], dim=0)
            values = sparse_logits[mask]
            sparse_vectors = torch.sparse_coo_tensor(indices, values, size=(input_ids.shape[0], self.vocab_size))
            return sparse_vectors  


    def sparse_weights(self, last_hidden_state, input_ids):
        """Compute and return the sparse embedding.
        TODO: 复用sparse_embedding的逻辑
        Args:
            hidden_state (torch.Tensor): The model output's last hidden state.
            input_ids (_type_): Ids from input features.
            return_embedding (bool, optional): If True, return the computed embedding, otherwise just return the token weights. 
                Defaults to ``True``.

        Returns:
            torch.Tensor: The sparse embedding or just the token weights.
        """
        unused_tokens = [
            self.tokenizer.unk_token_id,
            self.tokenizer.pad_token_id
        ]
        if hasattr(self.tokenizer, 'cls_token_id'):
            unused_tokens.append(self.tokenizer.cls_token_id)
        if hasattr(self.tokenizer, 'mask_token_id'):
            unused_tokens.append(self.tokenizer.mask_token_id)
        if hasattr(self.tokenizer, 'bos_token_id'):
            unused_tokens.append(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, 'eos_token_id'):
            unused_tokens.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'sep_token_id'):
            unused_tokens.append(self.tokenizer.sep_token_id)

        if 'splade' in self.sparse_mode:
            sparse_vectors = self.sparse_embedding(last_hidden_state, input_ids)
            ids = input_ids.cpu().numpy()
            sparse_weights_list = []
            for i in range(input_ids.size(0)):
                sparse_tuple = [(_id, val) for _id, val in enumerate(sparse_vectors[i].tolist()) if val >= 1e-2]
                sparse_weights = {_id: val for _id, val in sorted(sparse_tuple, key=lambda x: x[1], reverse=True)[:30]}
                sparse_weights_list.append(sparse_weights)
            return sparse_weights_list
        else:
            token_weights = torch.relu(self.sparse_linear(last_hidden_state))
            values = token_weights.squeeze(-1).cpu().numpy()
            ids = input_ids.cpu().numpy()
            sparse_weights_list = []
            for i in range(input_ids.size(0)):
                vals = values[i].tolist()
                sparse_weights_list.append({id: vals[j] for j, id in enumerate(ids[i].tolist()) if id not in unused_tokens})
            return sparse_weights_list

    def colbert_embedding(self, last_hidden_state, mask):
        """Get the colbert vectors.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.
        Returns:
            torch.Tensor: The colbert vectors.
        """
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs
    
    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    
    def cached_forward(self, last_hidden_state, input_ids, attention_mask, return_sparse_weights=False):
        res = dict()
        
        input_ids_clone = input_ids.clone()
        attention_mask_clone = attention_mask.clone()
        # res['input_ids'] = input_ids.clone()
        # res['attention_mask'] = attention_mask.clone()

        if self.use_dense:
            dense_vectors = self.dense_embedding(last_hidden_state, attention_mask_clone)
            dense_vectors = F.normalize(dense_vectors, dim=-1)
            res['dense_vectors'] = dense_vectors

        if self.use_sparse:
            # sparse_vectors = self.sparse_embedding(last_hidden_state, input_ids)
            res['sparse_vectors'] = self.sparse_embedding(last_hidden_state, input_ids_clone)
        
        if self.use_sparse and return_sparse_weights:
            res['sparse_weights'] = self.sparse_weights(last_hidden_state, input_ids_clone)
        
        if self.use_colbert:
            colbert_vectors = self.colbert_embedding(last_hidden_state, attention_mask)
            colbert_vectors = F.normalize(colbert_vectors, dim=-1)
            res['colbert_vectors'] = colbert_vectors

        return res
    
    def forward(self, input_ids, attention_mask, token_type_ids, return_sparse_weights=False):
        # input_ids = bert_inputs['input_ids']
        # attention_mask = bert_inputs.get('attention_mask', None)
        # token_type_ids = bert_inputs.get('token_type_ids', None)
        model_out = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = model_out.last_hidden_state
        res = self.cached_forward(last_hidden_state, input_ids, attention_mask, return_sparse_weights=return_sparse_weights)
        return res

    def _get_queries_attention_mask(self, queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]):
        """padding attention mask for colbert

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]]): Input queries.

        Returns:
            torch.Tensor: The query attention mask.
        """
        if not isinstance(queries, list):
            q_mask = queries['attention_mask']
        else:
            q_mask_list = [sub_features['attention_mask'] for sub_features in queries]
            _length = max([mask.shape[1] for mask in q_mask_list])
            if self.tokenizer.padding_side == 'right':
                q_mask = torch.cat([
                    F.pad(mask, (0, _length - mask.shape[1]), value=0)
                    for mask in q_mask_list
                ], dim=0)
            else:
                q_mask = torch.cat([
                    F.pad(mask, (_length - mask.shape[1], 0), value=0)
                    for mask in q_mask_list
                ], dim=0)
        return q_mask

    def encode(self, texts: List[str], max_length: int = 512, batch_size: int = 4000) -> Dict[str, Union[np.ndarray, None]]:
        """
        Encode a list of text strings into embeddings.
        
        Args:
            texts (List[str]): A list of text strings to encode.
            
        Returns:
            Dict[str, Union[np.ndarray, None]]: A dictionary containing the embeddings.
                - "dense_vectors": Dense embeddings as numpy array
                - "sparse_vectors": Sparse embeddings as numpy array
        """

        if len(texts) == 0:
            return None
        
        device = next(self.parameters()).device

        if max_length == -1:
            max_length = self.max_length

        attention_mask_list = []
        dense_vecs_list = []
        sparse_vecs_list = []
        sparse_weights_list = []
        colbert_vecs_list = []
        for _, batch_texts in batch_generator(texts, batch_size):
            with torch.no_grad():
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                ).to(device)

                res = self.forward(
                    **encoded,
                    return_sparse_weights=True
                )
                attention_mask_list.append(encoded['attention_mask'])
                if 'dense_vectors' in res:
                    dense_vecs_list.append(res['dense_vectors'])
                if 'sparse_vectors' in res:
                    sparse_vecs_list.append(res['sparse_vectors'])
                if "sparse_weights" in res:
                    sparse_weights_list.extend(res['sparse_weights'])
                if "colbert_vectors" in res:
                    colbert_vecs_list.append(res['colbert_vectors'])
        
        ret_dict = {}
        
        #if len(attention_mask_list) == 1:
        #    attention_mask = attention_mask_list[0]
        #elif len(attention_mask_list) > 1:
        #    # 这里，每个attention_mask的shape不一样，无法concat
        #    attention_mask = torch.cat(attention_mask_list, dim=0)
        #ret_dict['attention_mask'] = attention_mask

        if len(dense_vecs_list) > 0:
            if len(dense_vecs_list) == 1:
                dense_vectors = dense_vecs_list[0]
            elif len(dense_vecs_list) > 1:
                dense_vectors = torch.cat(dense_vecs_list, dim=0)
            # dense_vectors = dense_vectors.cpu().numpy()
            # if self.infer_dense_dim != self.dense_dim:
            #     dense_vectors = dense_vectors[:, :self.infer_dense_dim]
            ret_dict['dense_vectors'] = dense_vectors

        if len(sparse_vecs_list) > 0:
            if len(sparse_vecs_list) == 1:
                sparse_vectors = sparse_vecs_list[0]
            elif len(sparse_vecs_list) > 1:
                sparse_vectors = torch.cat(sparse_vecs_list, dim=0)
            # sparse_vectors = sparse_vectors.cpu().numpy()
            ret_dict['sparse_vectors'] = sparse_vectors

        if sparse_weights_list:
            ret_dict['sparse_weights'] = sparse_weights_list
        
        if len(colbert_vecs_list) > 0:
            if len(colbert_vecs_list) == 1:
                colbert_vectors = colbert_vecs_list[0]
            elif len(colbert_vecs_list) > 1:
                colbert_vectors = torch.cat(colbert_vecs_list, dim=0)
            # colbert_vectors = colbert_vectors.cpu().numpy()
            # if do_normalize:
            #     colbert_vectors = normalize_vectors(colbert_vectors)
            ret_dict['colbert_vectors'] = colbert_vectors

        return ret_dict
    
    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.encoder.save_pretrained(output_dir, state_dict=_trans_state_dict(self.encoder.state_dict()))
        logger.info(f"bert model saved to {output_dir}")

        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"tokenizer saved to {output_dir}")
        
        if self.use_dense:
            torch.save(_trans_state_dict(self.dense_linear.state_dict()),
                       os.path.join(output_dir, 'dense_linear.pt'))
            logger.info(f"dense linear saved to {os.path.join(output_dir, 'dense_linear.pt')}")
        
        if self.use_sparse:
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()),
                       os.path.join(output_dir, 'sparse_linear.pt'))
            logger.info(f"sparse linear saved to {os.path.join(output_dir, 'sparse_linear.pt')}")

        if self.use_colbert:
            torch.save(_trans_state_dict(self.colbert_linear.state_dict()),
                       os.path.join(output_dir, 'colbert_linear.pt'))
            logger.info(f"colbert_linear linear saved to {os.path.join(output_dir, 'colbert_linear.pt')}")


class BGEM3EmbedderForInference(BGEM3Embedder):
    def __init__(self, *args, **kwargs):
        super(BGEM3EmbedderForInference, self).__init__( *args, **kwargs)
    
    def sparse_weights(self, last_hidden_state):
        """Compute and return the sparse embedding.

        Args:
            hidden_state (torch.Tensor): The model output's last hidden state.
            input_ids (_type_): Ids from input features.
            return_embedding (bool, optional): If True, return the computed embedding, otherwise just return the token weights. 
                Defaults to ``True``.

        Returns:
            torch.Tensor: The sparse embedding or just the token weights.
        """
        unused_tokens = [
            self.tokenizer.unk_token_id,
            self.tokenizer.pad_token_id
        ]
        if hasattr(self.tokenizer, 'cls_token_id'):
            unused_tokens.append(self.tokenizer.cls_token_id)
        if hasattr(self.tokenizer, 'mask_token_id'):
            unused_tokens.append(self.tokenizer.mask_token_id)
        if hasattr(self.tokenizer, 'bos_token_id'):
            unused_tokens.append(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, 'eos_token_id'):
            unused_tokens.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'sep_token_id'):
            unused_tokens.append(self.tokenizer.sep_token_id)

        token_weights = torch.relu(self.sparse_linear(last_hidden_state)).squeeze(-1)
        return token_weights

    def forward(self, input_ids, attention_mask, token_type_ids):
        model_out = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = model_out.last_hidden_state
        dense_vectors = self.dense_embedding(last_hidden_state, attention_mask)
        dense_vectors = F.normalize(dense_vectors, dim=-1)
        
        if self.use_sparse:
            sparse_vectors = self.sparse_weights(last_hidden_state)
        else:
            sparse_vectors = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
        
        # input_ids = input_ids.cpu().numpy()
        # dense_vectors = dense_vectors.cpu().numpy()
        return input_ids, dense_vectors, sparse_vectors


class OnnxInferer(BaseEmbedder):
    def __init__(self, model_id_or_path, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.device = torch.device(device)
        onnx_fpath = os.path.join(model_id_or_path, "model.onnx")

        import onnxruntime
        ort_sessionopts = onnxruntime.SessionOptions()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        logging.info(f"loading onnx mmodel from {onnx_fpath}")
        self.ort_session = onnxruntime.InferenceSession(onnx_fpath, 
                                                        sess_options=ort_sessionopts,
                                                        providers=providers)
        logging.info("onnx inference session created")

    def encode(self, query_list, batch_size=128, max_length=128):
        ret_dict = {}

        input_ids_list = []
        dense_vector_list = []
        sparse_vector_list = []
        total = int(math.ceil(len(query_list) / batch_size))
        for idx, batch_texts in tqdm(batch_generator(query_list, batch_size), total=total):
            encoded = self.tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='np')
            onnx_inputs = {k: v for k, v in encoded.items()}
            onnx_outputs = self.ort_session.run(None, onnx_inputs)
            input_ids_list.append(onnx_outputs[0])
            dense_vector_list.append(onnx_outputs[1])
            sparse_vector_list.append(onnx_outputs[2])
        ret_dict['token_ids'] = np.concatenate(input_ids_list, axis=0)
        ret_dict['dense_vectors'] = np.concatenate(dense_vector_list, axis=0)
        ret_dict['sparse_vectors'] = np.concatenate(sparse_vector_list, axis=0)
        
        sparse_weights_list = []
        all_ids = ret_dict['token_ids'].tolist()
        all_values = ret_dict['sparse_vectors'].tolist()
        for ids, values in zip(all_ids, all_values):
            sparse_weights = {_id: val for _id, val in zip(ids, values) if _id not in [1, 2, 0]}
            sparse_weights_list.append(sparse_weights)
        ret_dict['sparse_weights'] = sparse_weights_list
        return ret_dict
    
