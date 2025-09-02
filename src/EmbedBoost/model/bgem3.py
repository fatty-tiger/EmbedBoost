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
import numpy as np

from typing import List, Dict, Union
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer

from EmbedBoost.abc.embedder import BaseEmbedder
from EmbedBoost.common.file_util import batch_generator


logger = logging.getLogger(__name__)


class BGEM3Embedder(BaseEmbedder, nn.Module):
    def __init__(self, model_name_or_path: str, pooling_method: str, max_length: int, dense_dim: int,
                 use_sparse: bool = True, colbert_dim: int = None):
        super(BGEM3Embedder, self).__init__()
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = config.vocab_size
        
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.pooling_method = pooling_method
        self.dense_dim = dense_dim
        self.use_sparse = use_sparse
        self.colbert_dim = colbert_dim
        self.use_colbert = True if colbert_dim else False
        self.normalize_embeddings = True
        self.max_length = max_length

        self.dense_linear = torch.nn.Linear(
            config.hidden_size,
            dense_dim if dense_dim > 0 else config.hidden_size
        )
        dense_state_fpath = os.path.join(model_name_or_path, 'dense_linear.pt')
        if os.path.exists(dense_state_fpath):    
            dense_state_dict = torch.load(dense_state_fpath, map_location='cpu', weights_only=True)
            self.dense_linear.load_state_dict(dense_state_dict)
            logger.info("dense linear checkpoint loaded.")

        if use_sparse:
            self.sparse_linear = torch.nn.Linear(
                in_features=config.hidden_size,
                out_features=1
            )
            sparse_state_fpath = os.path.join(model_name_or_path, 'sparse_linear.pt')
            if os.path.exists(sparse_state_fpath):    
                sparse_state_dict = torch.load(sparse_state_fpath, map_location='cpu', weights_only=True)
                self.sparse_linear.load_state_dict(sparse_state_dict)
                logger.info("sparse linear checkpoint loaded.")
            else:
                logger.warn("sparse linear checkpoint not found.")

    def _dense_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the dense embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The dense embeddings.
        """
        if self.pooling_method == "cls":
            logits = last_hidden_state[:, 0]
        elif self.pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            logits = s / d
        elif self.pooling_method == "last_token":
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
            raise NotImplementedError(f"pooling method {self.pooling_method} not implemented")
        logits = self.dense_linear(logits)
        if self.normalize_embeddings:
            logits = F.normalize(logits, dim=-1)
        return logits

    def _sparse_weights(self, last_hidden_state, input_ids):
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

        token_weights = torch.relu(self.sparse_linear(last_hidden_state))
        values = token_weights.squeeze(-1).cpu().numpy()
        ids = input_ids.cpu().numpy()
        token_weights_list = []
        for i in range(input_ids.size(0)):
            vals = values[i].tolist()
            token_weights_list.append({id: vals[j] for j, id in enumerate(ids[i].tolist()) if id not in unused_tokens})
        return token_weights_list
    
    def _sparse_embedding(self, last_hidden_state, input_ids):
        """Compute and return the sparse embedding.

        Args:
            hidden_state (torch.Tensor): The model output's last hidden state.
            input_ids (_type_): Ids from input features.

        Returns:
            torch.Tensor: The sparse embedding or just the token weights.
        """
        token_weights = torch.relu(self.sparse_linear(last_hidden_state))
        sparse_embedding = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device
        )
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [
            self.tokenizer.cls_token_id, 
            self.tokenizer.mask_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, return_sparse_vectors=True, return_sparse_weights=False):
        model_out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        dense_vectors = self._dense_embedding(model_out.last_hidden_state, attention_mask)
        
        res = {
            'dense_vectors': dense_vectors
        }
        if self.use_sparse and return_sparse_vectors:
            res['sparse_vectors'] = self._sparse_embedding(model_out.last_hidden_state, input_ids)
        
        if self.use_sparse and return_sparse_weights:
            res['sparse_weights'] = self._sparse_weights(model_out.last_hidden_state, input_ids)
        
        return res
    
    def encode(self, texts: List[str], max_length: int = -1, batch_size: int = 4000) -> Dict[str, Union[np.ndarray, None]]:
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
            return {"dense_vectors": None, "sparse_vectors": None}
        
        device = next(self.parameters()).device

        if max_length == -1:
            max_length = self.max_length

        dense_vecs_list = []
        sparse_weights_list = []
        for _, batch_texts in batch_generator(texts, batch_size):
            # Get embeddings
            with torch.no_grad():
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                ).to(device)

                res = self.forward(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    return_sparse_vectors=False,
                    return_sparse_weights=True
                )
                dense_vecs_list.append(res['dense_vectors'])
                if "sparse_weights" in res:
                    sparse_weights_list.extend(res['sparse_weights'])
        
        if len(dense_vecs_list) > 1:
            dense_vectors = torch.cat(dense_vecs_list, dim=0)
        else:
            dense_vectors = dense_vecs_list[0]

        return {
            "dense_vectors": dense_vectors.cpu().numpy(),
            "sparse_weights": sparse_weights_list
        }
    
    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.bert.save_pretrained(output_dir, state_dict=_trans_state_dict(self.bert.state_dict()))
        logger.info(f"bert model saved to {output_dir}")

        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"tokenizer saved to {output_dir}")
        
        torch.save(_trans_state_dict(self.dense_linear.state_dict()),
                   os.path.join(output_dir, 'dense_linear.pt'))
        logger.info(f"dense linear saved to {os.path.join(output_dir, 'dense_linear.pt')}")
        
        if self.use_sparse:
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()),
                       os.path.join(output_dir, 'sparse_linear.pt'))
            logger.info(f"sparse linear saved to {os.path.join(output_dir, 'sparse_linear.pt')}")
