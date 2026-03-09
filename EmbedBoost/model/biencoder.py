import os
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Union, Callable, Any, List, Tuple, Optional
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from EmbedBoost.grad_cache.context_managers import RandContext
from EmbedBoost.common.file_util import batch_generator


logger = logging.getLogger(__name__)


class BGEM3Biencoder(nn.Module):
    def __init__(self, model_name_or_path: str, 
                 use_dense: bool = True, dense_pooling: str = 'cls', dense_dim: int = 512, dense_normalize=True,
                 use_sparse: bool = False, sparse_mode: str = 'splade_exp', sparse_normalize=False,
                 use_colbert: bool = False, colbert_dim: int = -1, colbert_normalize=True):
        super(BGEM3Biencoder, self).__init__()

        self.config = config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = config.vocab_size
        
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.dense_pooling = dense_pooling
        self.use_dense = use_dense
        self.dense_dim = dense_dim
        self.dense_normalize = dense_normalize
        self.use_sparse = use_sparse
        self.sparse_mode = sparse_mode
        self.sparse_normalize = sparse_normalize
        self.use_colbert = use_colbert
        self.colbert_dim = colbert_dim
        self.colbert_normalize = colbert_normalize
        
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
        
        res['input_ids'] = input_ids.clone()
        res['attention_mask'] = attention_mask.clone()

        if self.use_dense:
            dense_vectors = self.dense_embedding(last_hidden_state, attention_mask)
            if self.dense_normalize:
                dense_vectors = F.normalize(dense_vectors, dim=-1)
            res['dense_vectors'] = dense_vectors

        if self.use_sparse:
            sparse_vectors = self.sparse_embedding(last_hidden_state, input_ids)
            if self.sparse_normalize:
                sparse_vectors = F.normalize(sparse_vectors, dim=-1)
            res['sparse_vectors'] = sparse_vectors
        
        if self.use_sparse and return_sparse_weights:
            res['sparse_weights'] = self.sparse_weights(last_hidden_state, input_ids)
        
        if self.use_colbert:
            colbert_vectors = self.colbert_embedding(last_hidden_state, attention_mask)
            if self.colbert_normalize:
                colbert_vectors = F.normalize(colbert_vectors, dim=-1)
            res['colbert_vectors'] = colbert_vectors

        return res
    
    def forward(self, q_input_ids, q_attention_mask, q_token_type_ids,
                p_input_ids, p_attention_mask, p_token_type_ids,
                n_input_ids, n_attention_mask, n_token_type_ids):
        q_hidden_state = self.encoder(q_input_ids, q_attention_mask, q_token_type_ids).last_hidden_state
        q_encoded = self.cached_forward(q_hidden_state, q_input_ids, q_attention_mask)
        p_hidden_state = self.encoder(p_input_ids, p_attention_mask, p_token_type_ids).last_hidden_state
        p_encoded = self.cached_forward(p_hidden_state, p_input_ids, p_attention_mask)
        
        n_encoded = None
        if n_input_ids is not None:
            n_hidden_state = self.encoder(n_input_ids, n_attention_mask, n_token_type_ids).last_hidden_state
            n_encoded = self.cached_forward(n_hidden_state, n_input_ids, n_attention_mask)
        
        return q_encoded, p_encoded, n_encoded
    
    
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

    def encode(self, texts: List[str], max_length: int = 512, batch_size: int = 4000, return_sparse_weights=False) -> Dict[str, Union[np.ndarray, None]]:
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

        
        dense_vecs_list = []
        sparse_vecs_list = []
        sparse_weights_list = []
        colbert_vecs_list = []
        attention_mask_list = []
        for _, batch_texts in batch_generator(texts, batch_size):
            with torch.no_grad():
                input_d = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                ).to(device)
                input_ids, attention_mask, token_type_ids = input_d['input_ids'], input_d['attention_mask'], input_d['token_type_ids']
                q_hidden_state = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state
                q_encoded = self.cached_forward(q_hidden_state, input_ids, attention_mask, return_sparse_weights=return_sparse_weights)
                
                if 'dense_vectors' in q_encoded:
                    dense_vecs_list.append(q_encoded['dense_vectors'])
                if 'sparse_vectors' in q_encoded:
                    sparse_vecs_list.append(q_encoded['sparse_vectors'])
                if "sparse_weights" in q_encoded:
                    sparse_weights_list.extend(q_encoded['sparse_weights'])
                if "colbert_vectors" in q_encoded:
                    colbert_vecs_list.append(q_encoded['colbert_vectors'])
                    attention_mask_list.append(q_encoded['attention_mask'])
        
        ret_dict = {}
        if len(dense_vecs_list) > 0:
            if len(dense_vecs_list) == 1:
                dense_vectors = dense_vecs_list[0]
            elif len(dense_vecs_list) > 1:
                dense_vectors = torch.cat(dense_vecs_list, dim=0)
            ret_dict['dense_vectors'] = dense_vectors

        if len(sparse_vecs_list) > 0:
            if len(sparse_vecs_list) == 1:
                sparse_vectors = sparse_vecs_list[0]
            elif len(sparse_vecs_list) > 1:
                sparse_vectors = torch.cat(sparse_vecs_list, dim=0)
            ret_dict['sparse_vectors'] = sparse_vectors

        if sparse_weights_list:
            ret_dict['sparse_weights'] = sparse_weights_list
        
        if len(colbert_vecs_list) > 0:
            if len(colbert_vecs_list) == 1:
                colbert_vectors = colbert_vecs_list[0]
                attention_mask = attention_mask_list[0]
            elif len(colbert_vecs_list) > 1:
                colbert_vectors = torch.cat(colbert_vecs_list, dim=0)
                attention_mask = torch.cat(attention_mask_list, dim=0)
            ret_dict['colbert_vectors'] = colbert_vectors
            ret_dict['attention_mask'] = attention_mask

        return ret_dict


class BiEncoder:
    def __init__(
            self,
            q_model: nn.Module,
            p_model: nn.Module,
            loss_fn: Callable[..., Dict[str, Tensor]],
            get_rep_fn: Callable[..., Tensor] = None
    ):
        self.q_model = q_model
        self.p_model = p_model
        self.loss_fn = loss_fn
        self.get_rep_fn = get_rep_fn
    
    def __call__(self, q_inputs, p_inputs, n_inputs, model_kwargs, loss_kwargs):
        q_encoded = self.q_model(**q_inputs, **model_kwargs)
        p_encoded = self.p_model(**p_inputs, **model_kwargs)
        n_encoded = None
        if n_inputs is not None:
            n_encoded = self.p_model(**n_inputs, **model_kwargs)
        loss, loss_dict = self.loss_fn(q_encoded, p_encoded, n_encoded, **loss_kwargs)
        loss.backward()
        return loss, loss_dict


class BiEncoderWithGradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            q_model: nn.Module,
            p_model: nn.Module,
            chunk_size: int,
            loss_fn: Callable[..., Dict[str, Tensor]],
            split_input_fn: Callable[[Any, int], Any] = None,
            get_rep_fn: Callable[..., Tensor] = None,
            fp16: bool = False,
            scaler: GradScaler = None,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.q_model = q_model
        self.p_model = p_model
        self.q_encoder = q_model.encoder
        self.p_encoder = p_model.encoder
        self.chunk_size = chunk_size
        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, q_inputs, p_inputs, n_inputs, model_kwargs, loss_kwargs, no_sync_except_last=False):
        if no_sync_except_last:
            assert isinstance(self.q_model, nn.parallel.DistributedDataParallel)
            assert isinstance(self.p_model, nn.parallel.DistributedDataParallel)
            # assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
            #     'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
            #     'proper initializations.'
        
        q_inputs_list = self.split_inputs(q_inputs, self.chunk_size)
        p_inputs_list = self.split_inputs(p_inputs, self.chunk_size)
        if n_inputs is not None:
            n_inputs_list = self.split_inputs(n_inputs, self.chunk_size)
        
        # graph-less获取编码结果；
        q_reps, q_rnd_states = self.forward_no_grad(self.q_encoder, q_inputs_list, model_kwargs)
        p_reps, p_rnd_states = self.forward_no_grad(self.p_encoder, p_inputs_list, model_kwargs)
        n_reps, n_rnd_states = None, None
        if n_inputs is not None:
            n_reps, n_rnd_states = self.forward_no_grad(self.p_encoder, n_inputs_list, model_kwargs)

        # 计算reps -> loss这部分的反向传播梯度值
        # q_cache就是对应到q_reps的梯度值
        # 返回的loss是detach的
        q_cache, p_cache, n_cache, loss = self.build_cache(q_inputs, p_inputs, q_reps, p_reps, n_inputs, n_reps, **loss_kwargs)
        
        # split cache
        q_cache = q_cache.split(self.chunk_size)
        p_cache = p_cache.split(self.chunk_size)
        if n_cache is not None:
            n_cache = n_cache.split(self.chunk_size)
        
        # 然后对encoder进行重新的前向+反向传播
        self.forward_backward(self.q_encoder, q_inputs_list, q_cache, q_rnd_states, model_kwargs, no_sync_except_last=no_sync_except_last)
        self.forward_backward(self.p_encoder, p_inputs_list, p_cache, p_rnd_states, model_kwargs, no_sync_except_last=no_sync_except_last)
        if n_inputs is not None:
            self.forward_backward(self.p_encoder, n_inputs_list, n_cache, n_rnd_states, model_kwargs, no_sync_except_last=no_sync_except_last)

        return loss

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out.last_hidden_state

    def forward_no_grad(
            self,
            model: nn.Module,
            model_inputs,
            model_kwargs
        ):
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_inputs:
                input_tensors = self.get_input_tensors(x)
                rnd_states.append(RandContext(*input_tensors))
                y = model(*input_tensors, **model_kwargs)
                reps = self.get_reps(y)
                model_reps.append(reps)

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, q_inputs, p_inputs, q_reps: Tensor, p_reps: Tensor, n_inputs: Union[Dict[str, Tensor], None], n_reps: Union[Tensor, None], **loss_kwargs) -> Tuple[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        q_reps = q_reps.detach().requires_grad_()
        p_reps = p_reps.detach().requires_grad_()
        if n_reps is not None:
            n_reps = n_reps.detach().requires_grad_()
        
        q_encoded = self.q_model.cached_forward(q_reps, q_inputs['input_ids'], q_inputs['attention_mask'])
        p_encoded = self.p_model.cached_forward(p_reps, p_inputs['input_ids'], p_inputs['attention_mask'])
        if n_reps is not None:
            n_encoded = self.p_model.cached_forward(n_reps, n_inputs['input_ids'], n_inputs['attention_mask'])
        else:
            n_encoded = None
        
        with autocast() if self.fp16 else nullcontext():
            # 从这里出发，向前推导参数格式
            loss = self.loss_fn(q_encoded, p_encoded, n_encoded, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        q_cache = q_reps.grad
        p_cache = p_reps.grad
        n_cache = n_reps.grad if n_reps is not None else None
        return q_cache, p_cache, n_cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            model_kwargs,
            no_sync_except_last: bool = False
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = model(**x, **model_kwargs)
                
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                # maybe your embed model's parameters are not trainable
                if not surrogate.requires_grad and surrogate.grad_fn is None:
                    break
                surrogate.backward()


