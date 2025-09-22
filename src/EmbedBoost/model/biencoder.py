import logging
from typing import Dict, Union, Callable, Any, List, Tuple
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict


import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from EmbedBoost.grad_cache.context_managers import RandContext


logger = logging.getLogger(__name__)


class BiEncoder:
    def __init__(
            self,
            q_encoder: nn.Module,
            p_encoder: nn.Module,
            loss_fn: Callable[..., Dict[str, Tensor]],
            get_rep_fn: Callable[..., Tensor] = None
    ):
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.loss_fn = loss_fn
        self.get_rep_fn = get_rep_fn
    
    def __call__(self, q_inputs, p_inputs, model_kwargs, loss_kwargs):
        q_encoded = self.q_encoder(q_inputs, **model_kwargs)
        p_encoded = self.p_encoder(p_inputs, **model_kwargs)
        q_vectors = self.get_rep_fn(q_encoded)
        p_vectors = self.get_rep_fn(p_encoded)
        loss = self.loss_fn(q_vectors, p_vectors, **loss_kwargs)
        # if not isinstance(loss_dict, dict):
        #     raise TypeError("loss fn output type must be Dict[str, Tensor]")
        # if not 'loss' in loss_dict:
        #     raise KeyError("'loss' must be in loss dict")

        # backward一定要在这里做吗？
        loss.backward()
        return loss


class BiEncoderWithGradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            q_encoder: nn.Module,
            p_encoder: nn.Module,
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
        self.models = [q_encoder, p_encoder]
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder

        # self.chunk_sizes = [q_chunk_size, p_chunk_size]
        self.chunk_size = chunk_size

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, q_inputs, p_inputs, model_kwargs, loss_kwargs, no_sync_except_last=False):
        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'
        
        q_inputs = self.split_inputs(q_inputs, self.chunk_size)
        p_inputs = self.split_inputs(p_inputs, self.chunk_size)

        q_reps, q_rnd_states = self.forward_no_grad(self.q_encoder, q_inputs, model_kwargs)
        p_reps, p_rnd_states = self.forward_no_grad(self.p_encoder, p_inputs, model_kwargs)

        q_cache, p_cache, loss = self.build_cache(q_reps, p_reps, **loss_kwargs)
        
        # 关键步骤：split cache
        q_cache = q_cache.split(self.chunk_size)
        p_cache = p_cache.split(self.chunk_size)

        self.forward_backward(self.q_encoder, q_inputs, q_cache, q_rnd_states, model_kwargs, no_sync_except_last=no_sync_except_last)
        self.forward_backward(self.p_encoder, p_inputs, p_cache, p_rnd_states, model_kwargs, no_sync_except_last=no_sync_except_last)

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
            return model_out

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
                y = model(x, **model_kwargs)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, q_reps: Tensor, p_reps: Tensor, **loss_kwargs) -> Tuple[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        q_reps = q_reps.detach().requires_grad_()
        p_reps = p_reps.detach().requires_grad_()
        
        
        with autocast() if self.fp16 else nullcontext():
            # 从这里出发，向前推导参数格式
            loss = self.loss_fn(q_reps, p_reps, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        q_cache = q_reps.grad
        p_cache = p_reps.grad
        return q_cache, p_cache, loss.detach()

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

        # TODO: 写脚本证明，对于拆分的多个输入，梯度累加后和不拆分直接计算的梯度相同
        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = model(x, **model_kwargs)
                reps = self.get_reps(y)

                # 这里为什么要flatten
                surrogate = torch.dot(reps.flatten(), gradient.flatten())

                # 这个backward一定要在这里做么？
                surrogate.backward()