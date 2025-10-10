import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import mteb
import torch
from typing import List
from mteb.encoder_interface import PromptType
from mteb import TaskResult
from FlagEmbedding import FlagModel
from EmbedBoost.model.bgem3 import BGEM3Embedder

import numpy as np


class BgeM3Model:
    def __init__(self) -> None:
        self.model = FlagModel('BAAI/bge-base-zh-v1.5', devices=['cuda:5'])
        
    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        return self.model.encode(sentences)


class CustomModel:
    # model_card_data = {
    #     "model_name": "ernie_3_medium_zh"
    # }

    def __init__(self) -> None:
        #model_name_or_path = "/data/jiangjie/FattyEmbedding/output/multicpr/ecom/v1/models/m1/ckp_epoch_20_step_180"
        model_name_or_path = "/data/jiangjie/FattyEmbedding/output/multicpr/ecom/v1/models/m1/ckp_epoch_10_step_90"
        use_dense = True
        dense_pooling = "cls"
        dense_dim = 96
        use_sparse = False
        device = torch.device("cuda:5")

        self.embedder = embedder = BGEM3Embedder(
            model_name_or_path,
            use_dense,
            dense_pooling,
            dense_dim, 
            use_sparse
        )
        embedder.to(device)
        embedder.eval()

    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        res = self.embedder.encode(sentences)
        return res['dense_vectors']




def main():
    model = CustomModel()
    # model = BgeM3Model()
    tasks = mteb.get_tasks(tasks=["EcomRetrieval"])
    evaluator = mteb.MTEB(tasks=tasks)
    results = evaluator.run(model, verbosity=2, eval_splits=['dev'], output_folder="output/mteb")
    for res in results:
    #     print(res.task_name)
        print(json.dumps(res.scores, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
