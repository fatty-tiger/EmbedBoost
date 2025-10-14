import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import time
import json
import mteb
import torch
from mteb.encoder_interface import PromptType
from mteb import TaskResult
from datasets import load_dataset
from FlagEmbedding import FlagModel
from EmbedBoost.model.bgem3 import BGEM3Embedder

import numpy as np


class BgeM3Model:
    def __init__(self) -> None:
        self.model = FlagModel('/data/pretrained_models/BAAI/bge-base-zh-v1.5', devices=['cuda:5'])
        
    def encode(
        self,
        sentences: list[str],
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
        model_name_or_path = "/data/jiangjie/FattyEmbedding/output/multicpr/ecom/v1/models/m1/ckp_epoch_20_step_180"
        # model_name_or_path = "/data/jiangjie/FattyEmbedding/output/multicpr/ecom/v1/models/m1/ckp_epoch_10_step_90"
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

    def encode(self, sentences, batch_size=32, **kwargs) -> np.ndarray:
        """Encodes the given sentences using the encoder.
        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        res = self.embedder.encode(sentences, batch_size=batch_size)
        return res['dense_vectors']


def main():
    # evaluating the model:
    model = CustomModel()
    # model = BgeM3Model()
    #tasks = mteb.get_tasks(tasks=["EcomRetrieval"])
    #tasks = mteb.get_tasks(tasks=["MedicalRetrieval"])
    tasks = mteb.get_tasks(tasks=["VideoRetrieval"])

    evaluator = mteb.MTEB(tasks=tasks)
    ts = int(time.time())
    encode_kwargs = {
        "batch_size": 200
    }
    results = evaluator.run(model, verbosity=2, output_folder=f"output/mteb_ts_{ts}", encode_kwargs=encode_kwargs)
    for res in results:
    #     print(res.task_name)
        print(json.dumps(res.scores, ensure_ascii=False, indent=4))


def test():
    from FlagEmbedding import FlagModel
    sentences_1 = ["样例数据-1", "样例数据-2"]
    sentences_2 = ["样例数据-3", "样例数据-4"]
    model = FlagModel('/data/pretrained_models/BAAI/bge-base-zh-v1.5', devices=['cuda:5'])
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)
    similarity = embeddings_1 @ embeddings_2.T
    print(embeddings_1)
    print(type(embeddings_1))
    print(similarity)

def show_data_lines():
    dataset = load_dataset("mteb/DuRetrieval")
    dev = dataset["dev"]
    print(type(dev))



if __name__ == "__main__":
    # main()
    # test()
    show_data_lines()