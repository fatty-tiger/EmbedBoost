import logging
import numpy as np
from typing import List, Dict, Any, Union

from EmbedBoost.abc.embedder import BaseEmbedder
from EmbedBoost.abc.vectordb_base import BaseVectorStore
from EmbedBoost.common.file_util import batch_generator
from EmbedBoost.common import rrf_util

from pymilvus import MilvusClient
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
)

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    def __init__(self, milvus_db_uri: str, col_name: str, dense_dim: int):
        self.milvus_db_uri = milvus_db_uri
        self.client = MilvusClient(milvus_db_uri)
        self.col_name = col_name
        self.dense_dim = dense_dim
    
    def create_collection(self, use_sparse=False, mode='overwrite') -> None:
        col_name = self.col_name
        
        if mode == 'overwrite':
            if self.client.has_collection(col_name):
                self.client.drop_collection(col_name)
                logger.info(f"Collection {col_name} dropped!")
        else:
            if self.client.has_collection(col_name):
                logger.info(f"Collection {col_name} exists!")
                return
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
        ]
        if use_sparse:
            fields.append(FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))
        schema = CollectionSchema(fields)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="IP"
        )
        if use_sparse:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )
        
        self.client.create_collection(
            collection_name=col_name,
            schema=schema,
            index_params=index_params
        )                
        logger.info(f"Collection {col_name} created.")
    
    def insert_docs(self, documents: List[Dict[str, Any]], embedder: BaseEmbedder, max_length: int, batch_size: int, mode: str = 'overwrite') -> Any:
        """
        Build vector index from texts using the provided embedder.
        
        Args:
            texts (List[str]): List of texts to index
            embedder (BaseEmbedder): Embedder model to use for encoding texts
            
        Returns:
            Any: Vector index (implementation-specific)
        """
        use_sparse = embedder.use_sparse
        self.create_collection(use_sparse=use_sparse, mode=mode)
        
        insert_count = 0
        for _, batch_docs in batch_generator(documents, batch_size=batch_size):
            batched_entities = []
            texts = [x['text'] for x in batch_docs]
            encoded = embedder.encode(texts, max_length=max_length)
            dense_vectors = encoded['dense_vectors'].tolist()
            for i, doc in enumerate(batch_docs):
                entity = {
                    "pk": doc['pk'],
                    "text": doc['text'],
                    "dense_vector": dense_vectors[i],
                }
                if embedder.use_sparse:
                    entity['sparse_vector'] = encoded['sparse_weights'][i]
                batched_entities.append(entity)

            self.client.insert(collection_name=self.col_name, data=batched_entities)
            insert_count += len(batched_entities)
            logger.info(f"{insert_count} entities inserted.")

    def dense_retrieval(self, dense_vectors: np.ndarray, topk: int) -> List[Dict[str, Union[str, float]]]:
        data = dense_vectors.tolist()
        res = self.client.search(
            collection_name=self.col_name,
            data=data,
            anns_field="dense_vector",
            search_params={"metric_type": "IP"},
            limit=topk,
            output_fields=["pk", "text"]
        )
        results = []
        for hits in res:
            result = []
            for hit in hits:
                item = {}
                item['id'] = hit['id']
                item['score'] = hit['distance']
                item.update(hit['entity'])
                result.append(item)
            results.append(result)
        return results

    def sparse_retrieval(self, sparse_vectors: np.ndarray, topk: int) -> List[Dict[str, Union[str, float]]]:
        res = self.client.search(
            collection_name=self.col_name,
            data=[sparse_vectors],
            anns_field="sparse_vector",
            limit=topk,
            output_fields=["pk", "text"],
            search_params={"metric_type": "IP"}
        )
        results = []
        for hits in res:
            result = []
            for hit in hits:
                item = {}
                item['id'] = hit['id']
                item['score'] = hit['distance']
                item.update(hit['entity'])
                result.append(item)
            results.append(result)
        return results