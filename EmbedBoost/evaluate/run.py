"""
Example usage of the embedding model evaluation framework.
"""
import os
import argparse
import sys
import logging
import json
import copy

from collections import defaultdict
from typing import List, Dict, Union, Any

from EmbedBoost.abc.embedder import BaseEmbedder
from EmbedBoost.model.bgem3 import BGEM3Embedder
from EmbedBoost.abc.vectordb_base import BaseVectorStore
from EmbedBoost.vectordb.milvus_db import MilvusVectorStore
from EmbedBoost.evaluate import evaluate_metric
from EmbedBoost.common.tensor_util import normalize_vectors


LOG_DATE_FMT = '%Y‐%m‐%d %H:%M:%S'
LOG_FMT = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stderr,
                    datefmt=LOG_DATE_FMT,
                    format=LOG_FMT)
logger = logging.getLogger(__name__)

BASE_DATA_DIR = "/home/jiangjie/zkh_search_r1/FattyEmbedding/data"


def rrf_ranking(sorted_lists, topk, rank_constant=60):
    score_dict = defaultdict(float)
    for lst in sorted_lists:
        for rank, item in enumerate(lst, start=1):
            score_dict[item['id']] += 1.0 / (rank_constant + rank)

    # 按照RRF得分降序排序
    ranked_items = [{'id': x[0], 'score': x[1]} for x in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)][:topk]
    return ranked_items


def retrieve(query_list: List[Dict[str, Any]], 
             embedder: BaseEmbedder,
             vector_store: BaseVectorStore,
             retrieval_modes: Union[List[str], str],
             merge_strategy: str = None,
             max_query_length: int = 128,
             topk: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform retrieval task.
    
    Args:
        query_list: List of queries with related documents
            [{"query": "query content here", "related_docs": [{"id": "id1", "text": "text1"}]}, ...]
        topk (int): Number of top results to retrieve
        
    Returns:
        result_list: Retrieval results with retrieved documents
            [{"query": "query content here", 
                "related_docs": [{"id": "id1", "text": "text1"}], 
                "retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]}]
    """
    result_list = []
    
    # Extract queries
    queries = [item["query"] for item in query_list]
    
    # Encode queries
    query_encoded = embedder.encode(queries, max_length=max_query_length, batch_size=500)

    if isinstance(retrieval_modes, str):
        retrieval_modes = [retrieval_modes]
    
    # Perform retrieval for each query
    for i, query_item in enumerate(query_list):
        result_item = copy.deepcopy(query_item)

        results_to_merge = []
        if "dense" in retrieval_modes:
            query_dense_vector = query_encoded["dense_vectors"][i:i+1, :]
            dense_retrieved_docs = vector_store.dense_retrieval(query_dense_vector, topk)
            result_item['dense_retrieved_docs'] = dense_retrieved_docs[0]
            if merge_strategy:
                results_to_merge.append(dense_retrieved_docs[0])
        
        if "sparse" in retrieval_modes:
            query_sparse_vector = query_encoded["sparse_weights"][i]
            sparse_retrieved_docs = vector_store.sparse_retrieval(query_sparse_vector, topk)
            result_item['sparse_retrieved_docs'] = sparse_retrieved_docs[0]
            if merge_strategy:
                results_to_merge.append(sparse_retrieved_docs[0])
        
        if merge_strategy and len(results_to_merge) > 1:
            if merge_strategy == 'rrf':
                result_item['merged_docs'] = rrf_ranking(results_to_merge, topk)
            else:
                raise ValueError(f"Invalid merge strategy: {merge_strategy}")

        result_list.append(result_item)

        if (i+1) % 50 == 0:
            logging.info(f"Processed {i+1} queries.")
    
    return result_list


def get_eval_dataset(dataset_name):
    if dataset_name == 'multicpr_med':
        from EmbedBoost.evaluate.multicpr_dataset import MultiCprRetrievalDataset
        query_fpath = os.path.join(BASE_DATA_DIR, dataset_name, "med_dev_query.tsv")
        query_doc_rel_fpath = os.path.join(BASE_DATA_DIR, dataset_name, "med_dev.tsv")
        corpus_fpath = os.path.join(BASE_DATA_DIR, dataset_name, "med_corpus.tsv")
        return MultiCprRetrievalDataset(query_fpath, query_doc_rel_fpath, corpus_fpath)
    if dataset_name == 'qts_benchmark':
        from EmbedBoost.evaluate.qts_dataset import QTSRetrievalEvalDataset
        query_fpath = os.path.join(BASE_DATA_DIR, dataset_name, "full_benchmark.jsonl")
        corpus_fpath = os.path.join(BASE_DATA_DIR, dataset_name, "full_corpus.jsonl")
        return QTSRetrievalEvalDataset(query_fpath, corpus_fpath)
    raise ValueError(f"Invalid dataset name: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="评估句子嵌入模型的命令行脚本")
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="预训练模型路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--use_dense",
        action="store_true",
        help="使用稀疏向量"
    )
    parser.add_argument(
        "--dense_pooling",
        type=str,
        default="cls",
        choices=["cls", "mean", "max"],
        help="池化方法 (默认: cls)"
    )
    parser.add_argument(
        "--dense_dim",
        type=int,
        default=128,
        help="稠密向量维度 (默认: 128)"
    )
    parser.add_argument(
        "--infer_dense_dim",
        type=int,
        default=-1,
        help="推理稠密向量维度"
    )
    parser.add_argument(
        "--use_sparse",
        action="store_true",
        help="使用稀疏向量"
    )
    parser.add_argument(
        "--use_mrl",
        action="store_true",
        help="使用MRL (默认: false)"
    )
    parser.add_argument(
        "--mrl_dims",
        type=str,
        default="AUTO",
        help="mrl_dims"
    )
    parser.add_argument(
        "--max_query_length",
        type=int,
        default=128,
        help="最大查询序列长度 (默认: 128)"
    )
    parser.add_argument(
        "--max_doc_length",
        type=int,
        default=128,
        help="最大文档序列长度 (默认: 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备编号 (默认: cuda:0)"
    )
    parser.add_argument(
        "--milvus_db_uri",
        type=str,
        help="milvus db local uri"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        help="milvus db collection_name"
    )
    parser.add_argument(
        "--do_insert",
        action="store_true",
        help="drop collection if exists"
    )
    parser.add_argument(
        "--overwrite_collection",
        action="store_true",
        help="drop collection if exists"
    )
    parser.add_argument(
        "--insert_batch_size",
        type=int,
        default=500,
        help="插入数据库的批次大小"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="评估数据集名称"
    )
    parser.add_argument(
        "--query_line_limit",
        type=int,
        default=-1,
        help="Query测试行数"
    )
    parser.add_argument(
        "--corpus_line_limit",
        type=int,
        default=-1,
        help="Doc测试行数"
    )
    parser.add_argument(
        "--topk_list",
        type=str,
        default="1,10,20,50,100,200,300",
        help="topk_list"
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="do_eval"
    )
    parser.add_argument(
        "--output_fpath",
        type=str,
        default=None,
        help="保存路径"
    )
    
    # 解析参数
    args = parser.parse_args()
    print("🚀 启动评估任务，参数配置如下：")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    embedder = BGEM3Embedder(
        args.model_name_or_path,
        use_dense=args.use_dense,
        dense_pooling=args.dense_pooling,
        dense_dim=args.dense_dim,
        infer_dense_dim=args.infer_dense_dim if args.infer_dense_dim > 0 else args.dense_dim,
        use_sparse=args.use_sparse,
        use_mrl=args.use_mrl,
        mrl_dims=args.mrl_dims,
    )
    embedder.to(args.device)
    embedder.eval()

    eval_dataset = get_eval_dataset(args.dataset_name)
    # 构建vector store
    vector_store = MilvusVectorStore(
        args.milvus_db_uri, col_name=args.collection_name, use_dense=embedder.use_dense,
        dense_dim=embedder.infer_dense_dim, use_sparse=embedder.use_sparse
    )

    # 加载querys and docs
    query_list, doc_list = eval_dataset.load_datas(query_line_limit=args.query_line_limit, corpus_line_limit=args.corpus_line_limit)
    
    # 插入文档
    if args.do_insert:
        mode = "overwrite" if args.overwrite_collection else "append"
        vector_store.insert_docs(doc_list, embedder, args.max_doc_length, args.insert_batch_size, mode=mode)

    retrieval_modes = []
    if embedder.use_dense:
        retrieval_modes.append("dense")
    if embedder.use_sparse:
        retrieval_modes.append("sparse")

    topk_list = [int(x) for x in args.topk_list.split(",")]
    topk = max(topk_list)
    retrieved_results = retrieve(query_list, embedder, vector_store, retrieval_modes, max_query_length=args.max_query_length, topk=topk, merge_strategy='rrf')
    # print(json.dumps(query_list[:5], ensure_ascii=False))
    # print(json.dumps(retrieved_results[:5], ensure_ascii=False))
    
    if args.do_eval:
        metric_dict = evaluate_metric.compute_metrics(retrieved_results, topk_list=topk_list, metric_list=["recall@k", "mrr@k", "ndcg@k"])
        for retrieve_type, metric in metric_dict.items():
            print(f"retrieve_type: {retrieve_type}")
            print(json.dumps(metric, ensure_ascii=False, indent=4))
            print("")
    
    if args.output_fpath:
        with open(args.output_fpath, "w", encoding="utf-8") as wr:
            for res in retrieved_results:
                wr.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

