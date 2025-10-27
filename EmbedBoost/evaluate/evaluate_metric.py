import json
import math
import collections
from typing import List, Dict, Any


def compute_metrics(
    result_list: List[Dict[str, Any]],
    topk_list: List[int],
    metric_list: List[str]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        result_list: Retrieval results
            [{"query": "query content here", 
                "related_docs": [{"id": "id1", "text": "text1"}], 
                "dense_retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]}],
                "sparse_retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]}],
                "merged_docs": [{"id": "id1", "text": "text1", "score": score}]}],
    Returns:
        metrics: Computed metric values
            {"Recall@10": 0.8, "Ndcg@10": 0.75}
    """       
    # ret_dict = collections.defaultdict(lambda: collections.defaultdict())
    ret_dict = {}
    # print(json.dumps(result_list[0], ensure_ascii=False))

    for retrieve_type in ["dense_retrieved_docs", "sparse_retrieved_docs", "merged_docs"]:
        sub_result_list = []
        for d in result_list:
            if retrieve_type not in d:
                continue
            sub_result_list.append({
                "query": d["query"],
                "related_docs": d["related_docs"],
                "retrieved_docs": d[retrieve_type][:]
            })

        if 'recall@k' in metric_list:
            recall_dict = compute_recall(sub_result_list, topk_list)
            if retrieve_type not in ret_dict:
                ret_dict[retrieve_type] = {}
            ret_dict[retrieve_type].update(recall_dict)
        if 'mrr@k' in metric_list:
            mrr_dict = compute_mrr(sub_result_list, topk_list)
            if retrieve_type not in ret_dict:
                ret_dict[retrieve_type] = {}
            ret_dict[retrieve_type].update(mrr_dict)
        if 'ndcg@k' in metric_list:
            ndcg_dict = compute_ndcg(sub_result_list, topk_list)
            if retrieve_type not in ret_dict:
                ret_dict[retrieve_type] = {}
            ret_dict[retrieve_type].update(ndcg_dict)
    return ret_dict


def compute_recall(result_list: List[Dict[str, Any]], topk_list: List[int]) -> Dict[str, float]:
    """
    Compute mrr metric.
    Args:
        result_list: Retrieval results
            [
                {
                    "query": "query content here", 
                    "related_docs": [{"id": "id1", "text": "text1"}], 
                    "retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]
                }
            ]
    Returns:
        metrics: Computed metric values  Example: {"RECALL@1": 0.3, "RECALL@10": 0.7}
    """    
    total = 0
    hit_dict = collections.defaultdict(int)
    for result in result_list:
        #print(json.dumps(result, ensure_ascii=False))
        target = result["related_docs"][0]["id"]
        retrieved_docs = [x['id'] for x in result['retrieved_docs']]
        total += 1
        pos = -1
        try:
            pos = retrieved_docs.index(target)
        except:
            pass
        for k in topk_list:
            if 0 <= pos < k:
                hit_dict[k] += 1
    if total > 0:
        print(f"total: {total}")
        ret_dict = {}
        for k in topk_list:
            recall_k = round(hit_dict[k] / total, 4)
            ret_dict[f"Recall@{k}"] = recall_k
        return ret_dict
        # return {f"Recall@{k}": hit_dict[k] / total for k in topk_list}
    
        
    return dict()


def compute_mrr(result_list: List[Dict[str, Any]], topk_list: List[int]) -> Dict[str, float]:
    """
    Compute mrr metric.

    Args:
        result_list: Retrieval results
            [
                {
                    "query": "query content here", 
                    "related_docs": [{"id": "id1", "text": "text1"}], 
                    "retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]
                }
            ]
    Returns:
        metrics: Computed metric values  Example: {"MRR@1": 0.3, "MRR@10": 0.7}
    """
    metrics = {}
    
    # For each topk value, compute MRR
    for topk in topk_list:
        mrr_sum = 0.0
        query_count = len(result_list)
        
        # For each query result
        for result in result_list:
            related_ids = set(doc["id"] for doc in result["related_docs"])
            retrieved_docs = result["retrieved_docs"]
            
            # Find the first relevant document within topk results
            reciprocal_rank = 0.0
            for i, doc in enumerate(retrieved_docs[:topk]):
                if doc["id"] in related_ids:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            
            mrr_sum += reciprocal_rank
        
        # Calculate mean reciprocal rank for this topk
        mrr = mrr_sum / query_count if query_count > 0 else 0.0
        metrics[f"MRR@{topk}"] = round(mrr, 4)
    
    return metrics



def compute_ndcg(result_list: List[Dict[str, Any]], topk_list: List[int]) -> Dict[str, float]:
    """
    Compute mrr metric.

    Args:
        result_list: Retrieval results
            [
                {
                    "query": "query content here", 
                    "related_docs": [{"id": "id1", "text": "text1"}], 
                    "retrieved_docs": [{"id": "id1", "text": "text1", "score": score}]
                }
            ]
    Returns:
        metrics: Computed metric values  Example: {"NDCG@1": 0.3, "NDCG@10": 0.7}
    """
    metrics = {}
    
    # For each topk value, compute NDCG
    for topk in topk_list:
        ndcg_sum = 0.0
        query_count = len(result_list)
        
        # For each query result
        for result in result_list:
            related_ids = set(doc["id"] for doc in result["related_docs"])
            retrieved_docs = result["retrieved_docs"]
            
            # Calculate DCG (Discounted Cumulative Gain)
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs[:topk]):
                if doc["id"] in related_ids:
                    if i == 0:
                        dcg += 1.0  # Rank 1 has discount of log(1+1) = log(2) = 1
                    else:
                        dcg += 1.0 / math.log2(i + 1)
            
            # Calculate IDCG (Ideal Discounted Cumulative Gain)
            ideal_len = min(len(related_ids), topk)
            idcg = 0.0
            for i in range(ideal_len):
                if i == 0:
                    idcg += 1.0
                else:
                    idcg += 1.0 / math.log2(i + 1)
            
            # Calculate NDCG for this query
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum += ndcg
        
        # Calculate mean NDCG for this topk
        ndcg_avg = ndcg_sum / query_count if query_count > 0 else 0.0
        metrics[f"NDCG@{topk}"] = round(ndcg_avg, 4)
    
    return metrics