from collections import defaultdict

def rrf_ranking(sorted_lists, topk, rank_constant=60):
    score_dict = defaultdict(float)
    for lst in sorted_lists:
        for rank, x in enumerate(lst, start=1):
            score_dict[x] += 1.0 / (rank_constant + rank)

    # 按照RRF得分降序排序
    ranked_items = [x[0] for x in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)][:topk]
    return ranked_items


if __name__ == '__main__':
    # 示例输入
    list1 = ['a', 'b', 'c', 'd']
    list2 = ['b', 'c', 'a', 'e']
    topk = 4

    # 合并并取 Top-3
    result = rrf_ranking([list1, list2], topk)
    print(f"Top-{topk} items by RRF: {result}")