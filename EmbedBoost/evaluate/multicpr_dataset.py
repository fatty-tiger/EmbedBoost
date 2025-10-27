import logging
import os
import random
import json
from .base_dataset import AbsRetrievalEvalDataset

logger = logging.getLogger(__name__)


class MultiCprRetrievalDataset(AbsRetrievalEvalDataset):
    def __init__(self, query_fpath, query_doc_rel_fpath, corpus_fpath) -> None:
        self.query_fpath = query_fpath
        self.query_doc_rel_fpath = query_doc_rel_fpath
        self.corpus_fpath = corpus_fpath

    def load_querys(self):
        query2doc = {}
        with open(self.query_doc_rel_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                qid = f"query_{splits[0]}"
                did = f"doc_{splits[1]}"
                query2doc[qid] = did
        doc_list, doc_dict = self.load_corpus(return_list=False, return_dict=True)

        query_list = []
        with open(self.query_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                qid = f"query_{splits[0]}"
                item = {
                    'qid': qid,
                    'query': splits[1]
                }
                # TODO: 增加related doc的text
                if qid in query2doc:
                    did = query2doc[qid]
                    if did not in doc_dict:
                        continue
                    dtext = doc_dict[did]['text']
                    item['related_docs'] = [{
                        "id": query2doc[qid],
                        "text": dtext
                    }]
                query_list.append(item)
        logger.info(f"{len(query_list)} querys loaded.")
        return query_list

    def load_corpus(self, return_list=True, return_dict=False):
        doc_list = []
        doc_dict = {}
        with open(self.corpus_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                did = f"doc_{splits[0]}"
                if return_list:
                    doc_list.append({
                        'pk': did,
                        'text': splits[1]
                    })
                if return_dict:
                    doc_dict[did] = {'pk': did, 'text': splits[1]}
        logger.info(f"{len(doc_list)} documents loaded.")
        return doc_list, doc_dict


class QTSRetrievalEvalDataset(AbsRetrievalEvalDataset):
    def __init__(self, query_fpath, corpus_fpath) -> None:
        self.query_fpath = query_fpath
        self.corpus_fpath = corpus_fpath

    def load_querys(self):
        query_list = []
        with open(self.query_fpath) as f:
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                query_list.append({
                    'query': d['text'],
                    'related_docs': [
                        {
                            "id": d['skuno']
                        }
                    ]
                })
        logger.info(f"{len(query_list)} querys loaded.")
        return query_list
    
    def load_corpus(self, return_list=True, line_limit=-1):
        """
        TODO
        增加指定行数加载，增加随机读取模式；
        必须加载query相关的文档，支持小规模测试
        """

        # 先加载全部文档
        full_doc_list = []
        doc_dict = {}
        with open(self.corpus_fpath) as f:
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                did = d['skuno']
                # 为了关联查询的skuno，这里先加上
                doc_dict[did] = {'pk': did, 'text': d['text']}

                if return_list:
                    full_doc_list.append({
                        'pk': did,
                        'text': d['text']
                    })
        
        doc_ids = set()
        doc_list = []
        # 有行数限制
        if line_limit > 0:
            # 先载入query关联的文档
            with open(self.query_fpath) as f:
                for idx, line in enumerate(f):
                    d = json.loads(line.strip())
                    skuno = d['skuno']
                    if skuno in doc_ids or skuno not in doc_dict:
                        continue
                    text = doc_dict[skuno]['text']
                    doc_list.append({
                        'pk': skuno,
                        'text': text
                    })
                    doc_ids.add(skuno)
            
            # 再从全量文档中随机选择
            random.shuffle(full_doc_list)
            for item in full_doc_list:
                if item['pk'] in doc_ids:
                    continue
                doc_list.append(item)
                doc_ids.add(item['pk'])
                if len(doc_ids) == line_limit:
                    break
        else:
            doc_list = full_doc_list
        logger.info(f"{len(doc_list)} documents loaded.")
        return doc_list, doc_dict