import logging
import random
import json
from .base_dataset import AbsRetrievalEvalDataset

logger = logging.getLogger(__name__)


class QTSRetrievalEvalDataset(AbsRetrievalEvalDataset):
    def __init__(self, query_fpath, corpus_fpath) -> None:
        self.query_fpath = query_fpath
        self.corpus_fpath = corpus_fpath

    def load_datas(self, query_line_limit=-1, corpus_line_limit=-1):
        """
        TODO
        增加指定行数加载，增加随机读取模式；
        必须加载query相关的文档，支持小规模测试
        """
        query_list = []
        with open(self.query_fpath) as f:
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                query_list.append({
                    'query': d['query'],
                    'related_docs': [
                        {
                            "id": d['skuno']
                        }
                    ]
                })
                if query_line_limit > 0 and len(query_list) >= query_line_limit:
                    break

        # 先加载全部文档
        full_doc_list = []
        doc_dict = {}
        with open(self.corpus_fpath) as f:
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                did = d['id']
                doc_dict[did] = {'pk': did, 'text': d['text']}
                full_doc_list.append({
                    'id': did,
                    'text': d['text']
                })
        
        doc_ids = set()
        doc_list = []
        # 有行数限制
        if corpus_line_limit > 0:
            # 先载入query关联的文档
            for d in query_list:
                for x in d['related_docs']:
                    qid = x['id']
                    if qid in doc_ids or qid not in doc_dict:
                        continue
                    text = doc_dict[qid]['text']
                    doc_list.append({
                        'id': qid,
                        'text': text
                    })
                    doc_ids.add(qid)
            
            # 再从全量文档中随机选择
            random.shuffle(full_doc_list)
            for item in full_doc_list:
                if item['id'] in doc_ids:
                    continue
                doc_list.append(item)
                doc_ids.add(item['id'])
                if len(doc_ids) == corpus_line_limit:
                    break
        else:
            doc_list = full_doc_list
        logger.info(f"{len(query_list)} querys loaded.")
        logger.info(f"{len(doc_list)} documents loaded.")
        return query_list, doc_list

