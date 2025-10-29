import logging

from .base_dataset import AbsRetrievalEvalDataset

logger = logging.getLogger(__name__)


class MultiCprRetrievalDataset(AbsRetrievalEvalDataset):
    def __init__(self, query_fpath, query_doc_rel_fpath, corpus_fpath) -> None:
        self.query_fpath = query_fpath
        self.query_doc_rel_fpath = query_doc_rel_fpath
        self.corpus_fpath = corpus_fpath
    
    def load_corpus(self):
        doc_list = []
        doc_dict = {}
        with open(self.corpus_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                did = f"doc_{splits[0]}"
                doc_list.append({
                    'id': did,
                    'text': splits[1]
                })
                doc_dict[did] = {'id': did, 'text': splits[1]}
        return doc_list, doc_dict
    
    def load_datas(self, query_line_limit=-1, corpus_line_limit=-1):
        full_doc_list, doc_dict = self.load_corpus()
        
        query2doc = {}
        with open(self.query_doc_rel_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                qid = f"query_{splits[0]}"
                did = f"doc_{splits[1]}"
                query2doc[qid] = did

        query_list = []
        doc_list = []
        exclude_docs = set()
        with open(self.query_fpath) as f:
            for idx, line in enumerate(f):
                splits = line.strip().split("\t")
                if len(splits) != 2:
                    continue
                qid = f"query_{splits[0]}"
                if qid not in query2doc:
                    continue
                
                did = query2doc[qid]
                if did not in doc_dict:
                    continue
                dtext = doc_dict[did]['text']

                item = {
                    'qid': qid,
                    'query': splits[1],
                    'related_docs': [{
                        "id": query2doc[qid],
                        "text": dtext
                    }]
                }
                query_list.append(item)
                if did not in exclude_docs:
                    doc_list.append(doc_dict[did])
                    exclude_docs.add(did)
                if query_line_limit > 0 and len(query_list) >= query_line_limit:
                    break
        logger.info(f"{len(query_list)} querys loaded.")

        for item in full_doc_list:
            if item['id'] in exclude_docs:
                continue
            doc_list.append(item)
            exclude_docs.add(item['id'])
            if corpus_line_limit > 0 and len(doc_list) >= corpus_line_limit:
                break
        logger.info(f"{len(doc_list)} documents loaded.")

        return query_list, doc_list