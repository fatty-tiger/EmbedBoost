from torch.utils.data import Dataset

from EmbedBoost.common import file_util


class BiEncoderDataset(Dataset):
    def __init__(self, data_fpath_or_list, q_tokenizer, p_tokenizer, max_query_length, max_doc_length, mode, group_size=0):
        self.q_tokenizer = q_tokenizer
        self.p_tokenizer = p_tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.mode = mode
        self.group_size = group_size
        if isinstance(data_fpath_or_list, str):
            data_fpath_or_list = [data_fpath_or_list]
        self.datas = []
        for data_fpath in data_fpath_or_list:
            for idx, d in file_util.reader(data_fpath):
                if 'query' not in d or 'pos' not in d:
                    continue
                if self.group_size > 0:
                    if 'neg' not in d:
                        raise ValueError(f"neg not in data")
                    if len(d['neg']) < self.group_size - 1:
                        continue
                self.datas.append(d)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        if self.mode == 'inbatch_negative':
            querys = []
            pos = []
            for x in batch:
                querys.append(x['query'])
                pos.append(x['pos'])
            
            feed_dict_a = self.q_tokenizer(querys, max_length=self.max_query_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_b = self.p_tokenizer(pos, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            
            return feed_dict_a, feed_dict_b, None
        
        elif self.mode == 'explicit_negative':
            querys = []
            pos = []
            neg = []
            for x in batch:
                querys.append(x['query'])
                pos.append(x['pos'])
                neg.extend(x['neg'][:self.group_size-1])
                
            feed_dict_q = self.q_tokenizer(querys, max_length=self.max_query_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_pos = self.p_tokenizer(pos, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_neg = self.p_tokenizer(neg, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            return feed_dict_q, feed_dict_pos, feed_dict_neg
        
        raise ValueError(f"Invlid mode: {self.mode}")
