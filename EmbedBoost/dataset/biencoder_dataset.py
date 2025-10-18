from torch.utils.data import Dataset

from EmbedBoost.common import file_util


class BiEncoderDataset(Dataset):
    def __init__(self, data_fpath_or_list, tokenizer, max_query_length, max_doc_length, mode, group_size=0):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.mode = mode
        self.group_size = group_size
        if isinstance(data_fpath_or_list, str):
            data_fpath_or_list = [data_fpath_or_list]
        self.datas = []
        for data_fpath in data_fpath_or_list:
            for idx, d in file_util.reader(data_fpath):
                if 'query' not in d or 'positive' not in d:
                    continue
                if self.group_size > 0:
                    if 'negatives' not in d:
                        raise ValueError(f"negatives not in data")
                    if len(d['negatives']) < self.group_size - 1:
                        continue
                self.datas.append(d)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        if self.mode == 'inbatch_negative':
            querys = []
            positives = []
            for x in batch:
                querys.append(x['query'])
                positives.append(x['positive'])
            
            feed_dict_a = self.tokenizer(querys, max_length=self.max_query_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_b = self.tokenizer(positives, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            
            return feed_dict_a, feed_dict_b, None
        
        elif self.mode == 'explicit_negative':
            querys = []
            positives = []
            negatives = []
            for x in batch:
                querys.append(x['query'])
                positives.append(x['positive'])
                #negatives.extend(x['negatives'][:self.group_size-1])
                negatives.extend(x['negatives'][10:10+self.group_size-1])
                # negatives.extend(x['negatives'][20:20+self.group_size-1])
                
            
            feed_dict_q = self.tokenizer(querys, max_length=self.max_query_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_pos = self.tokenizer(positives, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            feed_dict_neg = self.tokenizer(negatives, max_length=self.max_doc_length, add_special_tokens=True,
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
            return feed_dict_q, feed_dict_pos, feed_dict_neg
        
        raise ValueError(f"Invlid mode: {self.mode}")
