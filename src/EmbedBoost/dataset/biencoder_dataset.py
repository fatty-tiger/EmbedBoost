from torch.utils.data import Dataset

from EmbedBoost.common import file_util


class BiEncoderDataset(Dataset):
    def __init__(self, data_fpath_or_list, tokenizer, max_query_length, max_doc_length, mode):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.mode = mode
        if isinstance(data_fpath_or_list, str):
            data_fpath_or_list = [data_fpath_or_list]
        self.datas = []
        for data_fpath in data_fpath_or_list:
            self.datas.extend([d for idx, d in file_util.reader(data_fpath)])

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
            
            return feed_dict_a, feed_dict_b
        
        elif self.mode == 'explicit_negative':
            querys = []
            positives = []
            negatives = []
            for x in batch:
                querys.append(x['query'])
                positives.append(x['positive_text'])
                negatives.extend(x['negative_texts'][:self.group_size-1])
            
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
