import json


def reader(input_file, encoding='utf8', fmt='raw'):
    if fmt not in ["raw", "jsonl"]:
        raise ValueError("unsupported format")
    with open(input_file, encoding=encoding) as f:
        if fmt == 'jsonl' or input_file.endswith(".jsonl"):
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                yield idx, d
        else:
            for idx, line in enumerate(f):
                yield idx, line.strip()


def batch_generator(input_fpath_or_list, batch_size):
    idx, batch = 0, []
    if isinstance(input_fpath_or_list, str):
        f = open(input_fpath_or_list, encoding='utf8')
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield idx, batch
                batch = []
                idx += 1
        f.close()
    elif isinstance(input_fpath_or_list, list):
        for x in input_fpath_or_list:
            batch.append(x)
            if len(batch) == batch_size:
                yield idx, batch
                batch = []
                idx += 1

    if len(batch) > 0:
        yield idx, batch


def save_jsonl(datas, output_fpath): 
    if not output_fpath.endswith('.jsonl'):
        raise TypeError("output_fpath is not jsonl format!")
    with open(output_fpath, "w") as wr:
        for d in datas:
            wr.write(json.dumps(d, ensure_ascii=False) + '\n')
