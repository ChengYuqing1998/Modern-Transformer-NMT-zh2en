from pathlib import Path

from tokenizer.tokenizer import normalize_string


def read_parallel_corpus(path):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    pairs = []
    for line_number, line in enumerate(lines, start=1):
        columns = line.split("\t")
        if len(columns) < 2:
            raise ValueError(f"{path}:{line_number} is not a tab-separated pair")
        pairs.append(
            [normalize_string(columns[0]), normalize_string(columns[1])]
        )
    return pairs


def prepare_parallel_corpus(
    source_language,
    target_language,
    data_path="./data/zh-en.txt",
    max_target_length=128,
    reverse=False,
):
    pairs = read_parallel_corpus(data_path)
    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        source_language, target_language = target_language, source_language
    pairs = [
        pair
        for pair in pairs
        if len(pair[1].split()) <= max_target_length
    ]
    return pairs, source_language, target_language
