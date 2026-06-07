import torch
import torch.utils.data as Data
from tokenizer.tokenizer import Vocabulary

from .corpus import prepare_parallel_corpus


Lang = Vocabulary


def prepare_data(lang1_name, lang2_name,
                 max_target_sentence_split_length=10, reverse=False,
                 data_path="./data/zh-en.txt"):
    pairs, src_lang_name, trg_lang_name = prepare_parallel_corpus(
        lang1_name,
        lang2_name,
        data_path=data_path,
        max_target_length=max_target_sentence_split_length,
        reverse=reverse,
    )
    unified_lang = Lang(name='mixed')
    for pair in pairs:
        unified_lang.add_text(pair[0], character_level=(src_lang_name == 'zh'))
        unified_lang.add_text(pair[1], character_level=(trg_lang_name == 'zh'))
    return unified_lang, pairs, src_lang_name, trg_lang_name


def make_data(
    unified_lang,
    pairs,
    max_context_len,
    min_sequence_token_length,
    src_lang_name='zh',
    trg_lang_name='en',
):
    """
    Create data in decoder-only format: [BOS, src_tokens, trg_tokens, EOS, PAD, ...]
    Returns:
        - inputs: concatenated sequences [BOS, src, trg, EOS, PAD, ...]
        - targets: target sequences for loss [src, trg, EOS, PAD, ...] (input shifted by 1)
        - src_lengths: length of src part (excluding BOS) for each sample, used to mask src in loss
        - sequence_token_length: padded sequence length
    """

    assert max_context_len is not None and min_sequence_token_length is not None
    assert max_context_len > min_sequence_token_length


    encoded_pairs = []
    sequence_token_length = 0
    for source_text, target_text in pairs:
        source_ids = unified_lang.encode(
            source_text, character_level=src_lang_name == "zh"
        )
        target_ids = unified_lang.encode(
            target_text, character_level=trg_lang_name == "zh"
        )
        actual_len = len(source_ids) + len(target_ids) + 2
        sequence_token_length = max(sequence_token_length, actual_len)
        encoded_pairs.append((source_ids, target_ids, actual_len))

    if sequence_token_length > max_context_len:
        raise ValueError(
            "The longest decoder-only sample requires "
            f"{sequence_token_length} tokens, which exceeds max_context_len "
            f"{max_context_len}."
        )

    sequence_token_length  = max(min_sequence_token_length, sequence_token_length)

    inputs = []
    targets = []
    src_lengths = []

    for source_ids, target_ids, actual_len in encoded_pairs:
        # Build input sequence: [BOS, src_tokens, trg_tokens, EOS]
        single_input = [unified_lang.word2index["<bos>"]]
        single_input.extend(source_ids)
        single_input.extend(target_ids)

        # Add EOS
        single_input.append(unified_lang.word2index["<eos>"])

        # Target sequence is input shifted by 1: [src_tokens, trg_tokens, EOS, PAD, ...]
        # This means target[i] corresponds to the next token after input[i]
        single_target = single_input[1:]  # Remove BOS, target starts from first src token

        # Record src length (number of src tokens, excluding BOS)
        # This is used to mask src part in loss calculation
        src_length = len(source_ids)
        src_lengths.append(src_length)

        # Right padding - pad both input and target to sequence_token_length
        pad_token = unified_lang.word2index["<pad>"]
        for _ in range(sequence_token_length - actual_len):
            single_input.append(pad_token)
            single_target.append(pad_token)

        # Target is 1 shorter than input (removed BOS), add one more PAD to match length
        single_target.append(pad_token)

        inputs.append(single_input)
        targets.append(single_target)

    return (
        torch.LongTensor(inputs),
        torch.LongTensor(targets),
        torch.LongTensor(src_lengths),
        sequence_token_length,
    )


class GPTDataSet(Data.Dataset):
    def __init__(self, inputs, targets, src_lengths):
        super(GPTDataSet, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.src_lengths = src_lengths

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.src_lengths[idx]


def build_dataloader(
    lang1_name,
    lang2_name,
    max_target_sentence_split_length,
    max_context_len=None,
    min_sequence_token_length=None,
    batch_size=None,
    seed_worker=None,
    g=None,
    reverse=False,
    data_path="./data/zh-en.txt",
):
    if batch_size is None or seed_worker is None or g is None:
        raise ValueError("batch_size, seed_worker, and g must be provided")
    unified_lang, pairs, src_lang_name, trg_lang_name = prepare_data(
        lang1_name,
        lang2_name,
        max_target_sentence_split_length=max_target_sentence_split_length,
        reverse=reverse,
        data_path=data_path,
    )
    inputs, targets, src_lengths, sequence_token_length = make_data(
        unified_lang,
        pairs,
        max_context_len=max_context_len,
        min_sequence_token_length=min_sequence_token_length,
        src_lang_name=src_lang_name,
        trg_lang_name=trg_lang_name,
    )

    dataset = GPTDataSet(inputs, targets, src_lengths)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return dataloader, sequence_token_length, unified_lang, pairs
