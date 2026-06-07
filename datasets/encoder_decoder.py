import torch
import torch.utils.data as Data
from tokenizer.tokenizer import Vocabulary

from .corpus import prepare_parallel_corpus


Lang = Vocabulary


def prepare_data(lang1_name, lang2_name,
                 max_target_sentence_split_length=10, reverse=False,
                 data_path="./data/zh-en.txt"):
    pairs, source_language, target_language = prepare_parallel_corpus(
        lang1_name,
        lang2_name,
        data_path=data_path,
        max_target_length=max_target_sentence_split_length,
        reverse=reverse,
    )
    input_lang = Lang(source_language)
    output_lang = Lang(target_language)
    for pair in pairs:
        input_lang.add_text(pair[0], character_level=input_lang.name == "zh")
        output_lang.add_text(pair[1], character_level=output_lang.name == "zh")
    return input_lang, output_lang, pairs


def make_data(input_lang, output_lang, pairs, min_sequence_token_length):
    enc_inputs = []
    dec_inputs = []
    encoded_pairs = []
    for source_text, target_text in pairs:
        source_ids = input_lang.encode(
            source_text, character_level=input_lang.name == "zh"
        )
        target_ids = output_lang.encode(
            target_text, character_level=output_lang.name == "zh"
        )
        encoded_pairs.append((source_ids, target_ids))
    source_max_length = max(len(source_ids) for source_ids, _ in encoded_pairs)
    target_max_length = max(len(target_ids) for _, target_ids in encoded_pairs)
    sequence_token_length = max(
        min_sequence_token_length,
        source_max_length + 2,
        target_max_length + 2,
    )

    for source_ids, target_ids in encoded_pairs:
        single_enc_input = [input_lang.word2index["<bos>"]]
        single_dec_input = [output_lang.word2index["<bos>"]]
        single_enc_input.extend(source_ids)
        single_dec_input.extend(target_ids)
        single_enc_input.append(input_lang.word2index["<eos>"])
        single_dec_input.append(output_lang.word2index["<eos>"])
        single_enc_input_size = len(single_enc_input)
        single_dec_input_size = len(single_dec_input)
        for _ in range(sequence_token_length - single_enc_input_size):
            single_enc_input.append(input_lang.word2index["<pad>"])
        for _ in range(sequence_token_length - single_dec_input_size):
            single_dec_input.append(output_lang.word2index["<pad>"])
        enc_inputs.append(single_enc_input)
        dec_inputs.append(single_dec_input)
    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        sequence_token_length,
    )


class TransDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs):
        super(TransDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx]


def build_dataloader(
    lang1_name,
    lang2_name,
    max_target_sentence_split_length,
    min_sequence_token_length,
    batch_size,
    seed_worker,
    g,
    reverse=False,
    data_path="./data/zh-en.txt",
):
    input_lang, output_lang, pairs = prepare_data(
        lang1_name,
        lang2_name,
        max_target_sentence_split_length=max_target_sentence_split_length,
        reverse=reverse,
        data_path=data_path,
    )
    enc_inputs, dec_inputs, sequence_token_length = make_data(
        input_lang,
        output_lang,
        pairs,
        min_sequence_token_length,
    )
    dataloader = Data.DataLoader(TransDataSet(enc_inputs,
                                              dec_inputs),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 worker_init_fn=seed_worker,
                                 generator=g
                                 )
    return dataloader, sequence_token_length, input_lang, output_lang, pairs
