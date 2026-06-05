import torch
import torch.utils.data as Data
import unicodedata
import re

class Lang:
    def __init__(self, name='mixed'):
        self.name = name
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.word2index = {self.index2word[idx]: idx for idx in self.index2word}
        self.n_words = 4  # Count <bos> and <eos> and <pad>

    def index_words(self, sentence, is_cn=False):
        """Index words from a sentence, supporting both Chinese and English"""
        if is_cn:
            for word in sentence:
                self.index_word(word)
        else:
            for word in sentence.split(' '):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs if needed
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        src_lang_name = lang2
        trg_lang_name = lang1
    else:
        src_lang_name = lang1
        trg_lang_name = lang2

    return src_lang_name, trg_lang_name, pairs


def filter_pair(p, max_length=10):
    return len(p[1].split(' ')) < max_length


def filter_pairs(pairs, max_length=10):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


def prepare_data(lang1_name, lang2_name, max_length=10, reverse=False):
    src_lang_name, trg_lang_name, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))

    # Create a unified Lang for both source and target
    unified_lang = Lang(name='mixed')
    
    print("Indexing words...")
    for pair in pairs:
        # Index source language words (Chinese)
        unified_lang.index_words(pair[0], is_cn=(src_lang_name == 'cn'))
        # Index target language words (English)
        unified_lang.index_words(pair[1], is_cn=(trg_lang_name == 'cn'))
    
    print("Total vocabulary size:", unified_lang.n_words)
    print("Source language:", src_lang_name)
    print("Target language:", trg_lang_name)

    return unified_lang, pairs, src_lang_name, trg_lang_name


def make_data(unified_lang, pairs, max_len, src_lang_name='cn'):
    """
    Create data in decoder-only format: [BOS, src_tokens, trg_tokens, EOS, PAD, ...]
    Returns:
        - inputs: concatenated sequences [BOS, src, trg, EOS, PAD, ...]
        - targets: target sequences for loss [src, trg, EOS, PAD, ...] (input shifted by 1)
        - src_lengths: length of src part (excluding BOS) for each sample, used to mask src in loss
        - max_len: maximum sequence length
    """
    # Calculate max lengths
    src_max_len = max([len(pair[0]) for pair in pairs])  # Chinese character count
    trg_max_len = max([len(pair[1].split(' ')) for pair in pairs])  # English word count
    # max_len = BOS(1) + src + trg + EOS(1) + padding
    max_len = max(max_len, src_max_len + trg_max_len + 3)  # +3 for BOS, EOS, and some buffer
    print('max_len', max_len)
    print('src_max_len (chars)', src_max_len)
    print('trg_max_len (words)', trg_max_len)
    
    inputs = []
    targets = []
    src_lengths = []
    
    for i in range(len(pairs)):
        # Build input sequence: [BOS, src_tokens, trg_tokens, EOS]
        single_input = [unified_lang.word2index["<bos>"]]
        
        # Add source tokens (Chinese)
        for n in pairs[i][0]:
            try:
                token = unified_lang.word2index[n]
            except:
                token = unified_lang.word2index["<unk>"]
            single_input.append(token)
        
        # Add target tokens (English)
        for n in pairs[i][1].split(' '):
            try:
                token = unified_lang.word2index[n]
            except:
                token = unified_lang.word2index["<unk>"]
            single_input.append(token)
        
        # Add EOS
        single_input.append(unified_lang.word2index["<eos>"])
        
        # Target sequence is input shifted by 1: [src_tokens, trg_tokens, EOS, PAD, ...]
        # This means target[i] corresponds to the next token after input[i]
        single_target = single_input[1:]  # Remove BOS, target starts from first src token
        
        # Record src length (number of src tokens, excluding BOS)
        # This is used to mask src part in loss calculation
        src_length = len(pairs[i][0])  # Number of Chinese characters
        src_lengths.append(src_length)
        
        # Right padding - pad both input and target to max_len
        actual_len = len(single_input)
        pad_token = unified_lang.word2index["<pad>"]
        for _ in range(max_len - actual_len):
            single_input.append(pad_token)
            single_target.append(pad_token)
        
        # Target is 1 shorter than input (removed BOS), add one more PAD to match length
        single_target.append(pad_token)
        
        inputs.append(single_input)
        targets.append(single_target)
    
    return torch.LongTensor(inputs), torch.LongTensor(targets), torch.LongTensor(src_lengths), max_len


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


def build_dataloader(lang1_name, lang2_name, max_length, max_len, batch_size, seed_worker, g, reverse=False):
    unified_lang, pairs, src_lang_name, trg_lang_name = prepare_data(
        lang1_name, lang2_name, max_length=max_length, reverse=reverse
    )
    inputs, targets, src_lengths, max_len = make_data(
        unified_lang, pairs, max_len, src_lang_name=src_lang_name
    )
    
    dataset = GPTDataSet(inputs, targets, src_lengths)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return dataloader, max_len, unified_lang, pairs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def seed_worker():
        worker_seed = torch.initial_seed() % 2 ** 32
        import random
        import numpy as np
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    loader, max_len, lang, pairs = build_dataloader(
        'cn', 'eng', max_length=128, max_len=32, 
        batch_size=2, seed_worker=seed_worker, g=g
    )
    
    for inputs, targets, src_lengths in loader:
        print("Batch size:", inputs.shape[0])
        print("Sequence length:", inputs.shape[1])
        print("Input shape:", inputs.shape)
        print("Target shape:", targets.shape)
        print("Src lengths:", src_lengths)
        print("\nFirst sample:")
        print("Input:", inputs[0])
        print("Target:", targets[0])
        print("Src length (excluding BOS):", src_lengths[0].item())
        break
