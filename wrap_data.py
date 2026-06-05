import torch
import torch.utils.data as Data
import unicodedata
import re

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.word2index = {self.index2word[idx]: idx for idx in self.index2word}
        self.n_words = 4  # Count <bos> and <eos> and <pad>

    def index_words(self, sentence):
        if self.name == 'cn':
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
    lines = open('./data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)


    return input_lang, output_lang, pairs


def filter_pair(p, max_length=10):
    return len(p[1].split(' ')) < max_length


def filter_pairs(pairs, max_length=10):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


def prepare_data(lang1_name, lang2_name, max_length=10, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    print(len(input_lang.word2count))
    print(len(output_lang.word2count))

    return input_lang, output_lang, pairs


def make_data(input_lang, output_lang, pairs, max_len):
    src_max_len = max([len(pair[0]) for pair in pairs])
    trg_max_len = max([len(pair[1].split(' ')) for pair in pairs])
    max_len = max(max_len, src_max_len+2, trg_max_len+3)
    print('max_len', max_len)
    print('src', src_max_len+2)
    enc_inputs = []
    dec_inputs = []
    for i in range(len(pairs)):
        single_enc_input = [output_lang.word2index["<bos>"]]
        single_dec_input = [output_lang.word2index["<bos>"]]
        for n in pairs[i][0]:
            try:
                single_enc_input.append(input_lang.word2index[n])
            except:
                single_enc_input.append(input_lang.word2index["<unk>"])
        for n in pairs[i][1].split(' '):
            try:
                single_dec_input.append(output_lang.word2index[n])
            except:
                single_dec_input.append(output_lang.word2index["<unk>"])
        single_enc_input.append(input_lang.word2index["<eos>"])
        single_dec_input.append(output_lang.word2index["<eos>"])
        single_enc_input_size = len(single_enc_input)
        single_dec_input_size = len(single_dec_input)
        for _ in range(max_len - single_enc_input_size):
            single_enc_input.append(input_lang.word2index["<pad>"])
        for _ in range(max_len - single_dec_input_size):
            single_dec_input.append(output_lang.word2index["<pad>"])
        enc_inputs.append(single_enc_input)
        dec_inputs.append(single_dec_input)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), max_len


class TransDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs):
        super(TransDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx]


def build_dataloader(lang1_name, lang2_name, max_length, max_len, batch_size, seed_worker, g, reverse=False):
    input_lang, output_lang, pairs = prepare_data(lang1_name, lang2_name, max_length=max_length, reverse=reverse)
    enc_inputs, dec_inputs, max_len = make_data(input_lang, output_lang, pairs, max_len)
    dataloader = Data.DataLoader(TransDataSet(enc_inputs,
                                              dec_inputs),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 worker_init_fn=seed_worker,
                                 generator=g
                                 )
    return dataloader, max_len, input_lang, output_lang, pairs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader('cn', 'eng')
    for enc, dec in enumerate(loader):
        print(enc, '\n', dec)
        break
