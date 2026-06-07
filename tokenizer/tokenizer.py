import re
import unicodedata


class Vocabulary:
    """Minimal educational tokenizer and vocabulary."""

    def __init__(self, name):
        self.name = name
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.word2index = {token: index for index, token in self.index2word.items()}
        self.n_words = len(self.index2word)

    def add_text(self, text, character_level=False):
        tokens = text if character_level else text.split()
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.word2index:
            self.word2index[token] = self.n_words
            self.index2word[self.n_words] = token
            self.word2count[token] = 1
            self.n_words += 1
        else:
            self.word2count[token] = self.word2count.get(token, 0) + 1

    def encode(self, text, character_level=False, add_bos=False, add_eos=False):
        tokens = text if character_level else text.split()
        ids = [self.word2index.get(token, self.word2index["<unk>"]) for token in tokens]
        if add_bos:
            ids.insert(0, self.word2index["<bos>"])
        if add_eos:
            ids.append(self.word2index["<eos>"])
        return ids

    def decode(self, token_ids, skip_special_tokens=True, separator=" "):
        special_ids = {
            self.word2index["<pad>"],
            self.word2index["<bos>"],
            self.word2index["<eos>"],
        }
        tokens = []
        for token_id in token_ids:
            if token_id == self.word2index["<eos>"]:
                break
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.index2word.get(int(token_id), "<unk>"))
        return separator.join(tokens)

    def __len__(self):
        return self.n_words


# Backward-friendly name inside the new module.
Lang = Vocabulary


def unicode_to_ascii(text):
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def normalize_string(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", " ", text)
