from translator import *
import pickle
import argparse


def indexes_from_sentence(lang, sentence):
    enc_list = []
    if lang.name == 'cn':
        for word in sentence:
            try:
                enc_list.append(lang.word2index[word])
            except:
                enc_list.append(UNK_token)
        return enc_list
    else:
        for word in sentence.split(' '):
            try:
                enc_list.append(lang.word2index[word])
            except:
                enc_list.append(UNK_token)
        return enc_list


def variable_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    var = var.to(device)
    return var


def main():
    params = argparse.ArgumentParser("Augment data based on a specified method")
    params.add_argument('--model_path', type=str, default='./models/c2e_transformer_[0526-test1].pt',
                        help='Specifying the path where to look up for model.')
    params.add_argument('--input_lang_path', type=str, default='./data/input_lang.pkl',
                        help='Specifying the path where to look up for input lang.')
    params.add_argument('--output_lang_path', type=str, default='./data/output_lang.pkl',
                        help='Specifying the path where to look up for output lang.')
    params.add_argument('--device', type=str, default='auto', help='choose cpu cuda or auto')

    args = params.parse_args()
    assert args.device in ['cpu', 'cuda', 'auto']
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = torch.load(args.model_path, map_location=device)

    with open(args.input_lang_path, 'rb') as f:
        input_lang = pickle.load(f)
    with open(args.output_lang_path, 'rb') as f:
        output_lang = pickle.load(f)

    count = 1
    limit = 10  # it allows you try 10 sentences till it'll be closed

    translator = Translator(model, 5, model.max_len, model.trg_bos_idx, model.trg_eos_idx, device=device)
    while True:
        if count <= limit:
            description = "please enter Chinese sentence (" + str(limit-count+1) + " left) : "
            sentence = input(description).strip()
            enc_inputs = variable_from_sentence(input_lang, sentence, device).view(1, -1)
            enc_input_list = list(*enc_inputs.cpu().numpy())
            enc_input_list.pop(-1)
            enc_input_list.append(EOS_token)
            enc_inputs = [BOS_token]
            enc_inputs.extend(enc_input_list)
            enc_inputs = torch.tensor(enc_inputs, dtype=torch.long, device=device).reshape(1, -1)
            pred_seqs = translator.translate(enc_inputs)
            output_sentence = [' '.join([output_lang.index2word[_] for _ in pred_seq]) for pred_seq in pred_seqs]
            print(output_sentence[0])
            count += 1
        else:
            break


if __name__ == '__main__':
    main()



