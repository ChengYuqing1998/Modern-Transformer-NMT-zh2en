from trainer import *
import argparse
import wandb
from translator import *
from wrap_data import *
import random
import numpy as np
import yaml
from torch.utils.data import DataLoader, random_split
import nltk

# fix the seed of random number generators
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    params = argparse.ArgumentParser("Augment data based on a specified method")
    params.add_argument('--config_file_path', type=str, default='./configs/c2e_transformer.yaml',
                        help='Specifying the path where to look up for configs file.')
    args = params.parse_args()

    c2e_configs_path = args.config_file_path

    with open(c2e_configs_path, 'r') as f:
        c2e_configs = yaml.safe_load(f)
    print(c2e_configs['trial_id'])

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = c2e_configs['CUBLAS_WORKSPACE_CONFIG']
    torch.manual_seed(c2e_configs['random_seed'])
    random.seed(c2e_configs['random_seed'])
    np.random.seed(c2e_configs['random_seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    def seed_worker():
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(c2e_configs['random_seed'])

    wandb.init(project="c2e_transformer",
               entity=c2e_configs['wandb_entity'],
               name=str(c2e_configs['trial_id']),
               config=c2e_configs)

    loader, max_tensor_len, input_lang, output_lang, pairs = build_dataloader('cn',
                                                                              'eng',
                                                                              c2e_configs['max_trg_sent_len'],
                                                                              c2e_configs['refer_max_tensor_len'],
                                                                              c2e_configs['batch_size'],
                                                                              seed_worker,
                                                                              g,
                                                                              False)
    # save the input_lang and notice that when you change the max_trg_sent_len the langs would also be changed:
    if not os.path.exists('./data/input_lang.pkl'):
        with open('./data/input_lang.pkl', 'wb') as f:
            pickle.dump(input_lang, f)
        with open('./data/output_lang.pkl', 'wb') as f:
            pickle.dump(output_lang, f)

    len_dataloader = len(loader.dataset)

    train_ratio = c2e_configs['train_ratio']
    val_ratio = c2e_configs['val_ratio']

    train_len = int(train_ratio * len_dataloader)
    val_len = int(val_ratio * len_dataloader)
    test_len = len_dataloader - train_len - val_len

    train_set, val_set, test_set = random_split(loader.dataset, [train_len, val_len, test_len], generator=g)

    # build DataLoader
    train_loader = DataLoader(train_set, batch_size=c2e_configs['batch_size'],
                              shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_set, batch_size=c2e_configs['batch_size'],
                            shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=c2e_configs['batch_size'],
                             shuffle=False, worker_init_fn=seed_worker, generator=g)

    model = build_model(input_lang, output_lang, max_tensor_len, c2e_configs)
    # model = torch.load('./models/c2e_transformer_[0526-test1].pt')
    model = model.to(eval(c2e_configs['device']))
    wandb.watch(model)

    trainer = build_trainer(model, c2e_configs)
    trainer.fit(train_loader, val_loader,
                input_lang,
                output_lang,
                c2e_configs['max_epochs'],
                warmup=c2e_configs['warmup'])

    translator = Translator(model,
                            c2e_configs['beam_size'],
                            max_tensor_len,
                            c2e_configs['bos_token'],
                            c2e_configs['eos_token'],
                            device=eval(c2e_configs['device']))
    bleu4_test = 0.0
    sample_num = 0
    for x, y in test_loader:
        sample_num += len(x)
        res = translator.translate(x)
        refer = y.unsqueeze(1).cpu().numpy().tolist()
        refer_clean = remove_element(refer, 0)
        bleu4 = nltk.translate.bleu_score.corpus_bleu(refer_clean, res, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4_test += bleu4 * len(x)

    print('Testing Bleu4 =', '{:.6f}'.format(bleu4_test / sample_num))
    wandb.log({'test_bleu4': bleu4_test / sample_num})


if __name__ == '__main__':
    main()
