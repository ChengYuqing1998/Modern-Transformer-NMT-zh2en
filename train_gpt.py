from trainer_gpt import *
import argparse
import wandb
from wrap_data_gpt import *
import random
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, random_split
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    params = argparse.ArgumentParser("Train GPT model for Chinese to English translation")
    params.add_argument('--config_file_path', type=str, default='./c2e_gpt_configs.yaml',
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

    wandb.init(project="c2e_gpt",
               entity=c2e_configs['wandb_entity'],
               name=str(c2e_configs['trial_id']) + '_gpt',
               config=c2e_configs)

    # Build dataloader using GPT data format
    loader, max_tensor_len, unified_lang, pairs = build_dataloader('cn',
                                                                    'eng',
                                                                    c2e_configs['max_trg_sent_len'],
                                                                    c2e_configs['refer_max_tensor_len'],
                                                                    c2e_configs['batch_size'],
                                                                    seed_worker,
                                                                    g,
                                                                    False)
    
    # Save the unified_lang
    if not os.path.exists('./unified_lang.pkl'):
        with open('./unified_lang.pkl', 'wb') as f:
            pickle.dump(unified_lang, f)

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

    # Build GPT model
    from transformer import build_GPT
    model = build_GPT(unified_lang, max_tensor_len, c2e_configs)
    # model = torch.load('./models/c2e_gpt_[0729-test1].pt')
    model = model.to(eval(c2e_configs['device']))
    wandb.watch(model)

    # Build GPT trainer
    trainer = build_trainer_gpt(model, c2e_configs)
    
    # ====== Print Model Info Before Training ======
    print("\n" + "="*50)
    print("Model and Data Information:")
    print("="*50)
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vocabulary size:      {unified_lang.n_words:,}")
    print(f"Max sequence length:  {max_tensor_len}")
    
    # Batch counts
    print(f"\nDataset sizes:")
    print(f"  Train set:  {len(train_set):,} samples, {len(train_loader):,} batches")
    print(f"  Val set:    {len(val_set):,} samples, {len(val_loader):,} batches")
    print(f"  Test set:   {len(test_set):,} samples, {len(test_loader):,} batches")
    print(f"  Batch size: {c2e_configs['batch_size']}")
    
    wandb.log({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'vocab_size': unified_lang.n_words,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader)
    })
    
    print("="*50 + "\n")
    
    # Training
    clip = c2e_configs['clip_norm'] if c2e_configs.get('clip_flag', False) else None
    trainer.fit(train_loader, val_loader,
                unified_lang,
                c2e_configs['max_epochs'],
                warmup=c2e_configs['warmup'],
                # test_data=test_loader,  # Commented out: no longer monitoring test during training
                clip=clip)

    # ====== Test Set Evaluation with BLEU ======
    print("\n" + "="*50)
    print("Starting test set evaluation...")
    print("="*50)
    
    # Compute BLEU on test set with greedy decoding (consistent with training eval)
    print("Using greedy decoding (num_beams=1) for fair comparison with training eval...")
    bleu4_test_greedy = trainer.compute_bleu_on_data(
        test_loader, 
        unified_lang, 
        max_new_tokens=c2e_configs.get('max_new_tokens', 50),
        num_beams=1  # Use greedy for consistency
    )
    print('Testing BLEU-4 (greedy) =', '{:.4f}'.format(bleu4_test_greedy))
    
    # Also compute with beam search for comparison
    print("\nUsing beam search (num_beams=5) for comparison...")
    bleu4_test_beam = trainer.compute_bleu_on_data(
        test_loader, 
        unified_lang, 
        max_new_tokens=c2e_configs.get('max_new_tokens', 50),
        num_beams=c2e_configs.get('beam_size', 5)
    )
    print('Testing BLEU-4 (beam=5) =', '{:.4f}'.format(bleu4_test_beam))
    print('Beam vs Greedy diff =', '{:.4f}'.format(bleu4_test_beam - bleu4_test_greedy))
    
    wandb.log({
        'test_bleu4_greedy': bleu4_test_greedy,
        'test_bleu4_beam5': bleu4_test_beam,
        'beam_greedy_diff': bleu4_test_beam - bleu4_test_greedy
    })
    
    # ====== Example Translations ======
    print("\n" + "="*50)
    print("Example translations from test set:")
    print("="*50)
    
    from translator_gpt import GPTTranslator
    translator = GPTTranslator(model, unified_lang, device=eval(c2e_configs['device']))
    
    # Get a few examples from test set
    test_examples = []
    for inputs, targets, src_lengths in test_loader:
        for i in range(min(5, len(inputs))):  # Get up to 5 examples
            src_len = src_lengths[i].item()
            # Extract source tokens (skip BOS at position 0)
            src_tokens = inputs[i, 1:src_len+1].tolist()
            src_text = ''.join([unified_lang.index2word.get(t, '<unk>') for t in src_tokens])
            
            # Extract reference translation
            ref_tokens = targets[i, src_len:].tolist()
            ref_text = []
            for t in ref_tokens:
                if t == c2e_configs['eos_token'] or t == c2e_configs['pad_token']:
                    break
                ref_text.append(unified_lang.index2word.get(t, '<unk>'))
            ref_text = ' '.join(ref_text)
            
            test_examples.append((src_text, ref_text))
        
        if len(test_examples) >= 5:
            break
    
    # Translate and display
    for i, (src, ref) in enumerate(test_examples):
        pred = translator.translate(
            src, 
            max_new_tokens=c2e_configs.get('max_new_tokens', 50),
            num_beams=c2e_configs.get('beam_size', 5)
        )[0]
        
        print(f"\nExample {i+1}:")
        print(f"  Source:     {src}")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
    
    # ====== Custom Test Translation ======
    print("\n" + "="*50)
    print("Custom test translation:")
    print("="*50)
    
    test_sentence = "今天天气很不错，我早餐吃了一个鸡蛋和一杯牛奶。"
    print(f"Input:  {test_sentence}")
    
    # Translate using different strategies
    # Greedy
    pred_greedy = translator.translate(
        test_sentence,
        max_new_tokens=c2e_configs.get('max_new_tokens', 50),
        do_sample=False,
        num_beams=1
    )[0]
    print(f"Greedy: {pred_greedy}")
    
    # Beam search
    pred_beam = translator.translate(
        test_sentence,
        max_new_tokens=c2e_configs.get('max_new_tokens', 50),
        do_sample=False,
        num_beams=c2e_configs.get('beam_size', 5)
    )[0]
    print(f"Beam({c2e_configs.get('beam_size', 5)}): {pred_beam}")
    
    # Sampling
    pred_sample = translator.translate(
        test_sentence,
        max_new_tokens=c2e_configs.get('max_new_tokens', 50),
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )[0]
    print(f"Sample: {pred_sample}")
    
    print("\n" + "="*50)
    print("Training and evaluation completed!")
    print("="*50)


if __name__ == '__main__':
    main()
