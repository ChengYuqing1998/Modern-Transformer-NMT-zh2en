import pickle
import tqdm
import wandb
import os
from torch.optim import lr_scheduler
import logging
from tqdm import tqdm
from torch.nn.functional import log_softmax
import torch
import torch.nn as nn
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from translator_gpt import GPTGenerator
from translator_gpt import GPTTranslator


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def build_trainer_gpt(model, configs):
    trainer = GPTTrainer(model=model,
                        ckpt_dir=configs['ckpt_dir'],
                        ckpt_file_name=configs['ckpt_file_name'],
                        log_dir=configs['log_dir'],
                        log_file_name=configs['log_file_name'],
                        print_freq=configs['print_freq'],
                        eval_freq=configs['eval_freq'],
                        save_freq=configs['save_freq'],
                        device=eval(configs['device']),
                        write_config=configs
                        )
    trainer.build_loss(configs['loss_type'], configs['smoothing'], configs['ignore_index'])
    
    trainer.build_optimizer(configs['learning_rate'], configs['optimizer_type'])
    
    if configs['scheduler_flag']:
        trainer.build_scheduler(configs['anneal_rate'],
                               configs['scheduler_type'],
                               configs['patience'],
                               configs['threshold'])
    return trainer


def remove_element(lst, element):
    if isinstance(lst, list):
        return [remove_element(sublst, element) for sublst in lst if sublst != element]
    else:
        return lst if lst != element else None


class GPTTrainer:
    def __init__(self, model: nn.Module, ckpt_dir, ckpt_file_name, log_dir, log_file_name,
                 print_freq, eval_freq, save_freq, device, write_config, **kwargs):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_file_name = ckpt_file_name
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.device = device
        self.write_config = write_config

        self.optimizer = None
        self.loss = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = 1e9
        
        # GPT Generator for inference (low-level)
        self.generator = GPTGenerator(
            model,
            pad_token_id=write_config['pad_token'],
            eos_token_id=write_config['eos_token'],
            bos_token_id=write_config['bos_token']
        )
        
        # Translator will be initialized lazily when needed
        self.translator = None

        self.log = self.make_log(log_dir, log_file_name)
        self.log.info(msg=self.write_config)

    def build_optimizer(self, learning_rate, optimizer_type, **kwargs):
        assert optimizer_type in ['sgd', 'adam']
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def build_scheduler(self, anneal_rate, scheduler_type, patience, threshold):
        assert scheduler_type in ['exp', 'plateau', 'cosine']
        if scheduler_type == 'exp':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, anneal_rate)
        elif scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='min',
                                                            patience=patience,
                                                            factor=anneal_rate,
                                                            threshold=threshold,
                                                            threshold_mode='abs')
        elif scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=self.write_config['max_epochs'],
                                                            eta_min=self.write_config['learning_rate'] * 1e-2)

    def get_translator(self, unified_lang):
        """
        Get or create translator (lazy initialization to avoid creating it if not needed)
        """
        if self.translator is None:
            self.translator = GPTTranslator(self.model, unified_lang, device=self.device)
        return self.translator
    
    def build_loss(self, loss_type, smoothing, ignore_index=None, **kwargs):
        assert loss_type in ['ce', 'nll', 'kl']
        # For GPT, use unified vocab_size
        vocab_size = getattr(self.model, 'vocab_size', None)
        if vocab_size is None:
            # Try to get from generator if available
            if hasattr(self.model, 'generator'):
                vocab_size = self.model.generator.proj.out_features
            else:
                raise ValueError("Cannot determine vocab_size for GPT model")
        
        if loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))
        elif loss_type == 'nll':
            self.loss = torch.nn.NLLLoss(ignore_index=int(ignore_index))
        elif loss_type == 'kl':
            self.loss = LabelSmoothing(vocab_size, int(ignore_index), smoothing)

    def fit(self, train_data, val_data, unified_lang, max_epochs, warmup, test_data=None, clip=None, dict_flag=False, **kwargs):
        """
        Train GPT model
        train_data: DataLoader that returns (inputs, targets, src_lengths)
        val_data: validation DataLoader
        test_data: test DataLoader (optional, for monitoring test BLEU during training)
        unified_lang: unified language object for both source and target
        """
        self.model.train()
        for epoch in tqdm(range(1, max_epochs+1)):
            train_loss_in_epoch = []
            for inputs, targets, src_lengths in train_data:
                loss = self.fit_iter(inputs, targets, src_lengths, clip=clip)
                train_loss_in_epoch.append(loss)
                self.log.info(msg=('Epoch:', '%04d' % epoch, 'Training Loss =', '{:.6f}'.format(loss)))
                wandb.log({'epoch': epoch, 'train_loss': loss})
                if self.global_step % self.print_freq == 0:
                    avg_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
                    print('Epoch:', '%04d' % epoch, 'Average Training Loss =', '{:.6f}'.format(avg_loss))

            if epoch == 1:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print("total_para_counts: ", total_params)
                wandb.log({'total_para_counts': total_params})

            if epoch % self.save_freq == 0:
                self._save(self.model, self.ckpt_dir, self.ckpt_file_name, dict_flag=dict_flag)

            if epoch % self.eval_freq == 0:
                # Evaluate on validation set
                eval_loss, eval_bleu4 = self.eval(val_data, unified_lang)
                self.log.info(msg=('Epoch:', '%04d' % epoch, 'Val Loss =', '{:.6f}'.format(eval_loss),
                                   'Val Bleu4 =', '{:.6f}'.format(eval_bleu4)))
                
                # # Evaluate on test set if provided (commented out for faster training)
                # if test_data is not None:
                #     test_bleu4 = self.compute_bleu_on_data(test_data, unified_lang, max_new_tokens=50, num_beams=5)
                #     self.log.info(msg=('Epoch:', '%04d' % epoch, 'Test Bleu4 =', '{:.6f}'.format(test_bleu4)))
                #     
                #     # Print both val and test BLEU4 together for comparison
                #     print('Epoch:', '%04d' % epoch, 
                #           'Val Loss =', '{:.6f}'.format(eval_loss),
                #           '| Val Bleu4 =', '{:.6f}'.format(eval_bleu4),
                #           '| Test Bleu4 =', '{:.6f}'.format(test_bleu4),
                #           '| Diff =', '{:.6f}'.format(abs(eval_bleu4 - test_bleu4)))
                #     
                #     # Log to wandb
                #     wandb.log({
                #         'epoch': epoch, 
                #         'val_loss': eval_loss, 
                #         'val_bleu4': eval_bleu4,
                #         'test_bleu4': test_bleu4,
                #         'bleu4_diff': abs(eval_bleu4 - test_bleu4)
                #     })
                # else:
                print('Epoch:', '%04d' % epoch, 'Val Loss =', '{:.6f}'.format(eval_loss),
                      'Val Bleu4 =', '{:.6f}'.format(eval_bleu4))
                wandb.log({'epoch': epoch, 'val_loss': eval_loss, 'val_bleu4': eval_bleu4})
                
                if eval_loss < self.best_loss:
                    last_file_name = 'intermediate_model' + '-' + self.ckpt_file_name + '-{:.4f}.pt'.format(
                        self.best_loss)
                    last_file_path = os.path.join(self.ckpt_dir + '/intermediate', last_file_name)
                    if os.path.exists(last_file_path):
                        os.remove(last_file_path)
                    self.best_loss = eval_loss
                    self._save(self.model, self.ckpt_dir + '/intermediate',
                               'intermediate_model' + '-' + self.ckpt_file_name + '-{:.4f}.pt'.format(eval_loss),
                               dict_flag=True)

            if self.scheduler and epoch > warmup and self.eval_freq == 1:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

    def fit_iter(self, inputs, targets, src_lengths, clip=None):
        """
        Single training iteration for GPT
        inputs: [batch_size, seq_len] - [BOS, src, trg, EOS, PAD, ...]
        targets: [batch_size, seq_len] - [src, trg, EOS, PAD, ...] (shifted by 1)
        src_lengths: [batch_size] - length of src part (excluding BOS)
        """
        self.global_step += 1
        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        src_lengths = src_lengths.to(self.device)
        
        # GPT forward: only needs inputs
        logits = self.model.forward(inputs)  # [batch_size, seq_len, vocab_size]
        
        # Apply log_softmax for NLL loss, or use raw logits for CE loss
        if isinstance(self.loss, torch.nn.NLLLoss):
            logits = log_softmax(logits, dim=-1)
        
        # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
        logits_flat = logits.contiguous().view(-1, logits.shape[-1])
        targets_flat = targets.contiguous().view(-1).clone()
        
        # Vectorized masking: mask src part in targets
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        ignore_index = self.loss.ignore_index
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
        
        # Expand src_lengths to [batch_size, seq_len] for comparison
        src_lengths_expanded = src_lengths.unsqueeze(1)  # [batch_size, 1]
        
        # Create mask: True where position < src_length (src part), False otherwise
        src_mask = pos_indices < src_lengths_expanded  # [batch_size, seq_len]
        
        # Apply mask to targets_flat by reshaping and applying
        targets_reshaped = targets_flat.view(batch_size, seq_len)
        targets_reshaped = targets_reshaped.masked_fill(src_mask, ignore_index)
        targets_flat = targets_reshaped.view(-1)
        
        loss = self.loss(logits_flat, targets_flat)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        wandb.log({'global_step': self.global_step})
        wandb.log({'learning_rate': lr})
        return loss

    def _prepare_left_padded_src(self, inputs, src_lengths, batch_size):
        """
        Prepare left-padded source sequences for GPT inference.
        
        Args:
            inputs: [batch_size, seq_len] right-padded input sequences
            src_lengths: [batch_size] actual source lengths (without BOS)
            batch_size: batch size
            
        Returns:
            src_seqs: [batch_size, max_src_len] left-padded sequences
            attention_mask: [batch_size, max_src_len] attention mask
        """
        pad_idx = self.write_config['pad_token']
        max_src_len = src_lengths.max().item() + 1  # +1 for BOS
        
        # Create position indices
        positions = torch.arange(max_src_len, device=self.device).unsqueeze(0)  # [1, max_src_len]
        src_lengths_expanded = (src_lengths + 1).unsqueeze(1)  # [batch_size, 1], +1 for BOS
        
        # Calculate offset for left padding
        offsets = max_src_len - src_lengths_expanded  # [batch_size, 1]
        src_mask = (positions >= offsets) & (positions < max_src_len)  # [batch_size, max_src_len]
        
        # Calculate source positions to read from
        read_positions = positions - offsets  # [batch_size, max_src_len]
        read_positions = torch.clamp(read_positions, min=0, max=inputs.shape[1]-1)
        
        # Create left-padded sequences
        src_seqs = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long, device=self.device)
        src_seqs[src_mask] = inputs.gather(1, read_positions)[src_mask]
        
        # Create attention mask
        attention_mask = src_mask.long()
        
        return src_seqs, attention_mask
    
    def _extract_references(self, targets, src_lengths, batch_size):
        """
        Extract reference translations from targets.
        
        Args:
            targets: [batch_size, seq_len] target sequences
            src_lengths: [batch_size] source lengths
            batch_size: batch size
            
        Returns:
            refer: list of reference token lists (format for corpus_bleu)
        """
        pad_idx = self.write_config['pad_token']
        eos_idx = self.write_config['eos_token']
        
        seq_len = targets.shape[1]
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
        src_lengths_expanded = src_lengths.unsqueeze(1)  # [batch_size, 1]
        ref_mask = positions >= src_lengths_expanded  # [batch_size, seq_len]
        
        # Apply mask and convert to list
        refer = []
        for i in range(batch_size):
            ref_tokens = targets[i][ref_mask[i]].tolist()
            # Remove padding and EOS
            ref_clean = [t for t in ref_tokens if t != pad_idx and t != eos_idx]
            refer.append([ref_clean])  # Wrap in list for corpus_bleu
        
        return refer

    def eval(self, val_data, unified_lang, compute_bleu=True, max_new_tokens=50, num_beams=5):
        """
        Evaluate GPT model with loss and BLEU calculation (simplified version)
        """
        self.model.eval()
        translator = self.get_translator(unified_lang)
        validate_loss = 0.0
        bleu4 = 0.0
        sample_num = 0
        
        with torch.no_grad():
            for inputs, targets, src_lengths in tqdm(val_data, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                src_lengths = src_lengths.to(self.device)
                batch_size = len(inputs)
                sample_num += batch_size
                
                # ====== Loss Calculation ======
                logits = self.model.forward(inputs)
                
                if isinstance(self.loss, torch.nn.NLLLoss):
                    logits = log_softmax(logits, dim=-1)
                
                logits_flat = logits.contiguous().view(-1, logits.shape[-1])
                targets_flat = targets.contiguous().view(-1).clone()
                
                # Mask src part in targets
                seq_len = inputs.shape[1]
                ignore_index = self.loss.ignore_index
                pos_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)
                src_lengths_expanded = src_lengths.unsqueeze(1)
                src_mask = pos_indices < src_lengths_expanded
                targets_reshaped = targets_flat.view(batch_size, seq_len)
                targets_reshaped = targets_reshaped.masked_fill(src_mask, ignore_index)
                targets_flat = targets_reshaped.view(-1)
                
                loss = self.loss(logits_flat, targets_flat)
                validate_loss += loss.item() * batch_size
                
                # ====== BLEU Calculation (simplified like classic Transformer) ======
                if compute_bleu:
                    # Prepare left-padded source sequences
                    src_seqs, attention_mask = self._prepare_left_padded_src(inputs, src_lengths, batch_size)
                    
                    # Generate translations
                    res = translator.translate_batch(src_seqs, attention_mask, max_new_tokens=max_new_tokens, num_beams=num_beams)
                    
                    # Extract references
                    refer = self._extract_references(targets, src_lengths, batch_size)
                    
                    # Print first sample for debugging (only once)
                    # if sample_num == batch_size:  # First batch
                    #     print("\n" + "="*80)
                    #     print("【调试信息】第一个样本详情:")
                    #     print("="*80)
                        
                    #     # Helper function to decode token IDs with special tokens
                    #     def decode_with_special(token_ids):
                    #         tokens = []
                    #         for tid in token_ids:
                    #             if tid == self.write_config['pad_token']:
                    #                 tokens.append('<PAD>')
                    #             elif tid == self.write_config['bos_token']:
                    #                 tokens.append('<BOS>')
                    #             elif tid == self.write_config['eos_token']:
                    #                 tokens.append('<EOS>')
                    #             else:
                    #                 word = translator.unified_lang.index2word.get(tid, '<UNK>')
                    #                 tokens.append(word)
                    #         return tokens
                        
                    #     # Print original input (right-padded)
                    #     print("\n0. 原始输入序列 (右padding, 完整):")
                    #     print(f"   张量形状: {inputs[0].shape}")
                    #     print(f"   Token IDs: {inputs[0].tolist()}")
                    #     input_tokens = decode_with_special(inputs[0].tolist())
                    #     print(f"   文本表示: {' '.join(input_tokens)}")
                    #     print(f"   src_length: {src_lengths[0].item()} (不含BOS)")
                        
                    #     # Print source sequence (left-padded)
                    #     print("\n1. 提取的源序列 (左padding, 用于推理):")
                    #     print(f"   张量形状: {src_seqs[0].shape}")
                    #     print(f"   Token IDs: {src_seqs[0].tolist()}")
                    #     src_tokens = decode_with_special(src_seqs[0].tolist())
                    #     print(f"   文本表示: {' '.join(src_tokens)}")
                    #     print(f"   attention_mask: {attention_mask[0].tolist()}")
                        
                    #     # Print reference (target part only, already cleaned)
                    #     print("\n2. 参考翻译 (仅target部分, 已去除PAD和EOS):")
                    #     print(f"   Token IDs: {refer[0][0]}")  # refer[0][0] because of corpus_bleu format
                    #     ref_tokens = decode_with_special(refer[0][0])
                    #     print(f"   文本表示: {' '.join(ref_tokens)}")
                        
                    #     # Print generated translation (already cleaned)
                    #     print("\n3. 生成翻译 (已去除特殊token):")
                    #     print(f"   Token IDs: {res[0]}")
                    #     gen_tokens = decode_with_special(res[0])
                    #     print(f"   文本表示: {' '.join(gen_tokens)}")
                    #     print("="*80 + "\n")
                    
                    # Calculate BLEU
                    bleu4_batch = nltk.translate.bleu_score.corpus_bleu(refer, res, weights=(0.25, 0.25, 0.25, 0.25))
                    bleu4 += bleu4_batch * batch_size
        
        # ====== Test Translation ======
        test_sentence = "今天天气很不错，我早餐吃了一个鸡蛋和一杯牛奶。"
        test_output = translator.translate(test_sentence, max_new_tokens=max_new_tokens, num_beams=num_beams)[0]
        print(f"Test input: {test_sentence}")
        print(f"Test output: {test_output}")
        
        self.model.train()
        return validate_loss / sample_num, bleu4 / sample_num
    
    def compute_bleu_on_data(self, data_loader, unified_lang, max_new_tokens=50, num_beams=5):
        """
        Compute BLEU score on a dataset (simplified like classic Transformer)
        """
        self.model.eval()
        translator = self.get_translator(unified_lang)
        
        bleu4 = 0.0
        sample_num = 0
        
        with torch.no_grad():
            for inputs, targets, src_lengths in tqdm(data_loader, desc="Computing BLEU"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                src_lengths = src_lengths.to(self.device)
                batch_size = len(inputs)
                sample_num += batch_size
                
                # Prepare left-padded source sequences
                src_seqs, attention_mask = self._prepare_left_padded_src(inputs, src_lengths, batch_size)
                
                # Generate translations
                res = translator.translate_batch(src_seqs, attention_mask, max_new_tokens=max_new_tokens, num_beams=num_beams)
                
                # Extract references
                refer = self._extract_references(targets, src_lengths, batch_size)
                
                # Calculate BLEU
                bleu4_batch = nltk.translate.bleu_score.corpus_bleu(refer, res, weights=(0.25, 0.25, 0.25, 0.25))
                bleu4 += bleu4_batch * batch_size
        
        self.model.train()
        return bleu4 / sample_num

    def _save(self, model, ckpt_dir, ckpt_file_name, dict_flag=False):
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        save_path = os.path.join(ckpt_dir, ckpt_file_name)
        if not dict_flag:
            torch.save(model, save_path)
        else:
            torch.save(model.state_dict(), save_path)

    @staticmethod
    def make_log(log_dir, log_file_name):
        path = os.path.join(log_dir, log_file_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if os.path.exists(path):
            mode = 'a'
        else:
            mode = 'w'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            filename=path,
                            filemode=mode)
        return logging
