"""
GPT Translator/Generator - Industry-standard inference implementation
Supports:
- Left padding with correct position_ids handling
- Multiple decoding strategies: greedy, beam search, sampling (top-k, top-p/nucleus)
- Batch inference
- Similar API to Hugging Face transformers

Reference: Qwen, LLaMA, GPT-2 inference implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Generation configuration similar to HuggingFace"""
    max_new_tokens: int = 50
    min_new_tokens: int = 0
    
    # Decoding strategy
    do_sample: bool = False  # If False, use greedy; if True, use sampling
    num_beams: int = 1  # beam search size, 1 means no beam search
    
    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0  # nucleus sampling, 1.0 means disabled
    
    # Repetition penalty
    repetition_penalty: float = 1.0
    
    # Early stopping
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # Length penalty for beam search
    length_penalty: float = 1.0
    
    # Return options
    return_dict_in_generate: bool = False
    output_scores: bool = False


class GPTGenerator(nn.Module):
    """
    GPT Generator for inference with left padding support
    
    Usage:
        generator = GPTGenerator(model, tokenizer_info)
        
        # Prepare inputs (left padded)
        inputs = {
            'input_ids': tensor([[PAD, PAD, tok1, tok2, tok3], ...]),
            'attention_mask': tensor([[0, 0, 1, 1, 1], ...])
        }
        
        # Generate
        outputs = generator.generate(**inputs, max_new_tokens=50)
    """
    
    def __init__(self, model, pad_token_id: int = 0, eos_token_id: int = 2, bos_token_id: int = 1):
        super(GPTGenerator, self).__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.device = model.device
        
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Prepare inputs for generation, handling left padding correctly.
        
        Args:
            input_ids: [batch_size, seq_len] - left-padded input tokens
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
            
        Returns:
            dict with input_ids, attention_mask, position_ids
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        # Generate position_ids from attention_mask (handles left padding)
        position_ids = self._make_position_ids(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    
    def _make_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create position_ids from attention_mask for left padding.
        
        For left padding:
            attention_mask: [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
            position_ids:   [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
        
        Args:
            attention_mask: [batch_size, seq_len]
        Returns:
            position_ids: [batch_size, seq_len]
        """
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)
        return position_ids
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: torch.Tensor,
        model_kwargs: dict,
        new_token: torch.Tensor
    ) -> dict:
        """Update model kwargs for the next generation step."""
        # Append new token to input_ids
        model_kwargs['input_ids'] = torch.cat([model_kwargs['input_ids'], new_token], dim=-1)
        
        # Extend attention_mask
        new_attention = torch.ones((new_token.shape[0], 1), dtype=torch.long, device=new_token.device)
        model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], new_attention], dim=-1)
        
        # Recalculate position_ids
        model_kwargs['position_ids'] = self._make_position_ids(model_kwargs['attention_mask'])
        
        return model_kwargs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids (left-padded)
            attention_mask: [batch_size, seq_len] - attention mask
            generation_config: GenerationConfig object
            max_new_tokens: maximum number of new tokens to generate
            ... (other generation parameters)
            
        Returns:
            generated_ids: [batch_size, seq_len + generated_len]
        """
        # Build generation config
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Override with explicit parameters
        if max_new_tokens is not None:
            generation_config.max_new_tokens = max_new_tokens
        if min_new_tokens is not None:
            generation_config.min_new_tokens = min_new_tokens
        if do_sample is not None:
            generation_config.do_sample = do_sample
        if num_beams is not None:
            generation_config.num_beams = num_beams
        if temperature is not None:
            generation_config.temperature = temperature
        if top_k is not None:
            generation_config.top_k = top_k
        if top_p is not None:
            generation_config.top_p = top_p
        if repetition_penalty is not None:
            generation_config.repetition_penalty = repetition_penalty
        if length_penalty is not None:
            generation_config.length_penalty = length_penalty
        if eos_token_id is not None:
            generation_config.eos_token_id = eos_token_id
        else:
            generation_config.eos_token_id = self.eos_token_id
        if pad_token_id is not None:
            generation_config.pad_token_id = pad_token_id
        else:
            generation_config.pad_token_id = self.pad_token_id
        
        # Move to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Select generation method
        if generation_config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask, generation_config)
        elif generation_config.do_sample:
            return self._sample_generate(input_ids, attention_mask, generation_config)
        else:
            return self._greedy_generate(input_ids, attention_mask, generation_config)
    
    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """Greedy decoding: always select the most probable token."""
        batch_size = input_ids.shape[0]
        
        # Prepare inputs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, attention_mask)
        
        # Track which sequences have finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=model_kwargs['input_ids'],
                attention_mask=model_kwargs['attention_mask'],
                position_ids=model_kwargs['position_ids']
            )
            
            # Get logits for the last position
            next_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, model_kwargs['input_ids'], config.repetition_penalty
                )
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Update for finished sequences
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + \
                          config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            
            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, next_tokens
            )
            
            # Check for EOS
            unfinished_sequences = unfinished_sequences * (next_tokens.squeeze(-1) != config.eos_token_id).long()
            
            # Stop if all sequences have finished
            if unfinished_sequences.sum() == 0:
                break
        
        return model_kwargs['input_ids']
    
    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """Sampling with temperature, top-k, and top-p (nucleus sampling)."""
        batch_size = input_ids.shape[0]
        
        # Prepare inputs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, attention_mask)
        
        # Track which sequences have finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=model_kwargs['input_ids'],
                attention_mask=model_kwargs['attention_mask'],
                position_ids=model_kwargs['position_ids']
            )
            
            # Get logits for the last position
            next_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, model_kwargs['input_ids'], config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Update for finished sequences
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + \
                          config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            
            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, next_tokens
            )
            
            # Check for EOS
            unfinished_sequences = unfinished_sequences * (next_tokens.squeeze(-1) != config.eos_token_id).long()
            
            # Stop if all sequences have finished
            if unfinished_sequences.sum() == 0:
                break
        
        return model_kwargs['input_ids']
    
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Beam search decoding with Hugging Face style masking.
        
        Key features:
        - All beams continue to generate until max_new_tokens (no early removal)
        - Finished beams are masked and generate PAD tokens
        - attention_mask updated to ignore PADs from finished beams
        - All sequences maintain same length (no final padding needed)
        """
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        vocab_size = self.model.vocab_size
        
        # Prepare inputs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, attention_mask)
        
        # Expand inputs for beam search: [batch_size, seq_len] -> [batch_size * num_beams, seq_len]
        model_kwargs['input_ids'] = model_kwargs['input_ids'].repeat_interleave(num_beams, dim=0)
        model_kwargs['attention_mask'] = model_kwargs['attention_mask'].repeat_interleave(num_beams, dim=0)
        model_kwargs['position_ids'] = model_kwargs['position_ids'].repeat_interleave(num_beams, dim=0)
        
        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)  # [batch_size * num_beams]
        
        # Track unfinished sequences (1 = running, 0 = finished)
        unfinished_sequences = torch.ones(batch_size * num_beams, dtype=torch.long, device=self.device)
        
        for step in range(config.max_new_tokens):
            # Forward pass (all beams, including finished ones)
            outputs = self.model(
                input_ids=model_kwargs['input_ids'],
                attention_mask=model_kwargs['attention_mask'],
                position_ids=model_kwargs['position_ids']
            )
            
            # Get logits for the last position
            next_token_logits = outputs[:, -1, :]  # [batch_size * num_beams, vocab_size]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, model_kwargs['input_ids'], config.repetition_penalty
                )
            
            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # [batch_size * num_beams, vocab_size]
            
            # Mask finished beams: set all token scores to -inf, except PAD
            next_token_scores = next_token_scores.masked_fill(
                unfinished_sequences.unsqueeze(-1) == 0,
                -float('inf')
            )
            # Force finished beams to select PAD (score = 0)
            next_token_scores[unfinished_sequences == 0, config.pad_token_id] = 0
            
            # Add beam scores (finished beams will add 0 since only PAD is non-inf)
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            
            # Reshape for beam selection: [batch_size, num_beams * vocab_size]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top num_beams candidates per batch
            next_scores, next_tokens = torch.topk(
                next_token_scores, num_beams, dim=-1, largest=True, sorted=True
            )
            
            # Decode beam_id and token_id
            next_indices = next_tokens // vocab_size  # Which beam [batch_size, num_beams]
            next_tokens = next_tokens % vocab_size     # Which token [batch_size, num_beams]
            
            # Reorder beams
            beam_indices = (torch.arange(batch_size, device=self.device).unsqueeze(-1) * num_beams + next_indices).view(-1)
            
            # Reorder model_kwargs
            model_kwargs['input_ids'] = model_kwargs['input_ids'][beam_indices]
            model_kwargs['attention_mask'] = model_kwargs['attention_mask'][beam_indices]
            
            # Append new tokens
            model_kwargs['input_ids'] = torch.cat([model_kwargs['input_ids'], next_tokens.view(-1, 1)], dim=-1)
            
            # Update unfinished_sequences: mark beams that just generated EOS
            unfinished_sequences = unfinished_sequences[beam_indices]
            eos_in_current = next_tokens.view(-1) == config.eos_token_id
            unfinished_sequences = unfinished_sequences * (~eos_in_current).long()
            
            # Update attention_mask: finished beams get 0 (ignore new PAD), unfinished get 1
            new_attention = unfinished_sequences.unsqueeze(-1)
            model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], new_attention], dim=-1)
            
            # Update position_ids based on new attention_mask
            model_kwargs['position_ids'] = self._make_position_ids(model_kwargs['attention_mask'])
            
            # Update beam_scores
            beam_scores = next_scores.view(-1)
            
            # Stop if all sequences finished
            if unfinished_sequences.sum() == 0:
                break
        
        # Select best beam for each batch item
        # All beams have same length now (no padding needed)
        final_outputs = []
        for batch_idx in range(batch_size):
            # Get the first beam of this batch (could also select by score)
            best_beam_idx = batch_idx * num_beams
            final_outputs.append(model_kwargs['input_ids'][best_beam_idx])
        
        return torch.stack(final_outputs)
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        score = torch.gather(logits, 1, input_ids)
        
        # If score < 0, then repetition penalty has to be multiplied to reduce the token probability
        score = torch.where(score < 0, score * penalty, score / penalty)
        
        logits.scatter_(1, input_ids, score)
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to keep only top-k tokens."""
        top_k = min(top_k, logits.size(-1))
        
        # Get the top-k values and indices
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[:, -1].unsqueeze(-1)
        
        # Set all logits below the minimum to -inf
        logits = torch.where(logits < min_value, torch.full_like(logits, -float('inf')), logits)
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # cumsum on CUDA doesn't have deterministic implementation
        # temporarily disable deterministic check for this operation
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        if prev_deterministic:
            torch.use_deterministic_algorithms(False)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Restore deterministic setting
        if prev_deterministic:
            torch.use_deterministic_algorithms(True)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter the mask back to original ordering
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))
        
        return logits


class GPTTranslator:
    """
    High-level translator class for Chinese to English translation using GPT
    
    Usage:
        translator = GPTTranslator(model, unified_lang)
        results = translator.translate(["你好", "世界"], max_new_tokens=50)
    """
    
    def __init__(self, model, unified_lang, device=None):
        """
        Args:
            model: GPT model
            unified_lang: Lang object with word2index and index2word
        """
        self.model = model
        self.unified_lang = unified_lang
        self.device = device if device else model.device
        self.generator = GPTGenerator(
            model,
            pad_token_id=unified_lang.word2index.get('<pad>', 0),
            eos_token_id=unified_lang.word2index.get('<eos>', 2),
            bos_token_id=unified_lang.word2index.get('<bos>', 1)
        )
        self.pad_token_id = unified_lang.word2index.get('<pad>', 0)
        self.bos_token_id = unified_lang.word2index.get('<bos>', 1)
        self.eos_token_id = unified_lang.word2index.get('<eos>', 2)
        self.unk_token_id = unified_lang.word2index.get('<unk>', 3)
    
    def tokenize(self, texts: List[str], is_cn: bool = True) -> dict:
        """
        Tokenize texts with left padding.
        
        Args:
            texts: list of input strings
            is_cn: whether the input is Chinese (character-level tokenization)
            
        Returns:
            dict with 'input_ids' and 'attention_mask'
        """
        # Tokenize each text
        token_ids_list = []
        for text in texts:
            tokens = [self.bos_token_id]  # Start with BOS
            if is_cn:
                # Character-level for Chinese
                for char in text:
                    token_id = self.unified_lang.word2index.get(char, self.unk_token_id)
                    tokens.append(token_id)
            else:
                # Word-level for English
                for word in text.split():
                    token_id = self.unified_lang.word2index.get(word, self.unk_token_id)
                    tokens.append(token_id)
            token_ids_list.append(tokens)
        
        # Find max length
        max_len = max(len(tokens) for tokens in token_ids_list)
        
        # Left padding
        input_ids = []
        attention_mask = []
        for tokens in token_ids_list:
            pad_len = max_len - len(tokens)
            padded_tokens = [self.pad_token_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            input_ids.append(padded_tokens)
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token ids to text.
        
        Args:
            token_ids: [batch_size, seq_len]
            skip_special_tokens: whether to skip special tokens
            
        Returns:
            list of decoded strings
        """
        special_tokens = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        results = []
        for seq in token_ids:
            tokens = []
            for token_id in seq.tolist():
                if skip_special_tokens and token_id in special_tokens:
                    continue
                if token_id == self.eos_token_id:
                    break
                token = self.unified_lang.index2word.get(token_id, '<unk>')
                tokens.append(token)
            
            # Join tokens (space for English words)
            text = ' '.join(tokens)
            results.append(text)
        
        return results
    
    def translate(
        self,
        texts: Union[str, List[str]],
        max_new_tokens: int = 50,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Translate Chinese texts to English.
        
        Args:
            texts: single string or list of strings
            max_new_tokens: maximum new tokens to generate
            do_sample: use sampling instead of greedy
            num_beams: beam search size
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            repetition_penalty: penalty for repeated tokens
            
        Returns:
            list of translated strings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with left padding
        inputs = self.tokenize(texts, is_cn=True)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            output_ids = self.generator.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        # Decode only the generated part (after input)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        
        return self.decode(generated_ids, skip_special_tokens=True)
    
    def translate_batch(
        self,
        src_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        **kwargs
    ) -> List[List[int]]:
        """
        Translate batch of source sequences (tensor input, similar to classic Transformer).
        
        This method mimics the classic Transformer's translate API:
        - Input: tensor [batch_size, seq_len] with LEFT-PADDED token ids
        - Output: list of lists with generated token ids (padding removed)
        
        Args:
            src_seqs: [batch_size, seq_len] tensor of LEFT-PADDED source token ids
                     Format: [PAD, PAD, ..., BOS, src_token1, src_token2, ...]
            attention_mask: [batch_size, seq_len] attention mask (1 for real tokens, 0 for padding)
            max_new_tokens: maximum tokens to generate
            num_beams: beam search size
            
        Returns:
            list of lists of token ids (without padding, BOS, EOS)
            Example: [[tok1, tok2, tok3], [tok4, tok5], ...]
            
        Example usage:
            >>> # In evaluation loop (caller should prepare left-padded input)
            >>> for inputs, targets, src_lengths in test_loader:
            >>>     src_seqs, attention_mask = prepare_left_padded_src(inputs, src_lengths)
            >>>     res = translator.translate_batch(src_seqs, attention_mask)
            >>>     bleu4 = corpus_bleu(refer, res, weights=(0.25, 0.25, 0.25, 0.25))
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            src_seqs = src_seqs.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Generate (input is already left-padded)
            generated_ids = self.generator.generate(
                input_ids=src_seqs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                **kwargs
            )
            # Extract generated part (after input)
            # Since beam search now keeps all sequences at same length, input_len is fixed
            input_len = src_seqs.shape[1]
            generated_tokens = generated_ids[:, input_len:]  # [batch_size, generated_len]
            
            # Convert to list of lists (remove padding and special tokens)
            batch_size = generated_tokens.shape[0]
            res = []
            for i in range(batch_size):
                token_list = generated_tokens[i].tolist()
                
                # Collect tokens until EOS or PAD
                tokens = []
                for token_id in token_list:
                    if token_id == self.pad_token_id or token_id == self.eos_token_id:
                        break
                    if token_id == self.bos_token_id:  # Skip BOS if any
                        continue
                    tokens.append(token_id)
                res.append(tokens)
            
            return res


# Convenience function for quick setup
def build_gpt_translator(model, unified_lang, device=None):
    """Build a GPT translator from model and language object."""
    return GPTTranslator(model, unified_lang, device)


if __name__ == "__main__":
    # Example usage
    print("GPT Translator module loaded successfully.")
    print("\nExample usage:")
    print("""
    from transformer import GPT, build_GPT
    from translator_gpt import GPTTranslator, GPTGenerator
    
    # Load model
    model = build_GPT(unified_lang, max_seq_len, configs)
    
    # Create translator
    translator = GPTTranslator(model, unified_lang)
    
    # Translate
    results = translator.translate(
        ["你好世界", "今天天气很好"],
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    print(results)
    
    # Or use low-level generator directly
    generator = GPTGenerator(model, pad_token_id=0, eos_token_id=2)
    
    inputs = {
        'input_ids': torch.tensor([[0, 0, 1, 100, 200]]),  # left-padded
        'attention_mask': torch.tensor([[0, 0, 1, 1, 1]])
    }
    
    outputs = generator.generate(**inputs, max_new_tokens=50)
    """)
