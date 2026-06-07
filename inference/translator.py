"""Inference utilities for encoder-decoder and decoder-only translation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass, replace

from models.transformer import Transformer


@dataclass(frozen=True)
class EncoderDecoderGenerationConfig:
    strategy: str
    max_new_tokens: int
    beam_size: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    use_kv_cache: bool


class EncoderDecoderTranslator(nn.Module):
    """Encoder-decoder translation with beam, greedy, and nucleus decoding."""

    def __init__(
        self,
        model,
        beam_size=5,
        max_seq_len=None,
        device=None,
        use_kv_cache=True,
        decoding_strategy="beam_search",
        temperature=0.8,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        inference_max_new_tokens=160,
    ):
        super().__init__()
        self.model = model
        self.alpha = 0.7
        self.beam_size = beam_size
        self.use_kv_cache = use_kv_cache
        self.decoding_strategy = decoding_strategy
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.inference_max_new_tokens = int(inference_max_new_tokens)
        self.max_seq_len = (
            max_seq_len
            or getattr(model, "max_trg_len", None)
            or getattr(model, "max_len", None)
        )
        if self.max_seq_len is None:
            raise ValueError(
                "EncoderDecoderTranslator requires a target-side max length."
            )
        self.trg_bos_idx = model.trg_bos_idx
        self.trg_eos_idx = model.trg_eos_idx
        self.src_pad_idx = model.src_pad_idx
        self.trg_pad_idx = model.trg_pad_idx
        self.device = device or model.device
        self.register_buffer(
            "init_seq",
            torch.tensor([[self.trg_bos_idx]], dtype=torch.long, device=self.device),
        )

    def _model_decoder_logits(
        self,
        trg_seq,
        enc_output,
        src_mask,
        trg_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        dec_output = self.model.decoder(
            trg_seq,
            enc_output,
            src_mask,
            trg_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            dec_output, present_key_values = dec_output
            return self.model.generator(dec_output), present_key_values
        return self.model.generator(dec_output), None

    @staticmethod
    def _apply_repetition_penalty(logits, generated, penalty):
        scores = torch.gather(logits, 1, generated)
        scores = torch.where(
            scores < 0, scores * penalty, scores / penalty
        )
        return logits.scatter(1, generated, scores)

    @staticmethod
    def _top_k_filtering(logits, top_k):
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        return logits.masked_fill(logits < threshold, -float("inf"))

    @staticmethod
    def _top_p_filtering(logits, top_p):
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        remove = cumulative_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = remove.scatter(1, sorted_indices, remove)
        return logits.masked_fill(remove, -float("inf"))

    def _sample_next_token(
        self,
        logits,
        generated,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    ):
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
        logits = logits / temperature
        if top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        if top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)
        return torch.multinomial(
            F.softmax(logits, dim=-1), num_samples=1
        )

    def _decode_autoregressive(self, src_seqs, src_mask, config):
        batch_size = src_seqs.size(0)
        enc_outputs = self.model.encoder(src_seqs, src_mask)
        generated = self.init_seq.expand(batch_size, -1).clone()
        unfinished = torch.ones(
            batch_size, dtype=torch.bool, device=self.device
        )
        past_key_values = None

        for _ in range(config.max_new_tokens):
            if config.use_kv_cache and past_key_values is not None:
                decoder_input = generated[:, -1:]
                trg_mask = None
                position_ids = torch.full(
                    (batch_size, 1),
                    generated.size(1) - 1,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                decoder_input = generated
                trg_mask = Transformer.make_no_peak_mask(
                    decoder_input, device=self.device
                )
                position_ids = None

            logits, past_key_values = self._model_decoder_logits(
                decoder_input,
                enc_outputs,
                src_mask,
                trg_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=config.use_kv_cache,
            )
            next_tokens = self._select_next_token(
                logits[:, -1, :], generated, config
            )
            next_tokens = torch.where(
                unfinished.unsqueeze(-1),
                next_tokens,
                torch.full_like(next_tokens, self.trg_pad_idx),
            )
            generated = torch.cat((generated, next_tokens), dim=1)
            unfinished &= next_tokens.squeeze(-1) != self.trg_eos_idx
            if not unfinished.any():
                break

        results = []
        for sequence in generated:
            tokens = sequence.tolist()
            if self.trg_eos_idx in tokens:
                tokens = tokens[:tokens.index(self.trg_eos_idx) + 1]
            results.append(tokens)
        return results

    def _select_next_token(self, logits, generated, config):
        if config.strategy == "nucleus_sampling":
            return self._sample_next_token(
                logits,
                generated,
                config.temperature,
                config.top_p,
                config.top_k,
                config.repetition_penalty,
            )
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, generated, config.repetition_penalty
            )
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _new_beam_buffer(self, batch_size, beam_size, sequence_length):
        generated = torch.full(
            (batch_size * beam_size, sequence_length),
            self.trg_pad_idx,
            dtype=torch.long,
            device=self.device,
        )
        generated[:, 0] = self.trg_bos_idx
        return generated

    def _initialize_beams(
        self, src_seqs, src_mask, trg_mask, config, sequence_length
    ):
        batch_size = src_seqs.size(0)
        enc_outputs = self.model.encoder(src_seqs, src_mask)
        logits, _ = self._model_decoder_logits(
            self.init_seq.expand(batch_size, -1),
            enc_outputs,
            src_mask,
            trg_mask,
        )
        next_token_logits = logits[:, -1, :]
        if config.repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits,
                self.init_seq.expand(batch_size, -1),
                config.repetition_penalty,
            )
        scores, best_indices = F.log_softmax(
            next_token_logits, dim=-1
        ).topk(config.beam_size)
        generated = self._new_beam_buffer(
            batch_size, config.beam_size, sequence_length
        )
        generated[:, 1] = best_indices.reshape(-1)
        finished = generated[:, 1] == self.trg_eos_idx
        return (
            enc_outputs.repeat_interleave(config.beam_size, dim=0),
            generated,
            scores,
            finished,
        )

    def _select_beams(
        self,
        generated,
        logits,
        scores,
        finished,
        step,
        config,
        return_indices=False,
    ):
        beam_size = config.beam_size
        batch_size = generated.size(0) // beam_size
        vocab_size = logits.size(-1)
        next_token_logits = logits[:, -1, :]
        if config.repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits,
                generated[:, :step],
                config.repetition_penalty,
            )
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = next_token_scores.masked_fill(
            finished.unsqueeze(-1), -float("inf")
        )
        next_token_scores[finished, self.trg_pad_idx] = 0.0
        candidate_scores = next_token_scores + scores.reshape(-1, 1)
        scores, candidate_indices = candidate_scores.view(batch_size, -1).topk(
            beam_size
        )
        beam_indices = torch.div(
            candidate_indices, vocab_size, rounding_mode="floor"
        )
        next_tokens = torch.remainder(candidate_indices, vocab_size).reshape(-1)
        global_beam_indices = (
            torch.arange(batch_size, device=self.device).unsqueeze(1)
            * beam_size
            + beam_indices
        ).reshape(-1)
        generated = generated.index_select(0, global_beam_indices)
        parent_finished = finished.index_select(0, global_beam_indices)
        next_tokens = torch.where(
            parent_finished,
            torch.full_like(next_tokens, self.trg_pad_idx),
            next_tokens,
        )
        generated[:, step] = next_tokens
        finished = parent_finished | (next_tokens == self.trg_eos_idx)
        if not return_indices:
            return generated, scores, finished
        return generated, scores, finished, global_beam_indices

    @staticmethod
    def _repeat_cache(past_key_values, repeats):
        return tuple(
            (
                (
                    self_key.repeat_interleave(repeats, dim=0),
                    self_value.repeat_interleave(repeats, dim=0),
                ),
                (
                    cross_key.repeat_interleave(repeats, dim=0),
                    cross_value.repeat_interleave(repeats, dim=0),
                ),
            )
            for (
                (self_key, self_value),
                (cross_key, cross_value),
            ) in past_key_values
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_indices):
        return tuple(
            (
                (
                    self_key.index_select(0, beam_indices),
                    self_value.index_select(0, beam_indices),
                ),
                (
                    cross_key.index_select(0, beam_indices),
                    cross_value.index_select(0, beam_indices),
                ),
            )
            for (
                (self_key, self_value),
                (cross_key, cross_value),
            ) in past_key_values
        )

    def _beam_search_with_cache(
        self, src_seqs, src_mask, config, sequence_length
    ):
        beam_size = config.beam_size
        batch_size = src_seqs.size(0)
        enc_outputs = self.model.encoder(src_seqs, src_mask)
        init_tokens = self.init_seq.expand(batch_size, -1)
        position_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        logits, past_key_values = self._model_decoder_logits(
            init_tokens,
            enc_outputs,
            src_mask,
            trg_mask=None,
            position_ids=position_ids,
            use_cache=True,
        )
        next_token_logits = logits[:, -1, :]
        if config.repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits,
                init_tokens,
                config.repetition_penalty,
            )
        scores, best_indices = F.log_softmax(
            next_token_logits, dim=-1
        ).topk(beam_size)
        generated = self._new_beam_buffer(
            batch_size, beam_size, sequence_length
        )
        generated[:, 1] = best_indices.reshape(-1)
        finished = generated[:, 1] == self.trg_eos_idx
        enc_outputs = enc_outputs.repeat_interleave(
            beam_size, dim=0
        )
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)
        past_key_values = self._repeat_cache(
            past_key_values, beam_size
        )

        eos_locations = generated == self.trg_eos_idx
        sequence_lengths = self._sequence_lengths(
            eos_locations, sequence_length
        )

        for step in range(2, sequence_length):
            current_tokens = generated[:, step - 1:step]
            position_ids = torch.full(
                (current_tokens.size(0), 1),
                step - 1,
                dtype=torch.long,
                device=self.device,
            )
            logits, past_key_values = self._model_decoder_logits(
                current_tokens,
                enc_outputs,
                src_mask,
                trg_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            generated, scores, finished, beam_indices = self._select_beams(
                generated,
                logits,
                scores,
                finished,
                step,
                config,
                return_indices=True,
            )
            enc_outputs = enc_outputs.index_select(0, beam_indices)
            src_mask = src_mask.index_select(0, beam_indices)
            past_key_values = self._reorder_cache(
                past_key_values, beam_indices
            )
            eos_locations = generated == self.trg_eos_idx
            sequence_lengths = self._sequence_lengths(
                eos_locations, sequence_length
            )
            if finished.view(batch_size, beam_size).all(dim=1).all():
                break

        return self._finalize_results(
            generated, scores, sequence_lengths, batch_size, beam_size
        )

    def _sequence_lengths(self, eos_locations, sequence_length):
        length_map = torch.arange(
            1,
            sequence_length + 1,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        return length_map.masked_fill(
            ~eos_locations, sequence_length
        ).min(1).values

    def _finalize_results(
        self, generated, scores, sequence_lengths, batch_size, beam_size
    ):
        normalized_scores = scores / (
            sequence_lengths.view(batch_size, -1).float() ** self.alpha
        )
        answer_indices = normalized_scores.argmax(dim=1)
        generated = generated.view(batch_size, beam_size, -1)
        sequence_lengths = sequence_lengths.view(batch_size, beam_size)
        results = []
        for batch_index, beam_index in enumerate(answer_indices.tolist()):
            length = sequence_lengths[batch_index, beam_index].item()
            results.append(
                generated[batch_index, beam_index, :length].tolist()
            )
        return results

    @torch.no_grad()
    def translate(
        self,
        src_seqs,
        max_new_tokens=None,
        decoding_strategy=None,
        num_beams=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
    ):
        if src_seqs.ndim != 2 or src_seqs.size(0) == 0:
            raise ValueError(
                "src_seqs must be a non-empty [batch, sequence] tensor"
            )
        max_src_len = getattr(self.model, "max_src_len", None)
        if max_src_len is not None and src_seqs.size(1) > max_src_len:
            raise ValueError(
                "padded source width must not exceed max_src_len "
                f"({src_seqs.size(1)} > {max_src_len})"
            )
        src_seqs = src_seqs.to(self.device)
        src_mask = Transformer.make_pad_mask(
            src_seqs, self.src_pad_idx, device=self.device
        )
        config = self._resolve_generation_config(
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if config.strategy == "beam_search":
            return self._beam_search(src_seqs, src_mask, config)
        return self._decode_autoregressive(src_seqs, src_mask, config)

    def _resolve_generation_config(
        self,
        max_new_tokens,
        decoding_strategy,
        num_beams,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    ):
        strategy = decoding_strategy or self.decoding_strategy
        if strategy not in ("greedy", "beam_search", "nucleus_sampling"):
            raise ValueError(
                "decoding_strategy must be 'greedy', 'beam_search', "
                "or 'nucleus_sampling'"
            )
        requested_tokens = (
            self.inference_max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        token_limit = min(self.max_seq_len - 1, requested_tokens)
        beam_size = self.beam_size if num_beams is None else int(num_beams)
        temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        top_p = self.top_p if top_p is None else float(top_p)
        top_k = self.top_k if top_k is None else int(top_k)
        repetition_penalty = (
            self.repetition_penalty
            if repetition_penalty is None
            else float(repetition_penalty)
        )
        if token_limit < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if strategy == "beam_search" and beam_size < 2:
            raise ValueError(
                "beam_search requires num_beams to be at least 2"
            )
        target_vocab_size = self.model.generator.proj.out_features
        if strategy == "beam_search" and beam_size > target_vocab_size:
            raise ValueError(
                "num_beams must not exceed the target vocabulary size"
            )
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")
        return EncoderDecoderGenerationConfig(
            strategy=strategy,
            max_new_tokens=token_limit,
            beam_size=beam_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_kv_cache=self.use_kv_cache,
        )

    def _beam_search(self, src_seqs, src_mask, config):
        sequence_length = config.max_new_tokens + 1
        if config.use_kv_cache:
            return self._beam_search_with_cache(
                src_seqs,
                src_mask,
                config,
                sequence_length,
            )

        batch_size = src_seqs.size(0)
        trg_mask = Transformer.make_no_peak_mask(
            self.init_seq, device=self.device
        )
        enc_outputs, generated, scores, finished = self._initialize_beams(
            src_seqs,
            src_mask,
            trg_mask,
            config,
            sequence_length,
        )
        src_mask = src_mask.repeat_interleave(config.beam_size, dim=0)

        eos_locations = generated == self.trg_eos_idx
        sequence_lengths = self._sequence_lengths(
            eos_locations, sequence_length
        )
        for step in range(2, sequence_length):
            trg_mask = Transformer.make_no_peak_mask(
                generated[:, :step], device=self.device
            )
            logits, _ = self._model_decoder_logits(
                generated[:, :step], enc_outputs, src_mask, trg_mask
            )
            generated, scores, finished = self._select_beams(
                generated,
                logits,
                scores,
                finished,
                step,
                config,
            )
            eos_locations = generated == self.trg_eos_idx
            sequence_lengths = self._sequence_lengths(
                eos_locations, sequence_length
            )
            if finished.view(
                batch_size, config.beam_size
            ).all(dim=1).all():
                break

        return self._finalize_results(
            generated,
            scores,
            sequence_lengths,
            batch_size,
            config.beam_size,
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Resolved decoder-only generation settings."""

    max_new_tokens: int = 50
    min_new_tokens: int = 0
    decoding_strategy: str = "greedy"
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    length_penalty: float = 1.0
    use_kv_cache: bool = True


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
        model_kwargs: dict,
        new_token: torch.Tensor,
        unfinished_sequences: Optional[torch.Tensor] = None,
    ) -> dict:
        """Update model kwargs for the next generation step."""
        model_kwargs['input_ids'] = torch.cat([model_kwargs['input_ids'], new_token], dim=-1)
        if unfinished_sequences is None:
            new_attention = torch.ones(
                (new_token.shape[0], 1),
                dtype=torch.long,
                device=new_token.device,
            )
        else:
            new_attention = unfinished_sequences.unsqueeze(-1).to(
                dtype=torch.long
            )
        model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], new_attention], dim=-1)
        model_kwargs['position_ids'] = self._make_position_ids(model_kwargs['attention_mask'])
        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_indices):
        if past_key_values is None:
            return None
        return tuple(
            (
                key.index_select(0, beam_indices),
                value.index_select(0, beam_indices),
            )
            for key, value in past_key_values
        )

    def _model_step(
        self,
        model_kwargs,
        use_kv_cache,
        past_key_values=None,
    ):
        if use_kv_cache and past_key_values is not None:
            input_ids = model_kwargs["input_ids"][:, -1:]
            position_ids = model_kwargs["position_ids"][:, -1:]
        else:
            input_ids = model_kwargs["input_ids"]
            position_ids = model_kwargs["position_ids"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=model_kwargs["attention_mask"],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_kv_cache,
        )
        if use_kv_cache:
            return outputs
        return outputs, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        decoding_strategy: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        use_kv_cache: Optional[bool] = None,
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
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        config = self._resolve_generation_config(
            generation_config=generation_config,
            decoding_strategy=decoding_strategy,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            use_kv_cache=use_kv_cache,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self._validate_context_length(input_ids, config.max_new_tokens)
        if config.decoding_strategy == "beam_search":
            return self._beam_search_generate(
                input_ids, attention_mask, config
            )
        return self._autoregressive_generate(
            input_ids, attention_mask, config
        )

    def _resolve_generation_config(
        self,
        generation_config,
        decoding_strategy,
        max_new_tokens,
        min_new_tokens,
        do_sample,
        num_beams,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        length_penalty,
        use_kv_cache,
        eos_token_id,
        pad_token_id,
    ):
        base = generation_config or GenerationConfig()
        resolved_num_beams = (
            base.num_beams if num_beams is None else int(num_beams)
        )
        if decoding_strategy is not None:
            strategy = decoding_strategy
        elif resolved_num_beams > 1:
            strategy = "beam_search"
        elif do_sample:
            strategy = "nucleus_sampling"
        else:
            strategy = base.decoding_strategy

        config = replace(
            base,
            decoding_strategy=strategy,
            max_new_tokens=(
                base.max_new_tokens
                if max_new_tokens is None
                else int(max_new_tokens)
            ),
            min_new_tokens=(
                base.min_new_tokens
                if min_new_tokens is None
                else int(min_new_tokens)
            ),
            num_beams=resolved_num_beams,
            temperature=(
                base.temperature
                if temperature is None
                else float(temperature)
            ),
            top_k=base.top_k if top_k is None else int(top_k),
            top_p=base.top_p if top_p is None else float(top_p),
            repetition_penalty=(
                base.repetition_penalty
                if repetition_penalty is None
                else float(repetition_penalty)
            ),
            length_penalty=(
                base.length_penalty
                if length_penalty is None
                else float(length_penalty)
            ),
            use_kv_cache=(
                base.use_kv_cache
                if use_kv_cache is None
                else bool(use_kv_cache)
            ),
            eos_token_id=(
                self.eos_token_id
                if eos_token_id is None
                else int(eos_token_id)
            ),
            pad_token_id=(
                self.pad_token_id
                if pad_token_id is None
                else int(pad_token_id)
            ),
        )
        self._validate_generation_config(config)
        return config

    @staticmethod
    def _validate_generation_config(config):
        if config.decoding_strategy not in (
            "greedy",
            "beam_search",
            "nucleus_sampling",
        ):
            raise ValueError(
                "decoding_strategy must be 'greedy', 'beam_search', "
                "or 'nucleus_sampling'"
            )
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not 0 <= config.min_new_tokens <= config.max_new_tokens:
            raise ValueError(
                "min_new_tokens must be between 0 and max_new_tokens"
            )
        if (
            config.decoding_strategy == "beam_search"
            and config.num_beams < 2
        ):
            raise ValueError(
                "beam_search requires num_beams to be at least 2"
            )
        if config.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if config.top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")
        if not 0 < config.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if config.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")

    def _validate_context_length(self, input_ids, max_new_tokens):
        max_context_len = int(self.model.max_context_len)
        requested_length = input_ids.size(1) + max_new_tokens
        if requested_length > max_context_len:
            raise ValueError(
                "padded prompt width plus max_new_tokens must not exceed "
                f"max_context_len ({input_ids.size(1)} + "
                f"{max_new_tokens} > {max_context_len})"
            )

    def _autoregressive_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        model_kwargs = self.prepare_inputs_for_generation(input_ids, attention_mask)
        past_key_values = None
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.bool, device=self.device
        )

        for step in range(config.max_new_tokens):
            outputs, past_key_values = self._model_step(
                model_kwargs,
                config.use_kv_cache,
                past_key_values,
            )
            next_token_logits = outputs[:, -1, :]
            if step < config.min_new_tokens:
                next_token_logits[:, config.eos_token_id] = -float("inf")
            next_tokens = self._select_next_tokens(
                next_token_logits, model_kwargs["input_ids"], config
            )
            next_tokens = torch.where(
                unfinished_sequences.unsqueeze(-1),
                next_tokens,
                torch.full_like(next_tokens, config.pad_token_id),
            )
            unfinished_sequences &= (
                next_tokens.squeeze(-1) != config.eos_token_id
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                model_kwargs,
                next_tokens,
                unfinished_sequences,
            )
            if not unfinished_sequences.any():
                break

        return model_kwargs["input_ids"]

    def _select_next_tokens(self, logits, input_ids, config):
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, input_ids, config.repetition_penalty
            )
        if config.decoding_strategy == "greedy":
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / config.temperature
        if config.top_k > 0:
            logits = self._top_k_filtering(logits, config.top_k)
        if config.top_p < 1.0:
            logits = self._top_p_filtering(logits, config.top_p)
        return torch.multinomial(
            F.softmax(logits, dim=-1), num_samples=1
        )

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
        past_key_values = None

        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)  # [batch_size * num_beams]

        # Track unfinished sequences (1 = running, 0 = finished)
        unfinished_sequences = torch.ones(batch_size * num_beams, dtype=torch.long, device=self.device)

        for step in range(config.max_new_tokens):
            # Forward pass (all beams, including finished ones)
            outputs, past_key_values = self._model_step(
                model_kwargs,
                config.use_kv_cache,
                past_key_values,
            )

            # Get logits for the last position
            next_token_logits = outputs[:, -1, :]  # [batch_size * num_beams, vocab_size]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, model_kwargs['input_ids'], config.repetition_penalty
                )
            if step < config.min_new_tokens:
                next_token_logits[:, config.eos_token_id] = -float("inf")

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
            model_kwargs['position_ids'] = model_kwargs['position_ids'][beam_indices]
            past_key_values = self._reorder_cache(
                past_key_values, beam_indices
            )

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
        generated = model_kwargs["input_ids"].view(
            batch_size, num_beams, -1
        )
        final_scores = beam_scores.view(batch_size, num_beams)
        generated_lengths = (
            generated[:, :, input_ids.size(1):] != config.pad_token_id
        ).sum(dim=-1).clamp(min=1)
        normalized_scores = final_scores / (
            generated_lengths.float() ** config.length_penalty
        )
        best_beams = normalized_scores.argmax(dim=-1)
        return generated[
            torch.arange(batch_size, device=self.device), best_beams
        ]

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


class DecoderOnlyTranslator:
    """High-level batch translator for the decoder-only model."""

    def __init__(
        self,
        model,
        unified_lang,
        device=None,
        use_kv_cache=True,
        decoding_strategy="beam_search",
        num_beams=5,
        temperature=0.8,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        inference_max_new_tokens=160,
    ):
        self.model = model
        self.unified_lang = unified_lang
        self.device = device if device else model.device
        self.use_kv_cache = use_kv_cache
        self.decoding_strategy = decoding_strategy
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.inference_max_new_tokens = int(inference_max_new_tokens)
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
        if not texts:
            raise ValueError("texts must contain at least one sentence")
        token_ids_list = []
        for text in texts:
            token_ids_list.append(
                self.unified_lang.encode(
                    text,
                    character_level=is_cn,
                    add_bos=True,
                )
            )

        max_len = max(len(tokens) for tokens in token_ids_list)
        input_ids = []
        attention_mask = []
        for tokens in token_ids_list:
            pad_len = max_len - len(tokens)
            padded_tokens = [self.pad_token_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            input_ids.append(padded_tokens)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [
            self.unified_lang.decode(
                sequence.tolist(),
                skip_special_tokens=skip_special_tokens,
            )
            for sequence in token_ids
        ]

    def _resolve_generation_kwargs(
        self,
        max_new_tokens,
        decoding_strategy,
        num_beams,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        extra_kwargs,
    ):
        generation_kwargs = dict(extra_kwargs)
        generation_kwargs.setdefault("use_kv_cache", self.use_kv_cache)
        generation_kwargs.update(
            {
                "max_new_tokens": (
                    self.inference_max_new_tokens
                    if max_new_tokens is None
                    else max_new_tokens
                ),
                "decoding_strategy": (
                    decoding_strategy or self.decoding_strategy
                ),
                "num_beams": (
                    self.num_beams if num_beams is None else num_beams
                ),
                "temperature": (
                    self.temperature
                    if temperature is None
                    else temperature
                ),
                "top_k": self.top_k if top_k is None else top_k,
                "top_p": self.top_p if top_p is None else top_p,
                "repetition_penalty": (
                    self.repetition_penalty
                    if repetition_penalty is None
                    else repetition_penalty
                ),
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return generation_kwargs

    def _generate_new_tokens(
        self,
        input_ids,
        attention_mask,
        generation_kwargs,
    ):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        generated_ids = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        return generated_ids[:, input_ids.size(1):]

    def translate(
        self,
        texts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        decoding_strategy: Optional[str] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenize(texts, is_cn=True)
        generation_kwargs = self._resolve_generation_kwargs(
            max_new_tokens,
            decoding_strategy,
            num_beams,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            kwargs,
        )
        self.model.eval()
        generated_tokens = self._generate_new_tokens(
            inputs["input_ids"],
            inputs["attention_mask"],
            generation_kwargs,
        )
        return self.decode(generated_tokens, skip_special_tokens=True)

    def translate_batch(
        self,
        src_seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        decoding_strategy: Optional[str] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> List[List[int]]:
        if src_seqs.ndim != 2 or attention_mask.shape != src_seqs.shape:
            raise ValueError(
                "src_seqs and attention_mask must have matching 2D shapes"
            )
        generation_kwargs = self._resolve_generation_kwargs(
            max_new_tokens,
            decoding_strategy,
            num_beams,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            kwargs,
        )
        self.model.eval()
        generated_tokens = self._generate_new_tokens(
            src_seqs,
            attention_mask,
            generation_kwargs,
        )
        return [
            self._clean_generated_token_ids(sequence)
            for sequence in generated_tokens
        ]

    def _clean_generated_token_ids(self, sequence):
        tokens = []
        for token_id in sequence.tolist():
            if token_id in (self.pad_token_id, self.eos_token_id):
                break
            if token_id != self.bos_token_id:
                tokens.append(token_id)
        return tokens


# Convenience function for quick setup
def build_gpt_translator(model, unified_lang, device=None):
    """Build a GPT translator from model and language object."""
    return DecoderOnlyTranslator(model, unified_lang, device)


GPTTranslator = DecoderOnlyTranslator
