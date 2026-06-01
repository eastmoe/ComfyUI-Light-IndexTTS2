# coding=utf-8
"""Beam-search compatibility for Transformers versions without legacy helpers."""

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union

import torch


class BeamScorer(ABC):
    """Minimal legacy BeamScorer interface used by the vendored generation code."""

    @abstractmethod
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    """Legacy-compatible beam search scorer, adapted from Transformers 4.x."""

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}."
            )
        if not isinstance(num_beam_groups, int) or num_beam_groups > num_beams or num_beams % num_beam_groups != 0:
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams`, and `num_beams` "
                f"must be divisible by `num_beam_groups`, got {num_beam_groups=} and {num_beams=}."
            )

        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device
        )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[Union[int, torch.Tensor]] = None,
        eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        cur_len = input_ids.shape[-1] + 1
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if batch_size != input_ids.shape[0] // self.group_size:
            raise ValueError(
                f"A beam size of {input_ids.shape[0]} is used as input, but {self.group_size} is expected."
            )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
        eos_token_id = _normalize_eos_token_id(eos_token_id, device)

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                if eos_token_id is not None and _is_eos_token(next_token, eos_token_id):
                    if beam_token_rank >= self.group_size:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx] + (batch_beam_idx,)
                    else:
                        beam_index = None
                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                        generated_len=cur_len - decoder_prompt_len,
                    )
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to "
                    f"`eos_token_id: {eos_token_id}`."
                )

            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[Union[int, torch.Tensor]] = None,
        eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps) // self.num_beam_groups
        eos_token_id = _normalize_eos_token_id(eos_token_id, input_ids.device)

        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                generated_len = final_tokens.shape[-1] - decoder_prompt_len
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index, generated_len=generated_len)

        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score, best_hyp, best_index = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)
                best_indices.append(best_index)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: Optional[torch.LongTensor] = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo
            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx, device=indices.device)
            if sent_lengths[i] < sent_max_len and eos_token_id is not None:
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is a string, `max_length` must be passed to BeamSearchScorer."
            )

    def __len__(self):
        return len(self.beams)

    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        if generated_len is not None:
            score = sum_logprobs / (generated_len**self.length_penalty)
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        if len(self) < self.num_beams:
            return False

        if self.early_stopping is True:
            return True
        if self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            return self.worst_score >= highest_attainable_score

        if self.length_penalty > 0.0:
            if self.max_length <= decoder_prompt_len:
                raise ValueError("max_length is not larger than decoder prompt length")
            highest_attainable_score = best_sum_logprobs / (
                self.max_length - decoder_prompt_len
            ) ** self.length_penalty
        else:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
        return self.worst_score >= highest_attainable_score


class _UnavailableConstrainedBeamFeature:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Constrained beam search is unavailable with this Transformers version. Use ordinary beam search, "
            "sampling/greedy generation, or install transformers<5."
        )


class ConstrainedBeamSearchScorer(_UnavailableConstrainedBeamFeature):
    pass


class DisjunctiveConstraint(_UnavailableConstrainedBeamFeature):
    pass


class PhrasalConstraint(_UnavailableConstrainedBeamFeature):
    pass


def _normalize_eos_token_id(eos_token_id, device):
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, torch.Tensor):
        return eos_token_id.to(device)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return torch.tensor(eos_token_id, device=device)


def _is_eos_token(token, eos_token_id: torch.Tensor) -> bool:
    return bool((eos_token_id == token).any().item())
