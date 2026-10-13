"""DSpark acceptance. Greedy uses argmax; sampling uses Sampler.sample_with_probs."""

from dataclasses import dataclass

import torch

from nanovllm.layers.sampler import Sampler


@dataclass
class SpeculativeResult:
    token_ids: torch.Tensor
    num_tokens: torch.Tensor
    num_accepted: torch.Tensor


def _sample_row(sampler: Sampler, logits: torch.Tensor, temperature: torch.Tensor, greedy: bool) -> torch.Tensor:
    row = logits.unsqueeze(0)
    if greedy:
        return row.argmax(dim=-1)
    token_ids, _ = sampler.sample_with_probs(row, temperature.reshape(1))
    return token_ids


def accept_dspark_draft(
    draft_token_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: bool,
    sampler: Sampler,
) -> SpeculativeResult:
    """Accept a draft prefix and sample one target correction or bonus token."""
    batch_size, num_draft_tokens = draft_token_ids.shape
    draft_probs = target_probs = None
    if greedy:
        accepted = draft_token_ids.eq(target_logits.argmax(dim=-1))
    else:
        scaled_draft = draft_logits.float() / temperatures[:, None, None]
        scaled_target = target_logits.float() / temperatures[:, None, None]
        draft_probs = scaled_draft.softmax(dim=-1)
        target_probs = scaled_target.softmax(dim=-1)
        draft_token_probs = draft_probs.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1)
        target_token_probs = target_probs.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1)
        accept_probs = (target_token_probs / draft_token_probs.clamp_min(1e-10)).clamp(max=1)
        accepted = torch.rand_like(accept_probs).lt(accept_probs)
    accepted = accepted.cumprod(dim=-1, dtype=torch.int32).bool()
    num_accepted = accepted.sum(dim=-1)

    token_ids = torch.empty(
        batch_size,
        num_draft_tokens + 1,
        dtype=draft_token_ids.dtype,
        device=draft_token_ids.device,
    )
    token_ids[:, :num_draft_tokens] = draft_token_ids
    for batch_idx in range(batch_size):
        accepted_count = int(num_accepted[batch_idx])
        if accepted_count == num_draft_tokens:
            logits = bonus_logits[batch_idx]
            sample_greedy = greedy
        elif greedy:
            logits = target_logits[batch_idx, accepted_count]
            sample_greedy = True
        else:
            residual = (target_probs[batch_idx, accepted_count] - draft_probs[batch_idx, accepted_count]).clamp_min(0)
            if residual.sum() <= 1e-8:
                residual = target_probs[batch_idx, accepted_count]
            token_ids[batch_idx, accepted_count] = torch.multinomial(residual, 1)[0]
            continue
        token_ids[batch_idx, accepted_count] = _sample_row(
            sampler, logits, temperatures[batch_idx], sample_greedy
        )
    return SpeculativeResult(token_ids, num_accepted + 1, num_accepted)
