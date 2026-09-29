from typing import NamedTuple

import torch


class SpeculativeSample(NamedTuple):
    # 每条请求接受了多少个草稿 token
    num_accepted: torch.Tensor

    # 拒绝时的 correction token，或者全部接受时的 bonus token
    next_token_ids: torch.Tensor

    # 每个草稿 token 的接受概率
    acceptance_probs: torch.Tensor


def rejection_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SpeculativeSample:
    """
    对草稿模型生成的 K 个 token 执行拒绝采样。

    参数形状：
        target_probs:
            [batch_size, K + 1, vocab_size]

        draft_probs:
            [batch_size, K, vocab_size]

        draft_token_ids:
            [batch_size, K]

    返回：
        num_accepted:
            [batch_size]

        next_token_ids:
            [batch_size]

        acceptance_probs:
            [batch_size, K]
    """

    if target_probs.ndim != 3:
        raise ValueError(
            "target_probs must have shape [B, K+1, V]"
        )

    if draft_probs.ndim != 3:
        raise ValueError(
            "draft_probs must have shape [B, K, V]"
        )

    if draft_token_ids.ndim != 2:
        raise ValueError(
            "draft_token_ids must have shape [B, K]"
        )

    batch_size, num_draft_tokens, vocab_size = (
        draft_probs.shape
    )

    expected_target_shape = (
        batch_size,
        num_draft_tokens + 1,
        vocab_size,
    )

    if tuple(target_probs.shape) != expected_target_shape:
        raise ValueError(
            "target_probs must have shape "
            f"{expected_target_shape}"
        )

    expected_token_shape = (
        batch_size,
        num_draft_tokens,
    )

    if tuple(draft_token_ids.shape) != expected_token_shape:
        raise ValueError(
            "draft_token_ids must have shape "
            f"{expected_token_shape}"
        )

    if num_draft_tokens == 0:
        raise ValueError(
            "at least one draft token is required"
        )

    if (
        target_probs.device != draft_probs.device
        or draft_token_ids.device != draft_probs.device
    ):
        raise ValueError(
            "all inputs must be on the same device"
        )

    # target_probs 的前 K 个位置用于验证 K 个草稿 token。
    # 最后一个位置用于所有草稿 token 都接受时生成 bonus token。
    target_draft_probs = target_probs[
        :, :num_draft_tokens, :
    ]

    # [B, K] -> [B, K, 1]
    token_indices = draft_token_ids.long().unsqueeze(-1)

    # 取得目标模型给每个草稿 token 的概率 p(x)。
    p_x = torch.gather(
        target_draft_probs,
        dim=2,
        index=token_indices,
    ).squeeze(-1)

    # 取得草稿模型给自己生成 token 的概率 q(x)。
    q_x = torch.gather(
        draft_probs,
        dim=2,
        index=token_indices,
    ).squeeze(-1)

    # 接受概率 min(1, p(x) / q(x))。
    acceptance_probs = (
        p_x / q_x.clamp_min(1e-10)
    ).clamp(max=1.0)

    # 单元测试可以显式传入 uniforms，
    # 正常运行时由 PyTorch 生成随机数。
    if uniforms is None:
        uniforms = torch.rand(
            acceptance_probs.shape,
            dtype=acceptance_probs.dtype,
            device=acceptance_probs.device,
            generator=generator,
        )
    elif tuple(uniforms.shape) != tuple(
        acceptance_probs.shape
    ):
        raise ValueError(
            "uniforms must have shape "
            f"{tuple(acceptance_probs.shape)}"
        )

    # 每个位置是否通过接受判断。
    accepted = uniforms < acceptance_probs

    # 只接受第一个拒绝位置之前的连续前缀。
    #
    # 例如：
    # accepted = [True, False, True, True]
    # 最终只能接受第一个 token，后面的 True 无效。
    accepted_prefix = torch.cumprod(
        accepted.to(torch.int32),
        dim=1,
    ).bool()

    # 每条请求接受的草稿 token 数量。
    num_accepted = accepted_prefix.sum(dim=1)

    batch_indices = torch.arange(
        batch_size,
        device=draft_probs.device,
    )

    # 如果没有全部接受，它就是第一个拒绝的位置。
    # 如果全部接受，先限制到 K-1，避免数组越界；
    # 后面会改用 bonus_probs，所以该位置不会真正使用。
    first_rejected = num_accepted.clamp(
        max=num_draft_tokens - 1
    )

    p_rejected = target_draft_probs[
        batch_indices,
        first_rejected,
    ]

    q_rejected = draft_probs[
        batch_indices,
        first_rejected,
    ]

    # 拒绝后的修正分布 max(p-q, 0)。
    #在草稿 token 被拒绝时，构造一个新的替代 token 分布
    correction_probs = (
        p_rejected - q_rejected
    ).clamp_min(0)

    correction_mass = correction_probs.sum(
        dim=-1,
        keepdim=True,
    )

    correction_probs = (
        correction_probs
        / correction_mass.clamp_min(1e-10)
    )

    # 理论上发生拒绝时 correction_mass 应当大于 0。
    # 这里为了防止浮点误差导致全 0，退回目标模型分布。
    fallback_probs = (
        p_rejected
        / p_rejected.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-10)
    )

    correction_probs = torch.where(
        correction_mass > 1e-10,
        correction_probs,
        fallback_probs,
    )

    # K 个草稿 token 是否全部接受。
    all_accepted = (
        num_accepted == num_draft_tokens
    )

    # target_probs 最后一个位置用于 bonus token。
    bonus_probs = target_probs[:, -1, :]

    # 全部接受：从 bonus_probs 采样。
    # 出现拒绝：从 correction_probs 采样。
    next_probs = torch.where(
        all_accepted.unsqueeze(-1),
        bonus_probs, #True从这里采样
        correction_probs, #对应False
    )

    next_token_ids = torch.multinomial(
        next_probs,
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    return SpeculativeSample(
        num_accepted=num_accepted,
        next_token_ids=next_token_ids,
        acceptance_probs=acceptance_probs,
    )