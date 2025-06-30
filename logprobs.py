import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


class ForwardDiffusion:
    def __init__(self, mask_token_id: int, min_prob=0.1, max_prob=0.9):
        self.mask_token_id = mask_token_id

        self.min_prob = min_prob
        self.max_prob = max_prob

    def __call__(self, input_ids: torch.LongTensor, t: torch.FloatTensor):
        batch_size, seq_len = input_ids.shape

        assert t.shape == (batch_size, 1)

        t = t.clamp(min=self.min_prob, max=self.max_prob)

        mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        perm = torch.randperm(seq_len, device=input_ids.device)
        unmask_positions = perm[: int(seq_len * t[0])]

        mask_indices[:, unmask_positions] = True

        noisy_ids = input_ids.clone()

        noisy_ids[mask_indices] = self.mask_token_id

        return noisy_ids, mask_indices


def get_logprobs_elbo(
    model,
    prompt_ids,
    completion_ids,
    timesteps,
    mask_token_id,
    device,
    requires_grad=False,
):
    forward_diffusion = ForwardDiffusion(mask_token_id=mask_token_id)

    batch_size = prompt_ids.shape[0]
    prompt_length = prompt_ids.shape[1]

    log_probs_sum = torch.zeros(batch_size, device=device)

    for t in timesteps:
        t_expanded = t.unsqueeze(0).expand(batch_size, -1)

        masked_completion_ids, mask_indices = forward_diffusion(
            completion_ids, t_expanded
        )

        masked_sequence = torch.cat([prompt_ids, masked_completion_ids], dim=-1)

        if not requires_grad:
            with torch.no_grad():
                outputs = model(masked_sequence)
        else:
            outputs = checkpoint(model, masked_sequence, use_reentrant=False)

        logits = outputs.logits[:, prompt_length:, :]
        log_probs = F.log_softmax(logits, dim=-1)

        gathered_log_probs = log_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        gathered_log_probs = gathered_log_probs * mask_indices

        step_sum = gathered_log_probs.sum(dim=-1) / t

        log_probs_sum += step_sum

    return log_probs_sum / len(timesteps)
