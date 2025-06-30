import torch
from logprobs import get_logprobs_elbo
import torch.nn.functional as F


def dpo_loss(
    policy_model,
    ref_model,
    prompt_ids,
    chosen_ids,
    rejected_ids,
    mask_token_id,
    device,
    beta=0.1,
    label_smoothing=0.1,
):
    timesteps = torch.rand(8)

    policy_chosen_log_probs = get_logprobs_elbo(
        policy_model,
        prompt_ids,
        chosen_ids,
        timesteps,
        mask_token_id,
        device,
        requires_grad=True,
    )

    policy_rejected_log_probs = get_logprobs_elbo(
        policy_model,
        prompt_ids,
        rejected_ids,
        timesteps,
        mask_token_id,
        device,
        requires_grad=True,
    )

    with torch.no_grad():
        ref_chosen_log_probs = get_logprobs_elbo(
            ref_model,
            prompt_ids,
            chosen_ids,
            timesteps,
            mask_token_id,
            device,
        )

        ref_rejected_log_probs = get_logprobs_elbo(
            ref_model,
            prompt_ids,
            rejected_ids,
            timesteps,
            mask_token_id,
            device,
        )

    chosen_rewards = policy_chosen_log_probs - ref_chosen_log_probs
    rejected_rewards = policy_rejected_log_probs - ref_rejected_log_probs

    pi_logratios = beta * (chosen_rewards - rejected_rewards)

    loss_standard = -F.logsigmoid(pi_logratios)
    loss_uniform = -F.logsigmoid(-pi_logratios)

    loss = (1 - label_smoothing) * loss_standard + label_smoothing * loss_uniform

    return loss.mean()
