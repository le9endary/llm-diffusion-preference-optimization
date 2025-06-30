import torch
from tqdm import tqdm

from logprobs import get_logprobs_elbo


def evaluate_preference_alignment(model, test_dataset_clean, get_tokenized, device):
    model.eval()

    wins, losses, ties = 0, 0, 0

    batch_size = 8

    for i in tqdm(range(0, len(test_dataset_clean[:256]), batch_size)):
        batch = test_dataset_clean[i : i + batch_size]

        prompts = [sample["prompt"] for sample in batch]
        chosen = [sample["chosen"] for sample in batch]
        rejected = [sample["rejected"] for sample in batch]

        prompt_inputs = get_tokenized(prompts)
        chosen_inputs = get_tokenized(chosen)
        rejected_inputs = get_tokenized(rejected)

        prompt_ids = prompt_inputs["input_ids"].to(device)
        chosen_ids = chosen_inputs["input_ids"].to(device)
        rejected_ids = rejected_inputs["input_ids"].to(device)

        with torch.no_grad():
            timesteps = torch.rand(8)

            chosen_logprobs = get_logprobs_elbo(
                model,
                prompt_ids,
                chosen_ids,
                timesteps,
            )

            rejected_logprobs = get_logprobs_elbo(
                model,
                prompt_ids,
                rejected_ids,
                timesteps,
            )

        for chosen_lp, rejected_lp in zip(chosen_logprobs, rejected_logprobs):
            if chosen_lp > rejected_lp:
                wins += 1
            elif chosen_lp < rejected_lp:
                losses += 1
            else:
                ties += 1

    total = wins + losses + ties

    print(f"\nResults on HH-RLHF ({total} samples):")
    print(f"Wins     : {wins} ({wins / total:.2%})")
    print(f"Losses   : {losses} ({losses / total:.2%})")
    print(f"Ties     : {ties} ({ties / total:.2%})")
