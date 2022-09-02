import torch
from torch import Tensor


def top_logits(
    logits: Tensor, k: int = 0, p: float = 1.0, mask: float = 1e-10
) -> Tensor:
    """Remove low probability logits via masking.

    Args:
      logits: class logits in shape of (batch size, total_classes).
      k: specifying top k largest logits to keep.
      p: specifying a probability for finding a minimum set of largest
        logits to keep, where their cumulative probability is no less than p
        (actually in the following version, it is "...cumulative probability is
        the largest but no more than p").
      mask: an value that's used to replace logits that don't satisfy the
        keep conditions.

    Returns:
      logits where low probability ones are replaced with mask.
    """
    mask = torch.ones_like(logits) * mask
    if k > 0:
        min_logits = torch.topk(logits, k=k)[0][:, -1:]
        logits = torch.where(logits < min_logits, mask, logits)
    if p < 1.0:
        sorted_logits = logits.sort(descending=True)
        cum_probs = torch.cumsum(sorted_logits.softmax(), axis=-1)
        min_logits = -torch.max(
            torch.where(cum_probs <= p, -sorted_logits, mask), -1, keepdims=True
        )
        min_logits = torch.minimum(min_logits, sorted_logits[:, :1])
        logits = torch.where(logits < min_logits, mask, logits)
    return logits


def sample(next_logits, step, temperature, top_k, top_p):
    sampling_logits = next_logits / temperature
    sampling_logits = top_logits(sampling_logits, k=top_k, p=top_p)
    return torch.multinomial(sampling_logits, num_samples=1).squeeze(1)
