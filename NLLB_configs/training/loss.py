import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LengthAwareLoss(nn.Module):

    def __init__(
        self,
        pad_token_id: int,
        length_penalty_weight: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.weight = length_penalty_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        predicted_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch, seq_len, vocab = logits.shape
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        nll = loss_fct(logits.view(-1, vocab), labels.view(-1))

        if self.weight == 0.0 or predicted_ids is None:
            return nll

        ref_lengths = (labels != -100).sum(dim=1).float()

        if predicted_ids is not None:
            pred_lengths = (predicted_ids != self.pad_token_id).sum(dim=1).float()
        else:
            pred_tokens = logits.argmax(dim=-1)
            pred_lengths = (pred_tokens != self.pad_token_id).sum(dim=1).float()

        ref_lengths = ref_lengths.clamp(min=1.0)
        length_ratio_error = ((pred_lengths - ref_lengths).abs() / ref_lengths).mean()

        return nll + self.weight * length_ratio_error


class BucketBalancedLoss(nn.Module):

    def __init__(
        self,
        pad_token_id: int,
        bucket_loss_weights: list = None,
        length_penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = LengthAwareLoss(
            pad_token_id=pad_token_id,
            length_penalty_weight=length_penalty_weight,
        )
        self.bucket_weights = torch.tensor(
            bucket_loss_weights or [1.0, 1.0, 1.2, 1.5, 2.0]
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        buckets: torch.Tensor,
        predicted_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.bucket_weights = self.bucket_weights.to(logits.device)

        sample_weights = self.bucket_weights[buckets - 1]

        total_loss = self.base_loss(logits, labels, predicted_ids)

        weight_scale = sample_weights.mean() / self.bucket_weights.mean()
        return total_loss * weight_scale
