import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformers.modeling_outputs import BaseModelOutput


class PhraseBoundaryDetector:
    BOUNDARY_TOKEN_SUBSTRINGS = {"▁।", "▁॥", "▁,", "▁;", "▁?", "▁!", ".", ","}

    def __init__(self, segment_size: int = 12):
        self.segment_size = segment_size

    def split_token_ids(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> List[Tuple[int, int]]:
        seq_len = input_ids.shape[0]
        try:
            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        except Exception:
            tokens = [""] * seq_len

        segments = []
        start = 0
        while start < seq_len:
            end = min(start + self.segment_size, seq_len)
            if end < seq_len:
                for offset in range(-3, 4):
                    candidate = end + offset
                    if 0 < candidate < seq_len:
                        tok = tokens[candidate] or ""
                        if any(b in tok for b in self.BOUNDARY_TOKEN_SUBSTRINGS):
                            end = candidate + 1
                            break
            segments.append((start, end))
            start = end

        return segments if segments else [(0, seq_len)]


class CrossSegmentTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_segments: int = 64,
    ):
        super().__init__()
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.max_segments = max_segments
        self.segment_pos_emb = nn.Embedding(max_segments, d_model)

    def forward(
        self,
        segment_summaries: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = segment_summaries.shape

        pos_ids = torch.arange(N, device=segment_summaries.device).clamp(
            max=self.max_segments - 1
        )
        x = segment_summaries.float() + self.segment_pos_emb(pos_ids).float()

        key_padding_mask = None
        if segment_mask is not None:
            key_padding_mask = (segment_mask == 0)

        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return out.to(segment_summaries.dtype)


class HierarchicalEncoder(nn.Module):

    def __init__(
        self,
        base_encoder: nn.Module,
        d_model: int,
        tokenizer,
        segment_size: int = 12,
        cross_segment_layers: int = 2,
        nhead: int = 8,
        long_context_threshold: int = 20,
        max_segments: int = 64,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.long_context_threshold = long_context_threshold
        self.boundary_detector = PhraseBoundaryDetector(segment_size=segment_size)
        self.cross_segment_transformer = CrossSegmentTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=cross_segment_layers,
            max_segments=max_segments,
        )
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def _encode_segment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
        )
        return out.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutput:

        batch_size, seq_len = input_ids.shape

        _ENCODER_INVALID_KEYS = (
            "return_dict", "head_mask", "inputs_embeds",
            "output_attentions", "output_hidden_states",
            "labels", "decoder_input_ids", "decoder_attention_mask",
            "decoder_head_mask", "cross_attn_head_mask",
            "encoder_outputs",
        )
        for _k in _ENCODER_INVALID_KEYS:
            kwargs.pop(_k, None)

        if seq_len <= self.long_context_threshold:
            return self.base_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )

        all_token_hidden: List[torch.Tensor] = []
        all_seg_summaries: List[torch.Tensor] = []
        all_seg_positions: List[List[Tuple[int, int]]] = []

        for b in range(batch_size):
            ids_b  = input_ids[b]
            mask_b = attention_mask[b]

            segments = self.boundary_detector.split_token_ids(ids_b, self.tokenizer)
            segments = segments[:self.cross_segment_transformer.max_segments]

            token_hiddens = torch.zeros(
                seq_len, self.d_model,
                device=input_ids.device,
                dtype=torch.float32,
            )
            seg_summaries: List[torch.Tensor] = []

            for (start, end) in segments:
                seg_ids  = ids_b[start:end].unsqueeze(0)
                seg_mask = mask_b[start:end].unsqueeze(0)

                seg_hidden = self._encode_segment(seg_ids, seg_mask)
                seg_hidden_f = seg_hidden[0].float()

                token_hiddens[start:end] = seg_hidden_f
                seg_summaries.append(seg_hidden_f[0])

            all_token_hidden.append(token_hiddens)
            all_seg_summaries.append(torch.stack(seg_summaries))
            all_seg_positions.append(segments)

        max_segs = max(s.shape[0] for s in all_seg_summaries)
        padded = torch.zeros(
            batch_size, max_segs, self.d_model,
            device=input_ids.device, dtype=torch.float32,
        )
        seg_mask_tensor = torch.zeros(
            batch_size, max_segs,
            device=input_ids.device, dtype=torch.float32,
        )
        for b, s in enumerate(all_seg_summaries):
            n = s.shape[0]
            padded[b, :n] = s
            seg_mask_tensor[b, :n] = 1.0

        enriched_segs = self.cross_segment_transformer(
            padded, seg_mask_tensor
        )

        token_hiddens_batch = torch.stack(all_token_hidden)
        enriched_tokens = torch.zeros_like(token_hiddens_batch)

        for b in range(batch_size):
            for seg_idx, (start, end) in enumerate(all_seg_positions[b]):
                enriched_tokens[b, start:end] = enriched_segs[b, seg_idx].unsqueeze(0)

        fused = torch.cat([token_hiddens_batch, enriched_tokens], dim=-1)
        fused = self.fusion_proj(fused.to(self.fusion_proj.weight.dtype))
        fused = self.layer_norm(fused)

        return BaseModelOutput(last_hidden_state=fused)
