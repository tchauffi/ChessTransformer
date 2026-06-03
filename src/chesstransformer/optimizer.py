"""Combined AdamW + Muon optimizer.

The two underlying optimizers are kept as separate ``torch.optim`` instances
internally; this class is a thin wrapper that exposes the standard
``torch.optim.Optimizer`` interface (``step``, ``zero_grad``, ``param_groups``,
``state_dict`` / ``load_state_dict``) so that LR schedulers, gradient
accumulation, and HuggingFace ``Accelerator`` treat it as a single optimizer.

Routing rule (when constructed from an ``nn.Module``):

* parameters whose name contains an ``embedding_keyword`` → AdamW (low LR)
* parameters whose name contains a ``head_keyword``      → AdamW (no decay)
* remaining parameters with ``ndim >= 2``                → Muon
* remaining parameters with ``ndim < 2`` (biases/norms)  → AdamW (no decay)
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW, Muon, Optimizer


class AdamWMuon(Optimizer):
    """Wrapper that drives an inner ``AdamW`` and ``Muon`` in lock-step.

    Parameters
    ----------
    model:
        ``nn.Module`` whose parameters will be auto-routed by name.
    lr:
        Base AdamW learning rate (used for the ``other`` 1D parameter group).
    lr_embedding, lr_head:
        Per-group AdamW LR overrides. Default to ``lr``.
    lr_muon:
        Muon LR for ≥2D body weights. Default ``= 3 * lr`` (Muon tolerates a
        higher LR than AdamW because updates are orthogonalised).
    weight_decay:
        AdamW weight decay applied to the embedding group. Heads and 1D
        params get ``0.0`` by convention.
    weight_decay_muon:
        Muon weight decay. Default = ``weight_decay``.
    embedding_keywords / head_keywords:
        Substrings used to classify parameters by ``named_parameters`` key.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 3e-4,
        lr_embedding: float | None = None,
        lr_head: float | None = None,
        lr_muon: float | None = None,
        weight_decay: float = 0.1,
        weight_decay_muon: float | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        embedding_keywords: Iterable[str] = ("embedding", "cls_token"),
        head_keywords: Iterable[str] = ("lm_head", "value_head"),
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"AdamWMuon expects an nn.Module to auto-route parameters by name, "
                f"got {type(model).__name__}."
            )

        lr_embedding = lr if lr_embedding is None else lr_embedding
        lr_head = lr if lr_head is None else lr_head
        lr_muon = 3.0 * lr if lr_muon is None else lr_muon
        weight_decay_muon = weight_decay if weight_decay_muon is None else weight_decay_muon

        embedding_keywords = tuple(embedding_keywords)
        head_keywords = tuple(head_keywords)

        emb_params: list[torch.Tensor] = []
        head_params: list[torch.Tensor] = []
        other_params: list[torch.Tensor] = []
        muon_params: list[torch.Tensor] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in embedding_keywords):
                emb_params.append(param)
            elif any(kw in name for kw in head_keywords):
                head_params.append(param)
            elif param.ndim >= 2:
                muon_params.append(param)
            else:
                other_params.append(param)

        adamw_groups: list[dict] = []
        if emb_params:
            adamw_groups.append({
                "params": emb_params, "lr": lr_embedding,
                "weight_decay": weight_decay, "name": "adamw_embedding",
            })
        if other_params:
            adamw_groups.append({
                "params": other_params, "lr": lr,
                "weight_decay": 0.0, "name": "adamw_other",
            })
        if head_params:
            adamw_groups.append({
                "params": head_params, "lr": lr_head,
                "weight_decay": 0.0, "name": "adamw_head",
            })

        if not adamw_groups:
            raise ValueError("No AdamW-eligible parameters found in model.")

        self.adamw = AdamW(adamw_groups, lr=lr, betas=betas, eps=eps,
                           weight_decay=weight_decay)

        if muon_params:
            self.muon: Muon | None = Muon(
                [{"params": muon_params, "lr": lr_muon,
                  "weight_decay": weight_decay_muon, "name": "muon_body"}],
                lr=lr_muon, weight_decay=weight_decay_muon,
                momentum=muon_momentum, nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
            )
        else:
            self.muon = None

        # Initialise the ``Optimizer`` base with a placeholder group so
        # ``isinstance(opt, Optimizer)`` checks (Accelerate, schedulers) pass
        # and ``self.state`` / ``self.defaults`` exist. We then expose the
        # *inner* groups via ``param_groups`` so schedulers mutate them in
        # place and both sub-optimizers see the new LR.
        all_params = emb_params + other_params + head_params + muon_params
        self._init_done = False
        super().__init__([{"params": all_params}], defaults={"lr": lr})
        self.param_groups = self._combined_param_groups()
        self._init_done = True

    # ── helpers ─────────────────────────────────────────────────────────
    def _combined_param_groups(self) -> list[dict]:
        groups = list(self.adamw.param_groups)
        if self.muon is not None:
            groups += list(self.muon.param_groups)
        return groups

    # ── Optimizer API ───────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "adamw": self.adamw.state_dict(),
            "muon": self.muon.state_dict() if self.muon is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.adamw.load_state_dict(state_dict["adamw"])
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        # Re-link in case sub-optimizers rebuilt their group dicts.
        self.param_groups = self._combined_param_groups()

    def add_param_group(self, param_group) -> None:
        if not getattr(self, "_init_done", False):
            # Called from ``Optimizer.__init__`` with the placeholder group.
            super().add_param_group(param_group)
            return
        raise NotImplementedError(
            "add_param_group is not supported on AdamWMuon after construction; "
            "build a new optimizer with the full parameter set instead."
        )

    def __repr__(self) -> str:
        n_emb = sum(p.numel() for g in self.adamw.param_groups
                    if g.get("name") == "adamw_embedding" for p in g["params"])
        n_head = sum(p.numel() for g in self.adamw.param_groups
                     if g.get("name") == "adamw_head" for p in g["params"])
        n_other = sum(p.numel() for g in self.adamw.param_groups
                      if g.get("name") == "adamw_other" for p in g["params"])
        n_muon = (sum(p.numel() for g in self.muon.param_groups for p in g["params"])
                  if self.muon is not None else 0)
        return (
            f"AdamWMuon(emb={n_emb:,}, other_1d={n_other:,}, "
            f"head={n_head:,}, muon_2d={n_muon:,})"
        )


# Backwards-compat alias for the previous skeleton class.
AdamWMuonOptim = AdamWMuon