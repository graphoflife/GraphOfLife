#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for a GraphOfLife simulation run.

Every knob the web UI can set lives here in one dataclass. A run stores its own
SimConfig on disk, so a resumed run always continues under the exact settings it
started with, and old runs stay readable after the defaults change.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, List


@dataclass
class SimConfig:
    """All parameters of a single simulation run."""

    # ---- Economy ----
    # Total tokens in the world. Conserved unless tokens_created_per_phase > 0.
    total_tokens: int = 150_000
    # Tokens injected at every cleanup. 0 keeps the economy closed.
    tokens_created_per_phase: int = 0

    # ---- Seed graph (Watts-Strogatz) ----
    # 0 means "derive from total_tokens": n = total_tokens / 100.
    n_nodes: int = 0
    # 0 means "derive from n": k = max(n / 100, 5).
    k_neighbors: int = 0
    rewire_p: float = 0.2

    # ---- Brain ----
    hidden_layers: List[int] = field(default_factory=lambda: [50, 45, 40, 35, 30])
    message_amount: int = 5
    random_input_amount: int = 5
    exchange_messages: bool = True

    # ---- Mutation ----
    mutation_probability: float = 0.5
    mutation_noise_std: float = 0.2
    mutation_sparsity: float = 0.1

    # ---- Run control ----
    max_steps: int = 500_000
    # Below this node count the run is declared extinct and stops.
    extinction_threshold: int = 50
    # Write a resume checkpoint every N iterations. 0 disables checkpointing.
    checkpoint_every: int = 20
    # Random seed for reproducibility. None means "seed from entropy".
    seed: int | None = None

    # ---- Export ----
    # Record every Nth iteration into the viewer format. 1 = every iteration.
    export_every: int = 1
    # Store per-agent decisions alongside the topology. Costs disk, enables the
    # decision inspector in the UI.
    export_decisions: bool = True

    # --------------------------------------------------------------------
    # Derived values
    # --------------------------------------------------------------------

    def resolved_n(self) -> int:
        return self.n_nodes if self.n_nodes > 0 else int(self.total_tokens / 100)

    def resolved_k(self) -> int:
        if self.k_neighbors > 0:
            return self.k_neighbors
        return max(int(self.resolved_n() / 100), 5)

    def n_inputs(self) -> int:
        # 1 (is-self) + 4 (own/target tokens and degrees) + 24 (quantiles)
        base = 29
        return base + 4 * self.message_amount + self.random_input_amount

    def n_outputs(self) -> int:
        # 11 fixed heads + the message vector
        return 11 + self.message_amount

    def head_layout(self) -> Dict[str, Any]:
        """Output row layout. Mirrors HEAD in the engine, for the UI to display."""
        m = self.message_amount
        return {
            "REPRO_FRACTION": [0, 2],
            "LINK": [2, 4],
            "LINK_MODE": [4, 6],
            "BLOTTO": [6, 7],
            "BLOTTO_MODE": [7, 9],
            "REV_FRACTION": [9, 11],
            "MESSAGE": [11, 11 + m],
        }

    # --------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimConfig":
        """Build a config from untrusted input, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}

        if "hidden_layers" in clean:
            clean["hidden_layers"] = [int(x) for x in clean["hidden_layers"] if int(x) > 0]
            if not clean["hidden_layers"]:
                clean.pop("hidden_layers")

        cfg = cls(**clean)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject values that would produce a broken or unrunnable simulation."""
        if self.total_tokens < 1:
            raise ValueError("total_tokens must be at least 1")
        if self.resolved_n() < 2:
            raise ValueError("seed graph needs at least 2 nodes")
        if self.resolved_k() < 2:
            raise ValueError("k_neighbors must be at least 2")
        if self.resolved_k() >= self.resolved_n():
            raise ValueError("k_neighbors must be smaller than the node count")
        if not 0.0 <= self.rewire_p <= 1.0:
            raise ValueError("rewire_p must be between 0 and 1")
        if self.message_amount < 0 or self.random_input_amount < 0:
            raise ValueError("message_amount and random_input_amount cannot be negative")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 0.0 <= self.mutation_sparsity <= 1.0:
            raise ValueError("mutation_sparsity must be between 0 and 1")
        if self.mutation_noise_std < 0:
            raise ValueError("mutation_noise_std cannot be negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.export_every < 1:
            raise ValueError("export_every must be at least 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every cannot be negative")
        if self.total_tokens < self.resolved_n():
            raise ValueError("total_tokens must be at least the seed node count, or every node starts broke")
