"""V207 transparent multimodal model contract.

This module contains only the architecture-level contract used before the first
empirical fold.  It does not load health outcomes or start production training.

本模块只实现首折实证训练前的结构合同；不读取健康结局，也不启动生产训练。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class FoldFittedModality(nn.Module):
    """Fold-fitted projection and quality gate for one frozen 768-D channel.

    单个冻结 768 维通道的折内投影与质量门控。
    """

    def __init__(self, embedding_dim: int, latent_dim: int, quality_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, latent_dim)
        self.normalization = nn.LayerNorm(latent_dim)
        self.gate = nn.Linear(quality_dim, 1)

    def forward(
        self,
        embedding: torch.Tensor,
        quality: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = available.to(dtype=torch.bool).reshape(-1, 1)
        # Missing vectors may be NaN on disk. They must never become a fake zero
        # observation or leak NaNs through the projection.
        # 磁盘中的缺失向量可为 NaN；投影前只在缺失位置安全置零。
        safe_embedding = torch.where(
            available,
            torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(embedding),
        )
        latent = self.normalization(self.projection(safe_embedding))
        gate = torch.sigmoid(self.gate(quality)) * available.to(quality.dtype)
        return latent * gate, gate


def legal_relation_mean(
    node_context: torch.Tensor,
    adjacency: torch.Tensor,
    legal_senders: torch.Tensor,
) -> torch.Tensor:
    """Aggregate relation messages using receiver-by-sender legal masks.

    adjacency is ``[relation, receiver, sender]`` and legal_senders is
    ``[relation, sender]``. Illegal senders are removed before normalization.
    邻接矩阵按“关系×受体×发送者”排列，非法发送者在归一化前剔除。
    """

    if adjacency.ndim != 3 or legal_senders.ndim != 2:
        raise ValueError("adjacency and legal_senders must be rank 3 and 2")
    if adjacency.shape[0] != legal_senders.shape[0]:
        raise ValueError("relation count mismatch")
    if adjacency.shape[1] != node_context.shape[0] or adjacency.shape[2] != node_context.shape[0]:
        raise ValueError("adjacency node dimensions do not match node_context")
    if legal_senders.shape[1] != node_context.shape[0]:
        raise ValueError("legal_senders node dimension mismatch")

    weights = adjacency.to(node_context.dtype) * legal_senders.to(node_context.dtype)[:, None, :]
    denominator = weights.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(node_context.dtype).eps),
        torch.zeros_like(weights),
    )
    return torch.einsum("rns,sd->rnd", normalized, node_context)


def legal_relation_mean_from_edges(
    node_context: torch.Tensor,
    edge_index: torch.Tensor,
    legal_senders: torch.Tensor,
    relation_count: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse equivalent of ``legal_relation_mean`` for large real graphs.

    ``edge_index`` is ``[relation, 2, edge]`` with rows ``receiver, sender``.
    The operation keeps gradients through sender context while avoiding a dense
    ``relation × receiver × sender`` tensor.  Legal sender masks are applied
    before receiver-wise normalization, matching the dense contract exactly.

    对大图使用稀疏等价实现，避免构造关系×受体×发送者的巨型稠密张量；
    在受体归一化前应用合法发送者门，并保持发送者上下文的梯度路径。
    """
    if edge_index.ndim != 3 or edge_index.shape[1] != 2:
        raise ValueError("edge_index must have shape [relation, 2, edge]")
    if legal_senders.ndim != 2 or legal_senders.shape[0] != relation_count:
        raise ValueError("legal_senders must have shape [relation, node]")
    if edge_index.shape[0] != relation_count or legal_senders.shape[1] != node_context.shape[0]:
        raise ValueError("relation or node dimensions do not match")

    if edge_weight is not None and edge_weight.shape != edge_index.shape[:1] + edge_index.shape[2:3]:
        raise ValueError("edge_weight must have shape [relation, edge]")

    n_nodes, context_dim = node_context.shape
    output = node_context.new_zeros((relation_count, n_nodes, context_dim))
    for relation in range(relation_count):
        receiver = edge_index[relation, 0].to(dtype=torch.long)
        sender = edge_index[relation, 1].to(dtype=torch.long)
        if receiver.numel() == 0:
            continue
        allowed = legal_senders[relation].to(dtype=torch.bool)[sender]
        if edge_weight is not None:
            # Padded sparse batches use zero-weight placeholder edges.  They
            # contribute exactly zero to both numerator and denominator, so
            # remove them before gathering sender contexts; retaining them can
            # otherwise create millions of needless autograd intermediates.
            allowed = allowed & edge_weight[relation].ne(0)
        receiver = receiver[allowed]
        sender = sender[allowed]
        weights = (
            edge_weight[relation][allowed].to(dtype=node_context.dtype)
            if edge_weight is not None
            else node_context.new_ones((receiver.numel(),))
        )
        if receiver.numel() == 0:
            continue
        values = node_context[sender]
        sums = node_context.new_zeros((n_nodes, context_dim))
        counts = node_context.new_zeros((n_nodes, 1))
        sums.index_add_(0, receiver, values * weights[:, None])
        counts.index_add_(0, receiver, weights[:, None])
        output[relation] = torch.where(counts > 0, sums / counts.clamp_min(1.0), sums)
    return output


class V207TransparentMMGTGNNWR(nn.Module):
    """Frozen-embedding fusion with a transparent bounded local coefficient head.

    Prediction is exactly ``[1, x] @ beta``. No unconstrained residual branch is
    present. The caller supplies fold-specific legal sender masks.

    预测严格等于 ``[1, x] @ beta``，不含无约束残差分支；合法发送者掩码由
    当前空间折提供。
    """

    def __init__(
        self,
        modality_names: Sequence[str],
        numeric_dim: int,
        relation_count: int,
        quality_dim: int,
        space_time_dim: int,
        embedding_dim: int = 768,
        latent_dim: int = 32,
        context_dim: int = 64,
        c_beta: float = 0.25,
    ) -> None:
        super().__init__()
        if not modality_names:
            raise ValueError("at least one modality is required")
        if numeric_dim < 1 or relation_count < 1:
            raise ValueError("numeric_dim and relation_count must be positive")

        self.modality_names = tuple(modality_names)
        self.numeric_dim = int(numeric_dim)
        self.relation_count = int(relation_count)
        self.modalities = nn.ModuleDict(
            {
                name: FoldFittedModality(embedding_dim, latent_dim, quality_dim)
                for name in self.modality_names
            }
        )
        fusion_in = len(self.modality_names) * latent_dim + len(self.modality_names)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        coefficient_in = context_dim * (1 + relation_count) + space_time_dim
        self.coefficient_delta = nn.Sequential(
            nn.Linear(coefficient_in, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, numeric_dim + 1),
        )
        self.beta_anchor = nn.Parameter(torch.zeros(numeric_dim + 1))
        self.register_buffer("c_beta", torch.tensor(float(c_beta)), persistent=True)

    def forward(
        self,
        numeric_x: torch.Tensor,
        embeddings: Mapping[str, torch.Tensor],
        qualities: Mapping[str, torch.Tensor],
        availability: Mapping[str, torch.Tensor],
        adjacency: torch.Tensor,
        legal_senders: torch.Tensor,
        space_time: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        gated_latents: list[torch.Tensor] = []
        gates: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            gated, gate = self.modalities[name](
                embeddings[name], qualities[name], availability[name]
            )
            gated_latents.append(gated)
            gates[name] = gate

        context = self.fusion(torch.cat([*gated_latents, *gates.values()], dim=1))
        if edge_index is None:
            relation_messages = legal_relation_mean(context, adjacency, legal_senders)
        else:
            relation_messages = legal_relation_mean_from_edges(
                context,
                edge_index,
                legal_senders,
                relation_count=self.relation_count,
                edge_weight=edge_weight,
            )
        coefficient_input = torch.cat(
            [context, relation_messages.transpose(0, 1).flatten(start_dim=1), space_time],
            dim=1,
        )
        beta = self.beta_anchor + self.c_beta * torch.tanh(
            self.coefficient_delta(coefficient_input)
        )
        design = torch.cat([torch.ones_like(numeric_x[:, :1]), numeric_x], dim=1)
        contributions = design * beta
        prediction = contributions.sum(dim=1)
        return {
            "prediction": prediction,
            "beta": beta,
            "contributions": contributions,
            "gates": gates,
            "context": context,
            "relation_messages": relation_messages,
        }
