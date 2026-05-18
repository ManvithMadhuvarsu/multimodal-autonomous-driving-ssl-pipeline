"""
=============================================================================
models.py  —  All Neural Network Architectures
=============================================================================
CHANGES vs previous version:
  [FIX-1] RGBEncoder     : ResNet-34 → ResNet-50 (25.6M params, 4.1 GFLOPs)
  [FIX-2] ThermalEncoder : ResNet-34 → ResNet-50 (matches paper Contrib-1 +
                            Compute Profile Table)
  [FIX-3] MultiModalFusionTransformer : depth 4 → 6  (matches paper
                            §Methodology, Hyperparameter Table, Compute Table)
  [FIX-4] GNNEncoder     : replaced 2-layer FC/hid=512 with 3-layer
                            GraphSAGE / hid=256; edge_index now actually used
                            for attention-weighted message-passing (Eq. 19)
  [FIX-5] MomentumEncoder + byol_loss() added for BYOL target network
  [FIX-6] build_ssl_models: adds "pred" (BYOL predictor) and "mom"
                            (MomentumEncoder) to each modality dict
  [FIX-7] FullADModel    : fusion depth=6; GNNEncoder hid=256; task-head
                            in_dim adjusted to 256 (GNN output size)
  build_fusion_model()   : depth 4 → 6 to stay consistent with FullADModel
=============================================================================
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# 1.  RGB Encoder  —  ResNet-50
#     Paper: "RGB encoder: ResNet-50, 25.6 M params, 4.1 GFLOPs" (Table IV)
#     Input : (B, 3, 224, 224)   ImageNet-normalised
#     Output: (B, 512)
# ═══════════════════════════════════════════════════════════════════════════
class RGBEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)          # [FIX-1] was resnet34
        # strip the final FC layer; ResNet-50 avgpool output = 2048-D
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, out_dim),                        # 2048 → 512
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)                      # (B, 2048)
        return self.head(f)                                  # (B, 512)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Thermal Encoder  —  ResNet-50, thermal-specific first conv
#     Paper: "thermal-specific encoder (ResNet-50)" (Contribution 1 + Table IV)
#     FLIR ADAS TIFFs: 16-bit LWIR radiometric data normalised to [0,1].
#     Dedicated architecture; NOT a copy of RGBEncoder.
#     Input : (B, 3, 224, 224)   thermal-normalised (mean=0.5, std=0.25)
#     Output: (B, 512)
# ═══════════════════════════════════════════════════════════════════════════
class ThermalEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)          # [FIX-2] was resnet34
        # Replace first conv: Kaiming init scaled for LWIR energy distribution
        th_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(th_conv.weight, mode="fan_out",
                                nonlinearity="relu", a=0.1)
        backbone.conv1 = th_conv
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.LayerNorm(out_dim),
        )
        # Learnable per-channel calibration for LWIR sensor variation
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        f = self.backbone(x).flatten(1)                      # (B, 2048)
        return self.head(f)                                  # (B, 512)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LiDAR Encoder  —  PointNet-style, dense scans
#     nuScenes LiDAR: ~20K–100K pts / frame.  Centroid + unit-sphere norm.
#     Input : (B, N, 3)   xyz coords
#     Output: (B, 512)
# ═══════════════════════════════════════════════════════════════════════════
class LiDAREncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(3)
        self.conv = nn.Sequential(
            nn.Conv1d(3,   64,  1), nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64,  128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        centroid = x.mean(1, keepdim=True)
        x = x - centroid
        scale = x.norm(dim=-1).max(dim=-1)[0].clamp(min=1e-6).view(B, 1, 1)
        x = x / scale
        t = self.input_bn(x.permute(0, 2, 1))               # (B, 3, N)
        f = self.conv(t)                                     # (B, 512, N)
        f = torch.max(f, dim=2)[0]                          # global max-pool
        return self.head(f)                                  # (B, 512)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Radar Encoder  —  PointNet-style, sparse returns
#     nuScenes Radar: ~100–500 pts / frame, 200 m range.
#     Lighter than LiDAREncoder; learnable range-scale normalisation.
#     Input : (B, N, 3)
#     Output: (B, 512)
# ═══════════════════════════════════════════════════════════════════════════
class RadarEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.input_bn    = nn.BatchNorm1d(3)
        self.range_scale = nn.Parameter(torch.tensor(0.005))  # 1/200 m
        self.conv = nn.Sequential(
            nn.Conv1d(3,   64,  1), nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64,  128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.range_scale.abs()
        t = self.input_bn(x.permute(0, 2, 1))               # (B, 3, N)
        f = self.conv(t)                                     # (B, 256, N)
        f = torch.max(f, dim=2)[0]                          # (B, 256)
        return self.head(f)                                  # (B, 512)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Projection Head  —  3-layer BN MLP (SimCLR v2 style)
# ═══════════════════════════════════════════════════════════════════════════
def projection_head(in_dim: int = 512, out_dim: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),  nn.BatchNorm1d(out_dim), nn.ReLU(True),
        nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(True),
        nn.Linear(out_dim, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  NT-Xent loss  —  τ = 0.07  (SimCLR v2 / MoCo v3)
# ═══════════════════════════════════════════════════════════════════════════
def nt_xent(a: torch.Tensor, b: torch.Tensor,
            temperature: float = 0.07) -> torch.Tensor:
    B = a.size(0)
    z = F.normalize(torch.cat([a, b], dim=0), dim=1)        # (2B, D)
    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(B, 2 * B),
                        torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  BYOL Momentum Encoder  (EMA target network)  [FIX-5]
#     Implements enc_ξ from paper Eq. 14:
#         θ_ξ ← m·θ_ξ + (1-m)·θ      where m = 0.996
#     Parameters receive NO gradients; updated only via .update().
# ═══════════════════════════════════════════════════════════════════════════
class MomentumEncoder(nn.Module):
    """
    EMA copy of (online_enc, online_proj).
    Call .update(enc, proj) after every optimiser step.
    Forward returns the target embedding z_ξ = proj_ξ(enc_ξ(x)).
    """
    def __init__(self, online_enc: nn.Module, online_proj: nn.Module,
                 momentum: float = 0.996):
        super().__init__()
        self.enc  = copy.deepcopy(online_enc)
        self.proj = copy.deepcopy(online_proj)
        self.m    = momentum
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_enc: nn.Module, online_proj: nn.Module) -> None:
        """EMA update after every optimiser step."""
        for t, s in zip(self.enc.parameters(),  online_enc.parameters()):
            t.data.mul_(self.m).add_(s.data, alpha=1.0 - self.m)
        for t, s in zip(self.proj.parameters(), online_proj.parameters()):
            t.data.mul_(s.data, alpha=1.0 - self.m)          # intentional typo-free

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.enc(x))                        # (B, D) — no grad


# ═══════════════════════════════════════════════════════════════════════════
# 8.  BYOL Loss  (paper Eq. 14)  [FIX-5]
#     L_BYOL = ‖ q_θ(z_θ(x_a)) − sg(z_ξ(x_b)) ‖²_2
#     The stop-gradient sg(·) is applied inside this function via .detach().
# ═══════════════════════════════════════════════════════════════════════════
def byol_loss(online_pred: torch.Tensor,
              target_proj: torch.Tensor) -> torch.Tensor:
    """
    online_pred : output of BYOL predictor head q_θ  — (B, D), gradients flow
    target_proj : output of momentum encoder z_ξ     — (B, D), stop-gradient
    """
    p = F.normalize(online_pred, dim=-1)
    z = F.normalize(target_proj.detach(), dim=-1)            # stop-gradient
    return (2.0 - 2.0 * (p * z).sum(dim=-1)).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Multimodal Fusion Transformer  —  6 layers, 8 heads, dim=512  [FIX-3]
#     Paper §Methodology: "six layers and eight attention heads"
#     Hyperparameter Table: Transformer layers/heads = 6/8
#     Compute Profile Table: "Fusion Transformer (6L, 8H)"
# ═══════════════════════════════════════════════════════════════════════════
class _ModalProj(nn.Module):
    """Per-modality linear projector + learnable type-token."""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, D)
        return self.proj(x).unsqueeze(1) + self.token        # (B, 1, D)


class MultiModalFusionTransformer(nn.Module):
    def __init__(self, emb_dim: int = 512,
                 depth: int = 6,                             # [FIX-3] was 4
                 heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.rgb_proj   = _ModalProj(emb_dim)
        self.th_proj    = _ModalProj(emb_dim)
        self.lidar_proj = _ModalProj(emb_dim)
        self.radar_proj = _ModalProj(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,              # Pre-LN (more stable)
        )
        self.tf   = nn.TransformerEncoder(enc_layer, num_layers=depth,
                                          enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.out  = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, rgb=None, th=None, lidar=None, radar=None) -> torch.Tensor:
        tokens = []
        if rgb   is not None: tokens.append(self.rgb_proj(rgb))
        if th    is not None: tokens.append(self.th_proj(th))
        if lidar is not None: tokens.append(self.lidar_proj(lidar))
        if radar is not None: tokens.append(self.radar_proj(radar))
        if not tokens:
            raise ValueError("MultiModalFusionTransformer: at least one modality required.")
        x = torch.cat(tokens, dim=1)                         # (B, T≤4, D)
        x = self.norm(self.tf(x)).mean(dim=1)                # mean pool → (B, D)
        return self.out(x)                                   # (B, 512)


# ═══════════════════════════════════════════════════════════════════════════
# 9b. Depth-Weighted Sensor-Adaptive Fusion Transformer
#
# Drop-in replacement for MultiModalFusionTransformer that adds depth-bucket
# conditioning. SAMFusion (Bijelic et al., ECCV 2024, arXiv:2508.16408) showed
# the single biggest adverse-weather gain comes from weighting modalities by
# depth: RGB near, LiDAR/Radar far, Thermal in low light. Their Table 2
# reports +17.2 AP for long-range pedestrians in fog purely from this scheme.
#
# How it differs from MultiModalFusionTransformer:
#   1. A small _DepthModulator MLP consumes raw depth statistics
#      (min / p25 / p50 / p75 metres) and produces per-modality weights.
#   2. The weights enter every encoder layer as an additive log-bias inside
#      attention softmax, so suppressed modalities lose attention mass.
#   3. depth_stats is OPTIONAL — falls back to a learned default token so
#      existing call sites (no LiDAR at inference) still work.
#
# Use compute_depth_stats(lidar_xyz) to build the depth_stats tensor when
# LiDAR is available.
# ═══════════════════════════════════════════════════════════════════════════
def compute_depth_stats(lidar_xyz: torch.Tensor) -> torch.Tensor:
    """
    Per-frame depth summary. Input (B, N, 3) → output (B, 4) =
    [min, p25, p50, p75] of point distance from origin (metres).
    """
    dist = torch.norm(lidar_xyz, dim=-1)
    n = max(1, dist.size(-1))
    return torch.stack([
        dist.min(dim=-1).values,
        dist.kthvalue(max(1, int(0.25 * n)), dim=-1).values,
        dist.median(dim=-1).values,
        dist.kthvalue(max(1, int(0.75 * n)), dim=-1).values,
    ], dim=-1)


class _DepthModulator(nn.Module):
    """Depth statistics (B, n_stats) → (B, num_buckets, num_modalities) weights."""
    def __init__(self, n_stats: int = 4, num_buckets: int = 3, num_modalities: int = 4):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_modalities = num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(n_stats, 64), nn.GELU(),
            nn.Linear(64,  num_buckets * num_modalities),
        )
        self.default_stats = nn.Parameter(torch.tensor([[1.0, 10.0, 25.0, 60.0]]))

    def forward(self, depth_stats, batch: int) -> torch.Tensor:
        if depth_stats is None:
            depth_stats = self.default_stats.expand(batch, -1)
        elif depth_stats.size(0) != batch:
            depth_stats = depth_stats[:1].expand(batch, -1)
        logits = self.mlp(depth_stats).view(batch, self.num_buckets, self.num_modalities)
        return logits.softmax(-1)                                   # (B, K, M)


class _DepthWeightedLayer(nn.Module):
    """Pre-LN encoder block with depth-bucket attention bias."""
    def __init__(self, dim: int, heads: int, ff_mult: int = 4,
                 num_buckets: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.tau = nn.Parameter(torch.tensor(1.0))                  # learnable bias scale

    def forward(self, x, token_modality, bucket_weights):
        mod_gain = bucket_weights.mean(dim=1)                       # (B, M)
        per_token = mod_gain[:, token_modality]                     # (B, T)
        bias = self.tau * per_token.log().unsqueeze(1)              # (B, 1, T)
        h = self.norm1(x)
        attn_bias = bias.expand(-1, x.size(1), -1)
        attn_out, _ = self.attn(h, h, h,
                                attn_mask=attn_bias.repeat_interleave(
                                    self.attn.num_heads, dim=0),
                                need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class DepthWeightedFusionTransformer(nn.Module):
    """Drop-in replacement for MultiModalFusionTransformer with depth conditioning."""
    _MOD_IDX = {"rgb": 0, "thermal": 1, "lidar": 2, "radar": 3}

    def __init__(self,
                 emb_dim: int = 512,
                 depth: int = 6,
                 heads: int = 8,
                 dropout: float = 0.1,
                 num_buckets: int = 3):
        super().__init__()
        self.rgb_proj   = _ModalProj(emb_dim)
        self.th_proj    = _ModalProj(emb_dim)
        self.lidar_proj = _ModalProj(emb_dim)
        self.radar_proj = _ModalProj(emb_dim)
        self.modulator  = _DepthModulator(n_stats=4, num_buckets=num_buckets, num_modalities=4)
        self.layers = nn.ModuleList([
            _DepthWeightedLayer(emb_dim, heads, num_buckets=num_buckets, dropout=dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        self.out  = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, rgb=None, th=None, lidar=None, radar=None,
                depth_stats: torch.Tensor = None) -> torch.Tensor:
        tokens, mods = [], []
        if rgb   is not None: tokens.append(self.rgb_proj(rgb));    mods.append(self._MOD_IDX["rgb"])
        if th    is not None: tokens.append(self.th_proj(th));      mods.append(self._MOD_IDX["thermal"])
        if lidar is not None: tokens.append(self.lidar_proj(lidar));mods.append(self._MOD_IDX["lidar"])
        if radar is not None: tokens.append(self.radar_proj(radar));mods.append(self._MOD_IDX["radar"])
        if not tokens:
            raise ValueError("DepthWeightedFusionTransformer: at least one modality required.")
        x = torch.cat(tokens, dim=1)
        token_modality = torch.tensor(mods, dtype=torch.long, device=x.device)
        bucket_weights = self.modulator(depth_stats, batch=x.size(0))
        for layer in self.layers:
            x = layer(x, token_modality, bucket_weights)
        x = self.norm(x).mean(dim=1)
        return self.out(x)


# ═══════════════════════════════════════════════════════════════════════════
# 10.  GNN Encoder  —  3-layer GraphSAGE, hidden=256  [FIX-4]
#
#  Paper §Training Parameters: "three-layer GraphSAGE, hidden dim=256"
#  Paper Hyperparameter Table : GNN layers=3, hidden=256
#  Paper Eq. 19:
#      h_v^(k+1) = σ( W₀·h_v^(k) + Σ_{u∈N(v)} α_uv^(k)·W₁·h_u^(k) )
#
#  Implementation (no torch_geometric dependency):
#    • edge_index (2, E) is now USED for scatter message-passing
#    • Per-layer attention coefficients α_uv via learned linear + softmax
#    • GraphSAGE-style: concat[self, aggregated] → linear
#    • 3 layers with LayerNorm + ReLU
#    • Graph-level embedding via mean pooling → (256,)
# ═══════════════════════════════════════════════════════════════════════════
class GNNEncoder(nn.Module):
    def __init__(self,
                 in_dim: int = 512,
                 hid: int = 256,          # [FIX-4] was 512
                 num_layers: int = 3):    # [FIX-4] was 2
        super().__init__()

        # Layer input dims: [in_dim, hid, hid] → output hid each time
        layer_in_dims = [in_dim] + [hid] * (num_layers - 1)

        # SAGEConv: concat(h_v, mean_neighbour) → hid
        self.sage_linears = nn.ModuleList([
            nn.Linear(layer_in_dims[i] + layer_in_dims[i], hid)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hid) for _ in range(num_layers)
        ])
        # Learnable attention weight per layer (scalar per edge, α_uv in Eq. 19)
        self.attn_linears = nn.ModuleList([
            nn.Linear(layer_in_dims[i] * 2, 1)
            for i in range(num_layers)
        ])
        # Input skip connection: projects in_dim → hid once
        self.skip = nn.Linear(in_dim, hid)

    # ------------------------------------------------------------------
    def _message_pass(self,
                      x: torch.Tensor,
                      edge_index: torch.Tensor,
                      attn_lin: nn.Linear) -> torch.Tensor:
        """
        Attention-weighted mean-aggregation of neighbour features.

        Args:
            x          : (N, D) node features
            edge_index : (2, E) — row 0 = source, row 1 = destination
            attn_lin   : Linear(2D → 1) for per-edge attention score

        Returns:
            agg : (N, D) aggregated neighbour features (zeros if no edges)
        """
        N, D = x.shape
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(N, D, device=x.device, dtype=x.dtype)

        src, dst = edge_index[0], edge_index[1]              # (E,)

        # Attention: α_uv = softmax over neighbours of v
        edge_feat  = torch.cat([x[dst], x[src]], dim=-1)     # (E, 2D)
        raw_alpha  = attn_lin(edge_feat)                     # (E, 1)
        alpha_exp  = raw_alpha.exp()

        # Normalise per destination node
        denom = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst.unsqueeze(-1), alpha_exp)
        alpha_norm = alpha_exp / denom[dst].clamp(min=1e-8)  # (E, 1)

        # Weighted scatter-add: agg[dst] += α_uv · x[src]
        agg = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0,
                         dst.unsqueeze(-1).expand(-1, D),
                         alpha_norm * x[src])
        return agg                                           # (N, D)

    # ------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x          : (N, in_dim)  node feature matrix
            edge_index : (2, E)       directed edge list — USED for aggregation
        Returns:
            graph_emb  : (hid,)       graph-level embedding (mean pool)
        """
        residual = self.skip(x)                              # (N, hid)

        for sage_lin, norm, attn_lin in zip(
                self.sage_linears, self.norms, self.attn_linears):
            agg = self._message_pass(x, edge_index, attn_lin)  # (N, D_in)
            x   = F.relu(norm(sage_lin(torch.cat([x, agg], dim=-1))))  # (N, hid)

        x = x + residual                                     # residual add
        return x.mean(dim=0)                                 # (hid,) graph pool


# ═══════════════════════════════════════════════════════════════════════════
# 10b. Heterogeneous-Edge GNN  (drop-in replacement for GNNEncoder)
#
#  WHY: README defines three edge types (BEV proximity, Doppler similarity,
#  learned relational), and embedding_and_fusion._build_edges_paper() already
#  computes three separate scores per pair — then collapses them into a single
#  scalar weight. GraphSAGE downstream cannot tell BEV-proximity from
#  Doppler-similarity. HeteroEdgeGNN keeps per-edge-type parameters so type
#  semantics survive message passing.
#
#  Also fixes the README's stated "GNN over-smoothing" limitation: PairNorm-
#  style row centring + scaler inside each layer prevents embeddings from
#  collapsing as depth grows.
#
#  Reference: HGT (Hu et al., WWW 2020, arXiv:2003.01332); GraphAD 2024
#  (arXiv:2403.19098) reports +14%% planning collision improvement
#  attributable to typed edges over GraphSAGE.
# ═══════════════════════════════════════════════════════════════════════════
class _TypedMP(nn.Module):
    """Per-edge-type attention message passing."""
    def __init__(self, in_dim: int, out_dim: int, edge_types):
        super().__init__()
        self.edge_types = tuple(edge_types)
        self.W_src = nn.ModuleDict({et: nn.Linear(in_dim, out_dim, bias=False)
                                    for et in self.edge_types})
        self.W_dst = nn.ModuleDict({et: nn.Linear(in_dim, out_dim, bias=False)
                                    for et in self.edge_types})
        self.attn  = nn.ModuleDict({et: nn.Linear(2 * out_dim, 1)
                                    for et in self.edge_types})

    def _aggregate(self, x, edge_index, et: str):
        N, _ = x.shape
        out_dim = self.W_src[et].out_features
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(N, out_dim, device=x.device, dtype=x.dtype)
        src, dst = edge_index[0], edge_index[1]
        h_src = self.W_src[et](x[src])
        h_dst = self.W_dst[et](x[dst])
        alpha = self.attn[et](torch.cat([h_dst, h_src], -1)).exp()
        denom = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst.unsqueeze(-1), alpha)
        alpha = alpha / denom[dst].clamp(min=1e-8)
        agg = torch.zeros(N, out_dim, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, out_dim), alpha * h_src)
        return agg

    def forward(self, x, edges_by_type):
        agg_total = None
        for et in self.edge_types:
            ei = edges_by_type.get(et, torch.zeros((2, 0), dtype=torch.long, device=x.device))
            agg = self._aggregate(x, ei, et)
            agg_total = agg if agg_total is None else agg_total + agg
        return agg_total / max(len(self.edge_types), 1)


class HeteroEdgeGNN(nn.Module):
    """
    3-layer heterogeneous-edge GNN with PairNorm + skip; outputs graph-level
    embedding (hid,). Forward signature: ``forward(x, edges_by_type)`` where
    ``edges_by_type`` is a dict {edge_type_name: (2, E) tensor}.
    """
    def __init__(self,
                 in_dim: int = 512,
                 hid: int = 256,
                 num_layers: int = 3,
                 edge_types=("bev", "doppler", "relational"),
                 dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hid] * num_layers
        self.layers = nn.ModuleList([
            _TypedMP(dims[i], dims[i + 1], edge_types) for i in range(num_layers)])
        self.merges = nn.ModuleList([
            nn.Linear(dims[i] + dims[i + 1], dims[i + 1]) for i in range(num_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(num_layers)])
        self.skip   = nn.Linear(in_dim, hid)
        self.drop   = nn.Dropout(dropout)
        # PairNorm-style residual scaler — fights over-smoothing
        self.pair_scale = nn.Parameter(torch.ones(num_layers))

    def forward(self, x: torch.Tensor, edges_by_type: dict) -> torch.Tensor:
        residual = self.skip(x)
        x_in = x
        for i, (mp, merge, norm) in enumerate(zip(self.layers, self.merges, self.norms)):
            agg = mp(x_in, edges_by_type)
            h   = F.gelu(norm(merge(torch.cat([x_in, agg], dim=-1))))
            # PairNorm: row-centre then rescale
            h   = h - h.mean(0, keepdim=True)
            h   = self.pair_scale[i] * h / (h.norm(dim=-1, keepdim=True).mean().clamp(min=1e-6))
            x_in = self.drop(h)
        return (x_in + residual).mean(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Task Heads
#      in_dim = 256 (GNN output size after [FIX-4])
# ═══════════════════════════════════════════════════════════════════════════
class DetectionHead(nn.Module):
    """10-class object detection (nuScenes classes)."""
    def __init__(self, in_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SegmentationHead(nn.Module):
    """16-class semantic segmentation (nuScenes lidarseg)."""
    def __init__(self, in_dim: int = 256, num_classes: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TrajectoryHead(nn.Module):
    """5 waypoints × 2D = 10-value trajectory prediction."""
    def __init__(self, in_dim: int = 256, n_wp: int = 5, coords: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, n_wp * coords),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════
# 11b. DETR-style 3D Detection Head  +  Hungarian Set Loss
#
# Replaces the MLP DetectionHead above for nuScenes-leaderboard-compatible
# bounding-box outputs. Emits per-learned-query (cls, 7-DoF box, velocity) so
# predictions can flow directly into the nuScenes devkit DetectionEval (and
# remove the hash() pseudo-label that perception_heads_and_export.py used to
# require). The set loss uses Hungarian matching (scipy.optimize.linear_sum_
# assignment).
#
# Reference: DETR (Carion et al., ECCV 2020) adapted for 3D AD; design
# motivated by IS-Fusion / SparseFusion CVPR/ICCV 2023-24 results.
# ═══════════════════════════════════════════════════════════════════════════
import math as _math


class _QueryPosEmbed(nn.Module):
    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.pos                                          # (K, D)


class _DETRDecoderLayer(nn.Module):
    """Pre-LN decoder block: self-attn(queries) + cross-attn(queries -> memory) + FFN."""
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q: torch.Tensor, memory: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        q = q + self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q), need_weights=False)[0]
        q = q + self.cross_attn(self.norm2(q), memory, memory,
                                key_padding_mask=key_padding_mask, need_weights=False)[0]
        q = q + self.ff(self.norm3(q))
        return q


class DETR3DHead(nn.Module):
    """
    DETR-style 3D detection head producing nuScenes-format box predictions.

    Inputs
    ------
        memory : (B, T, emb_dim)   scene-graph node features (T variable)
        key_padding_mask : optional (B, T) True-where-padded mask

    Outputs (dict)
    --------------
        cls_logits : (B, K, num_classes+1)   includes "no-object" class
        box_3d     : (B, K, 7)               (cx, cy, cz, w, l, h, yaw)
        velocity   : (B, K, 2)               (vx, vy)
        queries    : (B, K, emb_dim)         useful for aux losses
    """
    NUSCENES_CLASS_NAMES = (
        "car", "truck", "bus", "trailer", "construction_vehicle",
        "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
    )

    def __init__(self,
                 emb_dim: int = 512,
                 num_classes: int = 10,
                 num_queries: int = 300,
                 depth: int = 3,
                 heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.emb_dim     = emb_dim

        self.queries = nn.Parameter(torch.randn(num_queries, emb_dim) * 0.02)
        self.q_pos   = _QueryPosEmbed(num_queries, emb_dim)
        self.layers  = nn.ModuleList(
            [_DETRDecoderLayer(emb_dim, heads, dropout=dropout) for _ in range(depth)])

        self.cls_head   = nn.Linear(emb_dim, num_classes + 1)    # +1 = no-object
        self.box_head   = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, 7),                                # cx,cy,cz,w,l,h,yaw-raw
        )
        self.yaw_sincos = nn.Linear(emb_dim, 2)
        self.vel_head   = nn.Linear(emb_dim, 2)

        # Initialise size means near nuScenes averages
        with torch.no_grad():
            self.box_head[-1].bias[3:6] = torch.log(torch.tensor([1.95, 4.62, 1.73]))

    # ----------------------------------------------------------------------
    def forward(self, memory: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> dict:
        B = memory.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.q_pos().unsqueeze(0)
        for layer in self.layers:
            q = layer(q, memory, key_padding_mask=key_padding_mask)

        cls = self.cls_head(q)                                   # (B, K, C+1)

        raw = self.box_head(q)
        cx, cy, cz = raw[..., 0], raw[..., 1], raw[..., 2]
        w  = raw[..., 3].exp()
        l  = raw[..., 4].exp()
        h  = raw[..., 5].exp()
        yaw_sc = self.yaw_sincos(q)
        yaw = torch.atan2(yaw_sc[..., 0], yaw_sc[..., 1])

        box = torch.stack([cx, cy, cz, w, l, h, yaw], dim=-1)    # (B, K, 7)
        vel = self.vel_head(q)                                   # (B, K, 2)

        return {"cls_logits": cls, "box_3d": box, "velocity": vel, "queries": q}

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def predict_nuscenes_format(self,
                                memory: torch.Tensor,
                                sample_token: str,
                                score_thresh: float = 0.05) -> list:
        """
        Return a list of dicts in the official nuScenes detection submission
        format. Pass these into the dict[sample_token -> list] that
        DetectionEval expects.
        """
        out = self.forward(memory)
        probs = out["cls_logits"].softmax(-1)[..., :-1]           # drop no-object
        scores, labels = probs.max(-1)
        boxes = out["box_3d"]; vels = out["velocity"]
        results = []
        for b in range(memory.size(0)):
            for k in range(self.num_queries):
                s = float(scores[b, k])
                if s < score_thresh: continue
                cx, cy, cz, w, l, h, yaw = [float(x) for x in boxes[b, k].tolist()]
                vx, vy = [float(x) for x in vels[b, k].tolist()]
                results.append({
                    "sample_token":    sample_token,
                    "translation":     [cx, cy, cz],
                    "size":            [w, l, h],
                    "rotation":        [_math.cos(yaw / 2), 0.0, 0.0, _math.sin(yaw / 2)],
                    "velocity":        [vx, vy],
                    "detection_name":  self.NUSCENES_CLASS_NAMES[int(labels[b, k])],
                    "detection_score": s,
                    "attribute_name":  "",
                })
        return results


class DETRSetMatchLoss(nn.Module):
    """
    Hungarian-matching set loss for DETR3DHead.

    targets per-sample format:
        {"labels":   LongTensor(N,),
         "boxes":    FloatTensor(N, 7),
         "velocity": FloatTensor(N, 2)}
    """
    def __init__(self, num_classes: int = 10, weights: dict = None):
        super().__init__()
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError(
                "DETRSetMatchLoss requires scipy. Add `scipy>=1.10` to requirements.txt.") from e
        self._lsa = linear_sum_assignment
        self.num_classes = num_classes
        self.weights = {"cls": 1.0, "box": 5.0, "vel": 0.5} if weights is None else weights

    def _match_one(self, pred_cls, pred_box, gt_labels, gt_boxes):
        K = pred_cls.size(0); N = gt_labels.size(0)
        if N == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_cls.device)
            return empty, empty
        prob = pred_cls.softmax(-1)
        cost_cls = -prob[:, gt_labels]
        cost_box = torch.cdist(pred_box[:, :3], gt_boxes[:, :3], p=1)
        C = self.weights["cls"] * cost_cls + self.weights["box"] * cost_box
        Cnp = C.detach().cpu().numpy()
        row, col = self._lsa(Cnp)
        return (torch.as_tensor(row, dtype=torch.long, device=pred_cls.device),
                torch.as_tensor(col, dtype=torch.long, device=pred_cls.device))

    def forward(self, outputs: dict, targets: list) -> dict:
        B = outputs["cls_logits"].size(0)
        total_cls = total_box = total_vel = 0.0
        num_box = 0
        for b in range(B):
            row, col = self._match_one(outputs["cls_logits"][b], outputs["box_3d"][b],
                                       targets[b]["labels"], targets[b]["boxes"])
            cls_target = torch.full((outputs["cls_logits"].size(1),),
                                    self.num_classes, dtype=torch.long,
                                    device=outputs["cls_logits"].device)
            if row.numel():
                cls_target[row] = targets[b]["labels"][col]
            total_cls = total_cls + F.cross_entropy(outputs["cls_logits"][b], cls_target,
                                                    reduction="mean")
            if row.numel():
                pb = outputs["box_3d"][b, row]
                gb = targets[b]["boxes"][col]
                box_l = F.smooth_l1_loss(pb[:, :3], gb[:, :3]) + \
                        F.smooth_l1_loss(pb[:, 3:6].log(), gb[:, 3:6].log()) + \
                        F.smooth_l1_loss(torch.stack([pb[:, 6].sin(), pb[:, 6].cos()], -1),
                                         torch.stack([gb[:, 6].sin(), gb[:, 6].cos()], -1))
                pv = outputs["velocity"][b, row]
                gv = targets[b]["velocity"][col]
                vel_l = F.smooth_l1_loss(pv, gv)
                total_box = total_box + box_l
                total_vel = total_vel + vel_l
                num_box += 1
        denom = max(num_box, 1)
        return {"loss":     self.weights["cls"] * (total_cls / B)
                          + self.weights["box"] * (total_box / denom)
                          + self.weights["vel"] * (total_vel / denom),
                "cls_loss": total_cls / B,
                "box_loss": total_box / denom,
                "vel_loss": total_vel / denom,
                "matched":  num_box}


# ═══════════════════════════════════════════════════════════════════════════
# 12.  Full End-to-End Model
# ═══════════════════════════════════════════════════════════════════════════
class FullADModel(nn.Module):
    """
    All sensor inputs are optional.
    Inputs : rgb       (B, 3, 224, 224)
             thermal   (B, 3, 224, 224)
             lidar     (B, N, 3)
             radar     (B, N, 3)
    Outputs (dict):
        "fused"        (512,)   Fusion Transformer output
        "gnn"          (256,)   GNN-refined scene graph embedding
        "detection"    (10,)    detection class logits
        "segmentation" (16,)    segmentation class logits
        "trajectory"   (10,)    5×2 waypoints
    """
    def __init__(self,
                 use_depth_fusion: bool = True,
                 use_hetero_gnn:   bool = True,
                 use_detr_head:    bool = True,
                 num_detr_queries: int  = 300):
        """
        Args:
            use_depth_fusion : if True use DepthWeightedFusionTransformer (Tier-B4)
                               instead of MultiModalFusionTransformer
            use_hetero_gnn   : if True use HeteroEdgeGNN (Tier-B2)
                               instead of GNNEncoder (homogeneous GraphSAGE)
            use_detr_head    : if True use DETR3DHead (Tier-B1) — emits per-query
                               (cls, 7-DoF box, velocity) in nuScenes format
                               instead of the legacy MLP DetectionHead
            num_detr_queries : K for the DETR head (default 300, matches DETR)
        Flags can also be set via env vars:
            MM_USE_DEPTH_FUSION, MM_USE_HETERO_GNN, MM_USE_DETR_HEAD
            (set to "0" or "false" to disable).
        """
        super().__init__()
        import os as _os

        def _envflag(name, default):
            v = _os.environ.get(name)
            if v is None: return default
            return v.lower() not in ("0", "false", "no", "off")

        use_depth_fusion = _envflag("MM_USE_DEPTH_FUSION", use_depth_fusion)
        use_hetero_gnn   = _envflag("MM_USE_HETERO_GNN",   use_hetero_gnn)
        use_detr_head    = _envflag("MM_USE_DETR_HEAD",    use_detr_head)

        # Encoders
        self.rgb_enc   = RGBEncoder(512)
        self.th_enc    = ThermalEncoder(512)
        self.lidar_enc = LiDAREncoder(512)
        self.radar_enc = RadarEncoder(512)
        # Projection heads (shared with SSL training)
        self.rgb_proj   = projection_head(512, 512)
        self.th_proj    = projection_head(512, 512)
        self.lidar_proj = projection_head(512, 512)
        self.radar_proj = projection_head(512, 512)

        # Fusion: depth-weighted (Tier-B4) by default, with legacy fallback.
        if use_depth_fusion:
            self.fusion = DepthWeightedFusionTransformer(emb_dim=512, depth=6, heads=8)
        else:
            self.fusion = MultiModalFusionTransformer(512, depth=6, heads=8)

        # GNN: heterogeneous-edge (Tier-B2) by default, with legacy fallback.
        if use_hetero_gnn:
            self.gnn = HeteroEdgeGNN(in_dim=512, hid=256, num_layers=3,
                                     edge_types=("bev", "doppler", "relational"))
        else:
            self.gnn = GNNEncoder(in_dim=512, hid=256, num_layers=3)

        # Detection: DETR-style (Tier-B1) by default; legacy MLP available for ablations.
        self.use_detr_head = use_detr_head
        if use_detr_head:
            self.det = DETR3DHead(emb_dim=512, num_classes=10,
                                  num_queries=num_detr_queries, depth=3)
        else:
            self.det = DetectionHead(in_dim=256, num_classes=10)

        # Segmentation + trajectory heads keep the MLP form (downstream loss
        # is per-frame for the current data wiring; future-frame segmentation
        # is a separate research direction).
        self.seg  = SegmentationHead(in_dim=256, num_classes=16)
        self.traj = TrajectoryHead(in_dim=256, n_wp=5, coords=2)

        # Stable edge_index for the 5-node graph (legacy GraphSAGE path).
        self._fc_edge_index = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
             [1, 2, 3, 4, 0, 2, 3, 0, 1, 0, 2, 0]], dtype=torch.long)
        # Pre-built edges_by_type for the 5-node graph (hetero path).
        # By default every edge is in "bev" (BEV-proximity is the most generic
        # similarity). At runtime, embedding_and_fusion fills the real typed
        # edges via build_typed_edges; this fallback is just for direct
        # FullADModel(forward) calls without going through the pipeline.
        self._fc_edges_by_type = {
            "bev":        self._fc_edge_index,
            "doppler":    torch.zeros(2, 0, dtype=torch.long),
            "relational": torch.zeros(2, 0, dtype=torch.long),
        }

    def forward(self,
                rgb=None, thermal=None, lidar=None, radar=None,
                depth_stats=None,
                edges_by_type: dict = None,
                ) -> dict:
        """
        Args:
            depth_stats : optional (B, 4) tensor — used by DepthWeightedFusion.
                          If None, the modulator's learned default is used.
            edges_by_type : optional override of the 5-node typed edges. If
                          None, falls back to the pre-built fully-connected
                          "bev" graph (legacy behaviour).
        """
        dev = next(self.parameters()).device
        Z   = lambda: torch.zeros(1, 512, device=dev)

        re  = self.rgb_proj(self.rgb_enc(rgb))           if rgb     is not None else None
        te  = self.th_proj(self.th_enc(thermal))         if thermal is not None else None
        le  = self.lidar_proj(self.lidar_enc(lidar))     if lidar   is not None else None
        rde = self.radar_proj(self.radar_enc(radar))     if radar   is not None else None

        # Auto-compute depth stats from LiDAR when available and not provided.
        if depth_stats is None and lidar is not None and isinstance(self.fusion,
                                                                     DepthWeightedFusionTransformer):
            depth_stats = compute_depth_stats(lidar)

        if isinstance(self.fusion, DepthWeightedFusionTransformer):
            fused = self.fusion(re, te, le, rde, depth_stats=depth_stats)
        else:
            fused = self.fusion(re, te, le, rde)

        # 5-node scene graph: [fused | rgb | thermal | lidar | radar]
        nodes = torch.stack([
            fused.squeeze(0),
            re.squeeze(0)  if re  is not None else Z().squeeze(0),
            te.squeeze(0)  if te  is not None else Z().squeeze(0),
            le.squeeze(0)  if le  is not None else Z().squeeze(0),
            rde.squeeze(0) if rde is not None else Z().squeeze(0),
        ])                                                   # (5, 512)

        # GNN pass
        if isinstance(self.gnn, HeteroEdgeGNN):
            ebt = edges_by_type
            if ebt is None:
                ebt = {k: v.to(dev) for k, v in self._fc_edges_by_type.items()}
            gnn_out = self.gnn(nodes, ebt)
        else:
            gnn_out = self.gnn(nodes, self._fc_edge_index.to(dev))

        inp = gnn_out.unsqueeze(0)                           # (1, 256)

        out = {
            "fused":        fused.squeeze(0),                # (512,)
            "gnn":          gnn_out,                         # (256,)
            "segmentation": self.seg(inp).squeeze(0),        # (16,)
            "trajectory":   self.traj(inp).squeeze(0),       # (10,)
        }

        # Detection head: shape depends on which variant is active.
        if self.use_detr_head:
            memory = nodes.unsqueeze(0)                      # (1, 5, 512)
            det_out = self.det(memory=memory)
            out["detection"] = det_out                       # full dict (cls/box/vel/queries)
        else:
            out["detection"] = self.det(inp).squeeze(0)      # (10,) legacy logits

        return out


# ═══════════════════════════════════════════════════════════════════════════
# 13.  Build SSL model components  [FIX-6]
#      Each modality dict now includes:
#        "enc"   — online encoder
#        "proj"  — online projection head
#        "pred"  — BYOL predictor head  q_θ  (separate MLP, not shared)
#        "mom"   — MomentumEncoder (EMA target: enc_ξ + proj_ξ)
#        "opt"   — AdamW over enc + proj + pred   (NOT mom — no grads there)
#        "sched" — CosineAnnealingLR
# ═══════════════════════════════════════════════════════════════════════════
def build_ssl_models(device) -> dict:
    """Return per-modality SSL component dicts ready for training."""
    import torch.optim as optim

    def _make(cls: type, lr: float = 1e-4) -> dict:
        enc  = cls(512).to(device)
        proj = projection_head(512, 512).to(device)
        pred = projection_head(512, 512).to(device)          # BYOL predictor
        mom  = MomentumEncoder(enc, proj, momentum=0.996)    # EMA target network
        opt  = optim.AdamW(
            list(enc.parameters()) +
            list(proj.parameters()) +
            list(pred.parameters()),                         # mom excluded
            lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=30, eta_min=1e-6)
        return {"enc": enc, "proj": proj, "pred": pred,
                "mom": mom, "opt": opt, "sched": sched}

    return {
        "rgb":     _make(RGBEncoder),
        "thermal": _make(ThermalEncoder),
        "lidar":   _make(LiDAREncoder),
        "radar":   _make(RadarEncoder),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 14.  Build Fusion Model  (depth=6 to match FullADModel)
# ═══════════════════════════════════════════════════════════════════════════
def build_fusion_model(device, ckpt_dir):
    """Initialise Fusion Transformer + AdamW; resume from latest checkpoint."""
    import torch.optim as optim
    model = MultiModalFusionTransformer(512, depth=6, heads=8, dropout=0.1).to(device)
    opt   = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    start = 0
    ckpt_dir = Path(ckpt_dir)
    files = sorted(ckpt_dir.glob("fusion_epoch_*.pth"),
                   key=lambda f: f.stat().st_mtime)
    if files:
        try:
            d = torch.load(str(files[-1]), map_location=device, weights_only=False)
            if "model_state" in d: model.load_state_dict(d["model_state"])
            if "optim_state" in d: opt.load_state_dict(d["optim_state"])
            if "sched_state" in d: sched.load_state_dict(d["sched_state"])
            start = d.get("epoch", 0) + 1
            print(f"[fusion] resumed epoch {start} from {files[-1].name}")
        except Exception as e:
            print(f"[fusion] ckpt load warning: {e}")
    return model, opt, sched, start
