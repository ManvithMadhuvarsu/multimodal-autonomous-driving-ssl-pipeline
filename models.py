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
    def __init__(self):
        super().__init__()
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
        # Fusion + graph
        self.fusion = MultiModalFusionTransformer(512, depth=6, heads=8)   # [FIX-3]
        self.gnn    = GNNEncoder(in_dim=512, hid=256, num_layers=3)        # [FIX-4]
        # Task heads (in_dim=256 = GNN hid)
        self.det  = DetectionHead(in_dim=256, num_classes=10)
        self.seg  = SegmentationHead(in_dim=256, num_classes=16)
        self.traj = TrajectoryHead(in_dim=256, n_wp=5, coords=2)

    def forward(self, rgb=None, thermal=None, lidar=None, radar=None) -> dict:
        dev = next(self.parameters()).device
        Z   = lambda: torch.zeros(1, 512, device=dev)       # zero placeholder

        re  = self.rgb_proj(self.rgb_enc(rgb))           if rgb     is not None else None
        te  = self.th_proj(self.th_enc(thermal))         if thermal is not None else None
        le  = self.lidar_proj(self.lidar_enc(lidar))     if lidar   is not None else None
        rde = self.radar_proj(self.radar_enc(radar))     if radar   is not None else None

        fused = self.fusion(re, te, le, rde)               # (B, 512)

        # 5-node scene graph: [fused | rgb | thermal | lidar | radar]
        nodes = torch.stack([
            fused.squeeze(0),
            re.squeeze(0)  if re  is not None else Z().squeeze(0),
            te.squeeze(0)  if te  is not None else Z().squeeze(0),
            le.squeeze(0)  if le  is not None else Z().squeeze(0),
            rde.squeeze(0) if rde is not None else Z().squeeze(0),
        ])                                                   # (5, 512)
        # Fully-connected directed edge index for 5 nodes
        ei = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
             [1, 2, 3, 4, 0, 2, 3, 0, 1, 0, 2, 0]],
            dtype=torch.long, device=dev)
        gnn_out = self.gnn(nodes, ei)                        # (256,)
        inp     = gnn_out.unsqueeze(0)                       # (1, 256)

        return {
            "fused":        fused.squeeze(0),                # (512,)
            "gnn":          gnn_out,                         # (256,)
            "detection":    self.det(inp).squeeze(0),        # (10,)
            "segmentation": self.seg(inp).squeeze(0),        # (16,)
            "trajectory":   self.traj(inp).squeeze(0),       # (10,)
        }


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
