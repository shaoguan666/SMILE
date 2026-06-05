import math
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from utils.utils import length_to_mask


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.mlp(self.norm2(x))
        return x


class SeqAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, lens, lens_mask):
        B, N, C = x.shape  # B*I, T, H
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        mask = mask.reshape(-1, N)
        mask = ((mask.unsqueeze(-1) + mask.unsqueeze(-2))).reshape(-1, 1, mask.shape[1], mask.shape[1]).repeat(1, self.num_heads, 1, 1)
        mask += (~(lens_mask.unsqueeze(-2) * lens_mask.unsqueeze(-1))).reshape(-1, 1, lens_mask.shape[1], lens_mask.shape[1]).repeat(1, self.num_heads, 1, 1).float() * -1e9
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqAttBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

    def forward(self, x, mask, lens, lens_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, mask, lens, lens_mask)
        x = torch.reshape(x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + x
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, lens, lens_mask):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)

        q = q[:, 0]
        mask = mask.reshape(B, N, P, 1, 1).repeat(1, 1, 1, self.num_heads,
                                self.head_dim).permute(0, 2, 3, 1, 4)
        k = k.masked_fill(~mask.bool(), 0).sum(dim=1) / mask.sum(dim=1)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

    def forward(self, x, mask, lens, lens_mask):
        x = x + self.attn_var(self.norm1(x), mask, lens, lens_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(-2)].clone().detach()


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, proj_drop=proj_drop,
                                         norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, proj_drop=proj_drop,
                                         norm_layer=norm_layer)

        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, 
                                    proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, mask, lens, lens_mask):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        x = self.var_att_block(x, mask, lens, lens_mask)
        x = self.mlp(x)
        return x


class MLPEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, mask):
        x = torch.stack((x, mask), dim=-1)
        x = self.embed(x)
        x = x.permute(0, 2, 1, 3)
        return x


class DensityMLPEmbedder(nn.Module):
    """MLPEmbedder with local observation density as a third input feature.

    Concatenates (value, mask, density) along the last dim and projects to
    d_model.  Density is the sliding-window observation rate computed from
    original_mask externally and passed in as (B, T, V).

    Compared to additive injection (x += obs_density_emb), fusing density as
    an input feature forces the model to jointly encode value, missingness, and
    measurement frequency from the very first projection.

    Input:
        x       (B, T, V)  -- observed values
        mask    (B, T, V)  -- observation mask
        density (B, T, V)  -- local obs density from avg_pool1d over mask
    Output: (B, V, T, d_model)
    """

    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, mask, density):
        inp = torch.stack((x, mask, density), dim=-1)  # (B, T, V, 3)
        out = self.embed(inp)                           # (B, T, V, d_model)
        return out.permute(0, 2, 1, 3)                 # (B, V, T, d_model)


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False,
                        mlp_ratio=4., proj_drop=args.dropout) for l in range(args.e_layers)]
        )

    def forward(self, x, lens, mask, **kwargs):
        x = self.embedder(x, mask)
        x = torch.cat((self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2)
        x = self.position_enc(x)
        lens_mask = length_to_mask(lens + 1)
        mask = torch.cat((torch.ones(mask.shape[0], 1, mask.shape[-1], device=mask.device, dtype=mask.dtype), mask), dim=1)
        mask = mask.transpose(1, 2).float()
        for block in self.blocks:
            x = block(x, mask, lens, lens_mask)
        return x


class EmbeddingDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = Mlp(
            in_features=args.d_model,
            hidden_features=int(args.d_model * 4),
            act_layer=nn.GELU,
            drop=args.dropout,
        )
        self.proj_out = nn.Linear(args.d_model, args.d_model)

    def forward(self, x):
        x = self.mlp(x)
        x = self.proj_out(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = MLPBlock(dim=args.d_model, mlp_ratio=4, 
                            proj_drop=args.dropout, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.out = nn.Linear(args.d_model * args.input_dim, args.num_class)
        
    def forward(self, h, **kwargs):
        B, I, T, H = h.shape
        cls_token = h[:, :, 0]
        cls_token = cls_token.reshape(B, I, H)
        cls_token = self.mlp(cls_token)
        logits = self.out(cls_token.reshape(B, -1))
        return logits


# ============================================================
# SMART-FiLM: Temporal Feature-wise Linear Modulation
# ============================================================

class TimeEncoder(nn.Module):
    """Learnable continuous time encoding.

    Phi(t) = [sin(w_1*t), cos(w_1*t), ..., sin(w_d*t), cos(w_d*t)]
    where w is a learnable frequency vector (log-linear initialized).

    Args:
        time_dim: Output dimension (must be even).

    Input:  t of shape (..., T)  -- real timestamps (e.g. hours since admission)
    Output: encoding of shape (..., T, time_dim)
    """

    def __init__(self, time_dim):
        super().__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        half_dim = time_dim // 2
        init_freq = torch.exp(torch.linspace(0, -math.log(1000.0), half_dim))
        self.w = nn.Parameter(init_freq)

    def forward(self, t):
        # t: (..., T)
        # angles: (..., T, half_dim)
        angles = t.unsqueeze(-1) * self.w
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimeFiLMMLPBlock(nn.Module):
    """MLPBlock augmented with time-conditional FiLM modulation.

    FiLM is applied to the intermediate hidden features after fc1:
        h_film = (1 + gamma) * h + beta
    where gamma, beta are generated from the time encoding.

    FiLM generators are zero-initialized so training starts from
    the standard MLP (identity modulation).

    Args:
        dim:         Input/output feature dimension.
        time_dim:    Dimension of the time encoding.
        mlp_ratio:   Hidden dimension multiplier.
        proj_drop:   Dropout rate.
        act_layer:   Activation (default GELU).
        norm_layer:  Normalization (default LayerNorm).
    """

    def __init__(
            self,
            dim,
            time_dim,
            mlp_ratio=4.,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.drop1 = nn.Dropout(proj_drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(proj_drop)
        # FiLM generator: time_enc -> (gamma, beta) for hidden features
        self.film_gen = nn.Linear(time_dim, 2 * hidden)
        # Zero-init: start from identity (no modulation)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x, time_enc=None):
        # x: (B, V, T, d)
        # time_enc: (B, T, time_dim) or None
        residual = x
        x = self.norm2(x)
        h = self.fc1(x)  # (B, V, T, hidden)
        if time_enc is not None:
            # (B, T, 2*hidden) -> gamma/beta: (B, T, hidden)
            film = self.film_gen(time_enc)
            gamma, beta = film.chunk(2, dim=-1)
            # broadcast over variable dimension V
            h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return residual + h


class TimeFiLMVarAttBlock(nn.Module):
    """VarAttBlock augmented with time-conditional FiLM on the attention output.

    After cross-variable attention, FiLM modulates the output based on the
    current timestamp: which variable interactions matter changes over time.

    FiLM generator is zero-initialized (identity at init).
    """

    def __init__(
            self,
            dim,
            num_heads,
            time_dim,
            qkv_bias=False,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )
        # Post-attention FiLM: time_enc -> (gamma, beta) over dim
        self.film_gen = nn.Linear(time_dim, 2 * dim)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x, mask, lens, lens_mask, time_enc=None):
        # Standard VarAtt residual
        attn_out = self.attn_var(self.norm1(x), mask, lens, lens_mask)
        if time_enc is not None:
            # Modulate the attention residual before adding to x
            film = self.film_gen(time_enc)            # (B, T, 2*dim)
            gamma, beta = film.chunk(2, dim=-1)       # each (B, T, dim)
            attn_out = (1.0 + gamma.unsqueeze(1)) * attn_out + beta.unsqueeze(1)
        return x + attn_out


class TimeFiLMBasicBlock(nn.Module):
    """Transformer block with time-conditional FiLM in VarAttBlock and MLPBlock.

    SeqAttBlock is unchanged (processes temporal ordering, not timestamps).
    VarAttBlock and MLPBlock both receive time_enc for FiLM conditioning.
    """

    def __init__(
            self,
            dim,
            num_heads,
            time_dim,
            mlp_ratio=8.,
            qkv_bias=False,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.var_att_block = TimeFiLMVarAttBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.mlp = TimeFiLMMLPBlock(
            dim=dim, time_dim=time_dim, mlp_ratio=mlp_ratio,
            proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer,
        )

    def forward(self, x, mask, lens, lens_mask, time_enc=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        x = self.var_att_block(x, mask, lens, lens_mask, time_enc)
        x = self.mlp(x, time_enc)
        return x


class TimeFiLMEncoder(nn.Module):
    """SMART Encoder augmented with temporal FiLM modulation (SMART-FiLM).

    Key differences from Encoder:
    1. Accepts real timestamps (hours since admission) via the 'time' argument.
    2. Encodes timestamps with a learnable TimeEncoder.
    3. Each BasicBlock is replaced with TimeFiLMBasicBlock that applies
       FiLM conditioning to VarAttBlock and MLPBlock using the time encoding.
    4. Existing PositionalEncoding is retained (complementary to FiLM).

    If time=None (or not provided), all FiLM modules act as identity
    transformations (zero-initialized gamma/beta), matching original SMART.

    Args:
        args: Namespace with d_model, n_heads, e_layers, dropout, input_dim,
              max_len, and time_dim (default 16).
    """

    def __init__(self, args):
        super().__init__()
        self.time_dim = getattr(args, 'time_dim', 16)
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.time_encoder = TimeEncoder(self.time_dim)
        self.blocks = nn.ModuleList([
            TimeFiLMBasicBlock(
                dim=args.d_model,
                num_heads=args.n_heads,
                time_dim=self.time_dim,
                qkv_bias=False,
                mlp_ratio=4.,
                proj_drop=args.dropout,
            )
            for _ in range(args.e_layers)
        ])

    def forward(self, x, lens, mask, time=None, **kwargs):
        # x: (B, T, V),  mask: (B, T, V),  time: (B, T) float or None
        x = self.embedder(x, mask)                              # (B, V, T, d)
        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)
        x = self.position_enc(x)
        lens_mask = length_to_mask(lens + 1)
        mask = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask = mask.transpose(1, 2).float()

        # Build time encoding for all positions including cls token (t=0)
        time_enc = None
        if time is not None:
            B = x.shape[0]
            cls_time = torch.zeros(B, 1, device=time.device, dtype=time.dtype)
            time_full = torch.cat([cls_time, time], dim=1)      # (B, T+1)
            time_enc = self.time_encoder(time_full)              # (B, T+1, time_dim)

        for block in self.blocks:
            x = block(x, mask, lens, lens_mask, time_enc)
        return x


# ============================================================
# SMILE: Structure and Missingness-Aware representation LEarning
# ============================================================

class MissingPatternEncoder(nn.Module):
    """
    Encode the clinical observation pattern original_mask (B, T, V) into embeddings.

    Two parallel branches:
      Branch 1 - Per-variable 1D CNN:
        Three dilated convolutions (receptive field 1/3/7 steps) capture multi-scale
        single-variable missing patterns (short/mid/long-term gaps).
      Branch 2 - Cross-variable 2D CNN:
        A 2D convolution on the (V, T) plane captures joint missing patterns across
        physiologically related variables (e.g., heart rate + blood pressure jointly
        absent), directly encoding MNAR co-occurrence structure that the 1D branch misses.

    The two branches are fused via a learned linear gate, then projected with
    zero-init so the MNAR branch contributes 0 at initialization.
    Stateless design, fully orthogonal to the Transformer backbone.
    """
    def __init__(self, d_model):
        super().__init__()
        # Branch 1: per-variable 1D CNN (multi-scale temporal missing patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
        )
        # Branch 2: cross-variable 2D CNN on (V, T) plane
        # kernel=(3,3): simultaneously captures adjacent-variable and adjacent-time
        # co-missingness (e.g., cardiovascular cluster missing together)
        self.cross_var_conv = nn.Sequential(
            nn.Conv2d(1, d_model // 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=(1, 1)),
        )
        # Learned fusion of per-variable and cross-variable features
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Zero-init: MNAR branch contributes 0 at start, equivalent to original Encoder
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Dataset-specific mapping (for grouping related variables before 2D CNN)
        self.register_buffer("var_order_idx", None)
        self.register_buffer("inv_order_idx", None)

    def set_variable_order(self, var_order_idx, inv_order_idx):
        self.var_order_idx = var_order_idx
        self.inv_order_idx = inv_order_idx

    def forward(self, original_mask):
        # original_mask: (B, T, V)
        B, T, V = original_mask.shape

        # Branch 1: per-variable 1D CNN
        x1 = original_mask.permute(0, 2, 1).reshape(B * V, 1, T).float()
        x1 = self.cnn(x1)               # (B*V, d_model, T)
        x1 = x1.permute(0, 2, 1)        # (B*V, T, d_model)
        x1 = x1.reshape(B, V, T, -1)    # (B, V, T, d_model)

        # Branch 2: cross-variable 2D CNN on (V, T) plane
        
        # [PRE-CNN PERMUTATION] Reorder V dimension so related systems are adjacent
        v_idx_fwd = self.var_order_idx if self.var_order_idx is not None else torch.arange(V, device=original_mask.device)
        v_idx_inv = self.inv_order_idx if self.inv_order_idx is not None else torch.arange(V, device=original_mask.device)

        x2_permuted = original_mask.permute(0, 2, 1)[:, v_idx_fwd, :].unsqueeze(1).float()  # (B, 1, V, T)
        x2_conv = self.cross_var_conv(x2_permuted)    # (B, d_model, V, T)
        
        # [POST-CNN PERMUTATION] Restore to original variable order
        x2 = x2_conv[:, :, v_idx_inv, :]             # (B, d_model, V, T)
        x2 = x2.permute(0, 2, 3, 1)    # (B, V, T, d_model)

        # Gated fusion
        x = self.fusion(torch.cat([x1, x2], dim=-1))  # (B, V, T, d_model)
        return self.proj(x)


class MNAREncoder(Encoder):
    """
    SMART + MNAR-Aware Encoding (simplified from SMILEEncoder).

    Keeps the core MNAR value (per-block gated injection + CLS injection)
    but with better initialization:
    - mnar_gates initialized to 0.01 (not 0.0) so gradient signal flows from epoch 1
    - mnar_cls_proj initialized with small random weights (not zeros)

    Designed to be used with simple random masking (no curriculum).
    When original_mask=None, degrades completely to standard Encoder.
    """
    def __init__(self, args):
        super().__init__(args)
        self.mnar_encoder = MissingPatternEncoder(args.d_model)
        if hasattr(args, 'var_order_idx') and hasattr(args, 'inv_order_idx'):
            self.mnar_encoder.set_variable_order(args.var_order_idx, args.inv_order_idx)
        # Per-block learnable scalar gates -- small positive init (not zero)
        self.mnar_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.01)) for _ in range(args.e_layers)]
        )
        # Small random init projection for global MNAR embedding -> CLS token
        self.mnar_cls_proj = nn.Linear(args.d_model, args.d_model)
        nn.init.normal_(self.mnar_cls_proj.weight, std=0.01)
        nn.init.zeros_(self.mnar_cls_proj.bias)
        # Attention pooling over T for global MNAR summary (zero-init = uniform = mean initially)
        self.mnar_time_pool = nn.Linear(args.d_model, 1)
        nn.init.zeros_(self.mnar_time_pool.weight)
        nn.init.zeros_(self.mnar_time_pool.bias)

    def forward(self, x, lens, mask, original_mask=None, **kwargs):
        x = self.embedder(x, mask)                              # (B, V, T, d)

        mnar_emb = None
        mnar_emb_padded = None
        if original_mask is not None:
            mnar_emb = self.mnar_encoder(original_mask)         # (B, V, T, d)
            cls_pad = torch.zeros(
                mnar_emb.shape[0], mnar_emb.shape[1], 1, mnar_emb.shape[3],
                device=mnar_emb.device, dtype=mnar_emb.dtype
            )
            mnar_emb_padded = torch.cat([cls_pad, mnar_emb], dim=2)  # (B, V, T+1, d)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)

        x = self.position_enc(x)
        # Inject MNAR after PE: positional geometry is intact, 1D CNN already has local equivariance
        if mnar_emb_padded is not None:
            x = x + mnar_emb_padded

        lens_mask = length_to_mask(lens + 1)
        mask = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask = mask.transpose(1, 2).float()

        for l, block in enumerate(self.blocks):
            x = block(x, mask, lens, lens_mask)
            if mnar_emb_padded is not None:
                x = x + self.mnar_gates[l] * mnar_emb_padded   # gated per-block injection

        # Inject global MNAR summary into CLS token (attention-pooled over T)
        if mnar_emb is not None:
            attn_scores = self.mnar_time_pool(mnar_emb)         # (B, V, T, 1)
            attn_w = torch.softmax(attn_scores, dim=2)          # softmax over T
            global_mnar = self.mnar_cls_proj(
                (attn_w * mnar_emb).sum(dim=2, keepdim=True)   # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_mnar, x[:, :, 1:]], dim=2)

        return x


class SMILEEncoder(Encoder):
    """
    SMART + MNAR-Aware Encoding (v2).

    Improvements over v1:
    1. MissingPatternEncoder now combines per-variable 1D CNN and cross-variable 2D CNN,
       capturing joint MNAR co-occurrence across physiologically related variables.
    2. MNAR embedding injected at every Transformer block via learnable channel-wise
       gate MLP (replaces scalar gate), giving per-channel control over injection strength.
    3. Global MNAR vector (mean-pooled over T) injected into the CLS token after all
       blocks, so missingness directly influences downstream classification.

    All new parameters are near-zero initialized: at init the model ~= standard Encoder.
    When original_mask=None, degrades completely to standard Encoder.
    """
    def __init__(self, args):
        super().__init__(args)
        self.mnar_encoder = MissingPatternEncoder(args.d_model)
        if hasattr(args, 'var_order_idx') and hasattr(args, 'inv_order_idx'):
            self.mnar_encoder.set_variable_order(args.var_order_idx, args.inv_order_idx)
        # Per-block channel-wise gate MLPs (replaces scalar gates for finer per-channel control)
        self.mnar_gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.d_model, args.d_model // 4),
                nn.GELU(),
                nn.Linear(args.d_model // 4, args.d_model),
                nn.Sigmoid(),
            )
            for _ in range(args.e_layers)
        ])
        # Init: sigmoid(-5) ~= 0.007, near-zero but non-zero to allow gradient flow from epoch 1
        for mlp in self.mnar_gate_mlps:
            nn.init.zeros_(mlp[0].weight)
            nn.init.constant_(mlp[0].bias, -5.0)
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)
        # Small random init for CLS projection (faster early learning vs zero-init)
        self.mnar_cls_proj = nn.Linear(args.d_model, args.d_model)
        nn.init.normal_(self.mnar_cls_proj.weight, std=0.01)
        nn.init.zeros_(self.mnar_cls_proj.bias)
        # Attention pooling over T for global MNAR summary (zero-init = uniform = mean initially)
        self.mnar_time_pool = nn.Linear(args.d_model, 1)
        nn.init.zeros_(self.mnar_time_pool.weight)
        nn.init.zeros_(self.mnar_time_pool.bias)

    def forward(self, x, lens, mask, original_mask=None, **kwargs):
        x = self.embedder(x, mask)                              # (B, V, T, d)

        mnar_emb = None
        mnar_emb_padded = None
        if original_mask is not None:
            mnar_emb = self.mnar_encoder(original_mask)         # (B, V, T, d)
            cls_pad = torch.zeros(
                mnar_emb.shape[0], mnar_emb.shape[1], 1, mnar_emb.shape[3],
                device=mnar_emb.device, dtype=mnar_emb.dtype
            )
            mnar_emb_padded = torch.cat([cls_pad, mnar_emb], dim=2)  # (B, V, T+1, d)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)

        x = self.position_enc(x)
        # Inject MNAR after PE: positional geometry is intact, 1D CNN already has local equivariance
        if mnar_emb_padded is not None:
            x = x + mnar_emb_padded

        lens_mask = length_to_mask(lens + 1)
        mask = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask = mask.transpose(1, 2).float()

        for l, block in enumerate(self.blocks):
            x = block(x, mask, lens, lens_mask)
            if mnar_emb_padded is not None:
                gate = self.mnar_gate_mlps[l](mnar_emb_padded)  # (B, V, T+1, d_model)
                x = x + gate * mnar_emb_padded                  # channel-wise gated injection

        # Inject global MNAR summary into CLS token (attention-pooled over T)
        if mnar_emb is not None:
            attn_scores = self.mnar_time_pool(mnar_emb)         # (B, V, T, 1)
            attn_w = torch.softmax(attn_scores, dim=2)          # softmax over T
            global_mnar = self.mnar_cls_proj(
                (attn_w * mnar_emb).sum(dim=2, keepdim=True)   # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_mnar, x[:, :, 1:]], dim=2)

        return x


# ============================================================
# SMILE-FiLM: Joint Temporal FiLM + MNAR Channel Gating
# ============================================================

class SMILEFiLMBasicBlock(nn.Module):
    """Transformer block combining time-conditional FiLM and per-channel MNAR gating.

    Dual modulation per block:
      - FiLM (gamma/beta from TimeEncoder): controls representation amplitude by timestamp.
      - MNAR channel gate (sigmoid MLP from mnar_emb): controls channel on/off by missingness.

    All MNAR gate parameters are near-zero initialized (sigmoid(-5) ~= 0.007).
    All FiLM generator parameters are zero-initialized (identity modulation at init).
    """

    def __init__(
            self,
            dim,
            num_heads,
            time_dim,
            mlp_ratio=8.,
            qkv_bias=False,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.var_att_block = TimeFiLMVarAttBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.mlp = TimeFiLMMLPBlock(
            dim=dim, time_dim=time_dim, mlp_ratio=mlp_ratio,
            proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer,
        )
        # Per-channel MNAR gate: near-zero init via sigmoid(-5) ~= 0.007
        self.mnar_gate_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.mnar_gate_mlp[0].weight)
        nn.init.constant_(self.mnar_gate_mlp[0].bias, -5.0)
        nn.init.zeros_(self.mnar_gate_mlp[2].weight)
        nn.init.zeros_(self.mnar_gate_mlp[2].bias)

    def forward(self, x, mask, lens, lens_mask, time_enc=None, mnar_emb=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        x = self.var_att_block(x, mask, lens, lens_mask, time_enc)
        x = self.mlp(x, time_enc)
        if mnar_emb is not None:
            gate = self.mnar_gate_mlp(mnar_emb)
            x = x + gate * mnar_emb
        return x


class SMILEFiLMEncoder(nn.Module):
    """SMART encoder combining SMILE (MNAR-aware) and FiLM (time-conditional) modulation.

    Integrates:
    1. MissingPatternEncoder: dual-branch 1D+2D CNN captures per-variable and
       cross-variable missingness patterns (MNAR co-occurrence).
    2. TimeEncoder: learnable frequencies -> continuous time encoding for FiLM.
    3. SMILEFiLMBasicBlock: each block applies both FiLM and MNAR channel gating.
    4. MNAR injected after PositionalEncoding (preserves PE absolute position geometry).
    5. Attention-pooled global MNAR summary -> CLS token (learns to focus on critical
       time windows, e.g., near decompensation onset).

    Falls back to standard Encoder when original_mask=None and time=None.
    """

    def __init__(self, args):
        super().__init__()
        self.time_dim = getattr(args, 'time_dim', 16)
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.time_encoder = TimeEncoder(self.time_dim)
        self.mnar_encoder = MissingPatternEncoder(args.d_model)
        if hasattr(args, 'var_order_idx') and hasattr(args, 'inv_order_idx'):
            self.mnar_encoder.set_variable_order(args.var_order_idx, args.inv_order_idx)
        self.blocks = nn.ModuleList([
            SMILEFiLMBasicBlock(
                dim=args.d_model,
                num_heads=args.n_heads,
                time_dim=self.time_dim,
                mlp_ratio=4.,
                qkv_bias=False,
                proj_drop=args.dropout,
            )
            for _ in range(args.e_layers)
        ])
        # Attention pooling: zero-init = uniform weights = mean pooling initially
        self.mnar_time_pool = nn.Linear(args.d_model, 1)
        nn.init.zeros_(self.mnar_time_pool.weight)
        nn.init.zeros_(self.mnar_time_pool.bias)
        self.mnar_cls_proj = nn.Linear(args.d_model, args.d_model)
        nn.init.normal_(self.mnar_cls_proj.weight, std=0.01)
        nn.init.zeros_(self.mnar_cls_proj.bias)

    def forward(self, x, lens, mask, time=None, original_mask=None, **kwargs):
        x = self.embedder(x, mask)                              # (B, V, T, d)

        mnar_emb = None
        mnar_emb_padded = None
        if original_mask is not None:
            mnar_emb = self.mnar_encoder(original_mask)         # (B, V, T, d)
            cls_pad = torch.zeros(
                mnar_emb.shape[0], mnar_emb.shape[1], 1, mnar_emb.shape[3],
                device=mnar_emb.device, dtype=mnar_emb.dtype
            )
            mnar_emb_padded = torch.cat([cls_pad, mnar_emb], dim=2)  # (B, V, T+1, d)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)

        x = self.position_enc(x)
        # Inject MNAR after PE so positional geometry is intact
        if mnar_emb_padded is not None:
            x = x + mnar_emb_padded

        lens_mask = length_to_mask(lens + 1)
        mask_full = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask_full = mask_full.transpose(1, 2).float()

        # Build time encoding including CLS position (t=0)
        time_enc = None
        if time is not None:
            cls_time = torch.zeros(x.shape[0], 1, device=time.device, dtype=time.dtype)
            time_full = torch.cat([cls_time, time], dim=1)      # (B, T+1)
            time_enc = self.time_encoder(time_full)              # (B, T+1, time_dim)

        for block in self.blocks:
            x = block(x, mask_full, lens, lens_mask, time_enc, mnar_emb_padded)

        # Attention-pooled global MNAR -> CLS token injection
        if mnar_emb is not None:
            attn_scores = self.mnar_time_pool(mnar_emb)         # (B, V, T, 1)
            attn_w = torch.softmax(attn_scores, dim=2)          # softmax over T
            global_mnar = self.mnar_cls_proj(
                (attn_w * mnar_emb).sum(dim=2, keepdim=True)   # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_mnar, x[:, :, 1:]], dim=2)

        return x


# ============================================================
# SMILE v2: MNAR-Guided Attention Bias + Obs Density + Cross-Attention MNAR Fusion
# ============================================================

class MNARCooccurrenceEncoder(nn.Module):
    """Compute MNAR co-occurrence matrix from observation mask (no parameters).

    co_occur[b, i, j] = proportion of time steps where both variable i and j are
    missing simultaneously.  This captures physiological co-missingness structure
    (e.g., cardiovascular variables dropping out together) without any learned
    parameters -- the bias scaling is handled by MNARBiasVarAttention.

    Input:  original_mask  (B, T, V)  1=observed, 0=missing
    Output: co_occur       (B, V, V)  symmetric, values in [0, 1]
    """

    def forward(self, original_mask):
        missing = 1.0 - original_mask.float()           # (B, T, V)  1=missing
        T = missing.shape[1]
        # co_occur[b, i, j] = E_t[missing_i(t) * missing_j(t)]
        co_occur = torch.bmm(missing.permute(0, 2, 1), missing) / (T + 1e-6)  # (B, V, V)
        # Subtract independent expectation: proper correlation, not joint probability.
        # Avoids inflating co-occurrence for variables with high marginal missing rates.
        marginal = missing.mean(dim=1)                   # (B, V)
        expected = marginal.unsqueeze(2) * marginal.unsqueeze(1)  # (B, V, V)
        return co_occur - expected                        # signed, in ~[-0.25, 0.75]


class MNARBiasVarAttention(nn.Module):
    """VarAttention augmented with MNAR co-occurrence attention bias.

    Adds a learnable per-head scalar scaling of the (B, V, V) MNAR co-occurrence
    matrix to the attention logits before softmax.  Variables that co-miss share
    physiological state and should attend to each other more strongly.

    mnar_bias_scale is zero-initialized: at init this is identical to VarAttention.

    Args:
        dim:       Feature dimension.
        num_heads: Number of attention heads.
        qkv_bias:  Whether to use bias in QKV projection.
        proj_drop: Dropout on output projection.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Per-head MNAR co-occurrence bias scale (zero-init = no bias at init)
        self.mnar_bias_scale = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, time_decay=None):
        B, N, P, C = x.shape
        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                   self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)

        q = q[:, 0]  # (B, heads, N, head_dim) -- CLS-position query per variable
        mask_r = mask.reshape(B, N, P, 1, 1).repeat(1, 1, 1, self.num_heads,
                              self.head_dim).permute(0, 2, 3, 1, 4)
        k = k.masked_fill(~mask_r.bool(), 0).sum(dim=1) / (mask_r.sum(dim=1) + 1e-6)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        # Build additive attention bias from MNAR co-occurrence
        attn_mask = None
        if mnar_cooccur is not None:
            scale = self.mnar_bias_scale.view(1, -1, 1, 1)
            if time_decay is not None:
                # time_decay: (B, num_heads) -- per-sample per-head temporal scaling
                scale = scale * time_decay.view(B, -1, 1, 1)
            # (B, V, V) -> (B, heads, V, V) -- added to raw attention logits
            attn_mask = mnar_cooccur.unsqueeze(1) * scale

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.view(B, self.num_heads, N, -1, P).permute(0, 2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MNARBiasVarAttBlock(nn.Module):
    """VarAttBlock using MNARBiasVarAttention."""

    def __init__(self, dim, num_heads, qkv_bias=False, proj_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = MNARBiasVarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop
        )

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None):
        x = x + self.attn_var(self.norm1(x), mask, lens, lens_mask, mnar_cooccur)
        return x


class ObsDensityEmbedder(nn.Module):
    """Local observation density embedding.

    Computes the proportion of observed time steps within a sliding window around
    each position for each variable.  This 'measurement frequency near time t'
    signal is a clinical acuity proxy: more frequent measurements indicate
    increased clinical concern.

    Implemented as F.avg_pool1d over the binary observation mask.  The two-layer
    MLP output is zero-initialized so the contribution is zero at init (backward
    compatible with standard SMART).

    Args:
        d_model:     Output embedding dimension.
        window_size: Odd integer; local window for density estimation (default 5).

    Input:  original_mask  (B, T, V)
    Output: embedding      (B, V, T, d_model)
    """

    def __init__(self, d_model, window_size=5):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, original_mask):
        B, T, V = original_mask.shape
        m = original_mask.float().permute(0, 2, 1).reshape(B * V, 1, T)  # (B*V, 1, T)
        density = F.avg_pool1d(
            m, kernel_size=self.window_size, stride=1, padding=self.window_size // 2
        )                                                 # (B*V, 1, T)
        density = density.reshape(B, V, T, 1)             # (B, V, T, 1)
        return self.proj(density)                         # (B, V, T, d_model)


class MNARCrossAttention(nn.Module):
    """Cross-attention MNAR fusion: main representation selectively queries MNAR embeddings.

    Replaces per-block gated additive injection (x += gate * mnar_emb) with
    cross-attention where x acts as Query and mnar_emb as Key/Value:

        out = CrossAttn(Q=x, K=mnar_emb, V=mnar_emb)
        x   = x + out_proj(out)

    Each position can attend to DIFFERENT aspects of the missing pattern rather
    than receiving the same gated MNAR vector.  The output projection is
    zero-initialized so there is no contribution at initialization.

    Applied independently per variable (T+1 including CLS is the sequence dimension).

    Args:
        dim:       Feature dimension.
        num_heads: Number of attention heads.
        proj_drop: Dropout rate.
    """

    def __init__(self, dim, num_heads, proj_drop=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=proj_drop, batch_first=True
        )
        self.out_proj = nn.Linear(dim, dim)
        # Small random init (std=0.01): near-zero contribution at init but gradients
        # flow from epoch 1, unlike zero-init which blocks gradients entirely.
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, mnar_emb):
        # x: (B, V, T, d),  mnar_emb: (B, V, T, d)
        B, V, T, d = x.shape
        xf = self.norm_q(x).reshape(B * V, T, d)
        mf = self.norm_kv(mnar_emb).reshape(B * V, T, d)
        out, _ = self.cross_attn(xf, mf, mf, need_weights=False)
        out = out.reshape(B, V, T, d)
        return self.out_proj(out)


class SMILEv2BasicBlock(nn.Module):
    """Transformer block with all three SMILE v2 improvements.

    1. SeqAttBlock:         temporal attention (unchanged from SMART).
    2. MNARBiasVarAttBlock: cross-variable attention + MNAR co-occurrence bias.
    3. MLPBlock:            feed-forward (unchanged from SMART).
    4. MNARCrossAttention:  selective MNAR fusion via cross-attention.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 abl_no_mnar_bias=False, abl_no_cross_attn=False):
        super().__init__()
        self.abl_no_mnar_bias = abl_no_mnar_bias
        self.abl_no_cross_attn = abl_no_cross_attn
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        if abl_no_mnar_bias:
            self.var_att_block = VarAttBlock(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                proj_drop=proj_drop, norm_layer=norm_layer,
            )
        else:
            self.var_att_block = MNARBiasVarAttBlock(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                proj_drop=proj_drop, norm_layer=norm_layer,
            )
        self.mlp = MLPBlock(
            dim=dim, mlp_ratio=mlp_ratio, proj_drop=proj_drop,
            act_layer=act_layer, norm_layer=norm_layer,
        )
        if not abl_no_cross_attn:
            self.mnar_cross_attn = MNARCrossAttention(dim, num_heads, proj_drop=proj_drop)

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, mnar_emb=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        if self.abl_no_mnar_bias:
            x = self.var_att_block(x, mask, lens, lens_mask)
        else:
            x = self.var_att_block(x, mask, lens, lens_mask, mnar_cooccur)
        x = self.mlp(x)
        if (not self.abl_no_cross_attn) and mnar_emb is not None:
            x = x + self.mnar_cross_attn(x, mnar_emb)
        return x


class SMILEv2Encoder(nn.Module):
    """SMART encoder with three architectural improvements over SMILEEncoder.

    Improvement 1 -- MNAR-guided attention bias:
        VarAttention is augmented with a per-head learnable scaling of the
        (B, V, V) MNAR co-occurrence matrix added to cross-variable attention logits.
        Variables that co-miss share physiological state and are biased to attend
        to each other.  Zero-initialized -> identical to VarAttention at init.

    Improvement 2 -- Observation density embedding:
        A local measurement-frequency signal (proportion of observed steps in a
        sliding window) is computed per variable and projected to d_model via a
        zero-init MLP, then added to the value embedding.  Captures clinical acuity.

    Improvement 3 -- Cross-attention MNAR fusion:
        Per-block MNAR injection uses cross-attention (x queries mnar_emb) instead
        of gated additive fusion.  Each position selectively retrieves the MNAR
        context it needs.  Output projection is zero-init.

    Degrades to standard Encoder when original_mask=None.
    """

    def __init__(self, args):
        super().__init__()
        self.abl_no_density = getattr(args, 'abl_no_density', False)
        self.abl_no_mnar_bias = getattr(args, 'abl_no_mnar_bias', False)
        self.abl_no_cross_attn = getattr(args, 'abl_no_cross_attn', False)
        self.abl_no_mnar_cls = getattr(args, 'abl_no_mnar_cls', False)
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)

        self.mnar_encoder = MissingPatternEncoder(args.d_model)
        if hasattr(args, 'var_order_idx') and hasattr(args, 'inv_order_idx'):
            self.mnar_encoder.set_variable_order(args.var_order_idx, args.inv_order_idx)

        if not self.abl_no_mnar_bias:
            self.mnar_cooccur_encoder = MNARCooccurrenceEncoder()
        if not self.abl_no_density:
            self.obs_density_embedder = ObsDensityEmbedder(
                args.d_model, window_size=getattr(args, 'obs_density_window', 5)
            )

        self.blocks = nn.ModuleList([
            SMILEv2BasicBlock(
                dim=args.d_model, num_heads=args.n_heads, mlp_ratio=4.,
                qkv_bias=False, proj_drop=args.dropout,
                abl_no_mnar_bias=self.abl_no_mnar_bias,
                abl_no_cross_attn=self.abl_no_cross_attn,
            )
            for _ in range(args.e_layers)
        ])

        if not self.abl_no_mnar_cls:
            self.mnar_time_pool = nn.Linear(args.d_model, 1)
            nn.init.zeros_(self.mnar_time_pool.weight)
            nn.init.zeros_(self.mnar_time_pool.bias)
            self.mnar_cls_proj = nn.Linear(args.d_model, args.d_model)
            nn.init.normal_(self.mnar_cls_proj.weight, std=0.01)
            nn.init.zeros_(self.mnar_cls_proj.bias)
        if not self.abl_no_density:
            # Project mean observation density to d_model and inject into CLS token.
            # Zero-init: no contribution at init; learned as training progresses.
            self.cls_density_proj = nn.Linear(args.d_model, args.d_model)
            nn.init.zeros_(self.cls_density_proj.weight)
            nn.init.zeros_(self.cls_density_proj.bias)

    def forward(self, x, lens, mask, original_mask=None, **kwargs):
        x = self.embedder(x, mask)                              # (B, V, T, d)

        mnar_emb = None
        mnar_emb_padded = None
        mnar_cooccur = None
        obs_density_emb = None
        if original_mask is not None:
            # Encode missing patterns via dual-branch 1D+2D CNN
            mnar_emb = self.mnar_encoder(original_mask)         # (B, V, T, d)
            # MNAR co-occurrence matrix for attention bias
            if not self.abl_no_mnar_bias:
                mnar_cooccur = self.mnar_cooccur_encoder(original_mask)  # (B, V, V)
            # Observation density embedding (zero-init -> no-op at init)
            if not self.abl_no_density:
                obs_density_emb = self.obs_density_embedder(original_mask)  # (B, V, T, d)
                x = x + obs_density_emb
            cls_pad = torch.zeros(
                mnar_emb.shape[0], mnar_emb.shape[1], 1, mnar_emb.shape[3],
                device=mnar_emb.device, dtype=mnar_emb.dtype
            )
            mnar_emb_padded = torch.cat([cls_pad, mnar_emb], dim=2)  # (B, V, T+1, d)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)
        x = self.position_enc(x)

        # Inject global observation density into CLS token (zero-init -> no-op at init).
        # obs_density_emb covers T positions; mean pools to a per-variable scalar that
        # summarises overall measurement intensity, then projects to d_model.
        if obs_density_emb is not None:
            global_density = self.cls_density_proj(
                obs_density_emb.mean(dim=2, keepdim=True)       # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_density, x[:, :, 1:]], dim=2)

        # Initial MNAR injection after PE (positional geometry intact)
        if mnar_emb_padded is not None:
            x = x + mnar_emb_padded

        lens_mask = length_to_mask(lens + 1)
        mask_full = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask_full = mask_full.transpose(1, 2).float()

        for block in self.blocks:
            x = block(x, mask_full, lens, lens_mask, mnar_cooccur, mnar_emb_padded)

        # Attention-pooled global MNAR -> CLS token injection
        if (not self.abl_no_mnar_cls) and mnar_emb is not None:
            attn_scores = self.mnar_time_pool(mnar_emb)         # (B, V, T, 1)
            attn_w = torch.softmax(attn_scores, dim=2)
            global_mnar = self.mnar_cls_proj(
                (attn_w * mnar_emb).sum(dim=2, keepdim=True)   # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_mnar, x[:, :, 1:]], dim=2)

        return x


# ============================================================
# SMILEv2-FiLM: SMILEv2 + Time-Conditional FiLM Modulation
# ============================================================

class MNARBiasFiLMVarAttBlock(nn.Module):
    """VarAttBlock with MNAR co-occurrence attention bias AND time-conditional FiLM.

    Combines MNARBiasVarAttention (learnable per-head co-occurrence bias on
    cross-variable attention logits) with post-attention FiLM modulation
    (time-conditional scaling of the attention residual before it is added to x).

    FiLM generator is zero-initialized (identity modulation at init).

    Args:
        use_time_mnar: If True, adds a per-head time-decay gate on the MNAR
            co-occurrence bias.  Projects mean(time_enc) -> num_heads via a
            sigmoid to produce a (B, heads) scalar that multiplies mnar_bias_scale.
            Zero-initialized so there is no change at init.  This gives the
            MNAR bias temporal dynamics: early vs. late co-missingness has
            different physiological meaning.
        use_film: If True, applies the post-attention time-conditioned affine
            transform. If False, time_enc can still feed the MNAR time gate.
    """

    def __init__(self, dim, num_heads, time_dim, qkv_bias=False, proj_drop=0.,
                 norm_layer=nn.LayerNorm, use_time_mnar=False, use_film=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.use_time_mnar = use_time_mnar
        self.use_film = use_film
        self.attn_var = MNARBiasVarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop
        )
        if use_film:
            # Post-attention FiLM: time_enc -> (gamma, beta) over dim; zero-init = identity
            self.film_gen = nn.Linear(time_dim, 2 * dim)
            nn.init.zeros_(self.film_gen.weight)
            nn.init.zeros_(self.film_gen.bias)
        # Optional time-dynamic MNAR bias scaling
        if use_time_mnar:
            self.mnar_time_proj = nn.Linear(time_dim, num_heads)
            nn.init.zeros_(self.mnar_time_proj.weight)
            nn.init.zeros_(self.mnar_time_proj.bias)

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, time_enc=None):
        time_decay = None
        if self.use_time_mnar and time_enc is not None and mnar_cooccur is not None:
            # mean over T+1 positions to get a per-sample time summary
            mean_time = time_enc.mean(dim=1)                    # (B, time_dim)
            time_decay = torch.sigmoid(self.mnar_time_proj(mean_time))  # (B, num_heads)
        attn_out = self.attn_var(self.norm1(x), mask, lens, lens_mask, mnar_cooccur,
                                 time_decay)
        if self.use_film and time_enc is not None:
            film = self.film_gen(time_enc)              # (B, T+1, 2*dim)
            gamma, beta = film.chunk(2, dim=-1)         # each (B, T+1, dim)
            attn_out = (1.0 + gamma.unsqueeze(1)) * attn_out + beta.unsqueeze(1)
        return x + attn_out


class SMILEv2FiLMBasicBlock(nn.Module):
    """Transformer block combining all SMILEv2 improvements with FiLM time-conditioning.

    1. SeqAttBlock:               temporal attention (unchanged).
    2. MNARBiasFiLMVarAttBlock:   cross-variable attention + MNAR co-occurrence bias + FiLM.
    3. TimeFiLMMLPBlock:          feed-forward + time-conditional FiLM.
    4. MNARCrossAttention:        selective MNAR fusion via cross-attention.
    """

    def __init__(self, dim, num_heads, time_dim, mlp_ratio=4., qkv_bias=False,
                 proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 abl_no_mnar_bias=False, abl_no_cross_attn=False):
        super().__init__()
        self.abl_no_mnar_bias = abl_no_mnar_bias
        self.abl_no_cross_attn = abl_no_cross_attn
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        if abl_no_mnar_bias:
            self.var_att_block = TimeFiLMVarAttBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
            )
        else:
            self.var_att_block = MNARBiasFiLMVarAttBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
            )
        self.mlp = TimeFiLMMLPBlock(
            dim=dim, time_dim=time_dim, mlp_ratio=mlp_ratio,
            proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer,
        )
        if not abl_no_cross_attn:
            self.mnar_cross_attn = MNARCrossAttention(dim, num_heads, proj_drop=proj_drop)

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, mnar_emb=None,
                time_enc=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        if self.abl_no_mnar_bias:
            x = self.var_att_block(x, mask, lens, lens_mask, time_enc)
        else:
            x = self.var_att_block(x, mask, lens, lens_mask, mnar_cooccur, time_enc)
        x = self.mlp(x, time_enc)
        if (not self.abl_no_cross_attn) and mnar_emb is not None:
            x = x + self.mnar_cross_attn(x, mnar_emb)
        return x


class SMILELeanBasicBlock(nn.Module):
    """Lean transformer block: SeqAtt + MNARBiasFiLMVarAtt + standard MLP.

    Keeps the two highest-value improvements from SMILEv2-FiLM:
      - MNAR co-occurrence attention bias (MNARBiasFiLMVarAttBlock)
      - Time-conditional FiLM on VarAtt (MNARBiasFiLMVarAttBlock)
    Adds vs. original SMILELean:
      - Time-dynamic MNAR bias scaling (use_time_mnar=True)
    Removes:
      - TimeFiLMMLPBlock -> standard MLPBlock (FFN needs no time conditioning)
      - MNARCrossAttention (per-block MNAR cross-attn removed)

    Ablation switches:
      - abl_no_film: disable FiLM affine on VarAtt only
      - abl_no_mnar_bias: disable MNAR co-occurrence attention bias
      - abl_no_time_mnar: disable time-dynamic MNAR scaling only
    """

    def __init__(self, dim, num_heads, time_dim, mlp_ratio=4., qkv_bias=False,
                 proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 abl_no_film=False, abl_no_mnar_bias=False, abl_no_time_mnar=False):
        super().__init__()
        self.abl_no_film = abl_no_film
        self.abl_no_mnar_bias = abl_no_mnar_bias
        self.use_time_mnar = not abl_no_time_mnar and not abl_no_mnar_bias
        self.use_film = not abl_no_film
        self.needs_time_enc = self.use_time_mnar or self.use_film
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.var_att_block = MNARBiasFiLMVarAttBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
            use_time_mnar=self.use_time_mnar,
            use_film=self.use_film,
        )
        self.mlp = MLPBlock(
            dim=dim, mlp_ratio=mlp_ratio,
            proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer,
        )

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, time_enc=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        _mnar = None if self.abl_no_mnar_bias else mnar_cooccur
        _time = time_enc if self.needs_time_enc else None
        x = self.var_att_block(x, mask, lens, lens_mask, _mnar, _time)
        x = self.mlp(x)
        return x


class SMILEv2FiLMEncoder(nn.Module):
    """SMILEv2Encoder augmented with time-conditional FiLM modulation.

    Combines all four SMILEv2 architectural improvements with TimeFiLM:

    1. MNARCooccurrenceEncoder: correlation-form (B,V,V) co-occurrence bias
       (joint missing rate minus independent expectation).
    2. ObsDensityEmbedder: local measurement-frequency signal added to value
       embeddings; global mean also injected into CLS token.
    3. MissingPatternEncoder: dual-branch 1D+2D CNN for MNAR pattern encoding.
    4. SMILEv2FiLMBasicBlock: each block applies MNAR attention bias, FiLM
       time-conditioning on VarAtt and MLP, and cross-attention MNAR fusion.
    5. Attention-pooled global MNAR -> CLS token (same as SMILEv2).

    Degrades to TimeFiLMEncoder when original_mask=None.
    Degrades to SMILEv2Encoder when time=None.
    """

    def __init__(self, args):
        super().__init__()
        self.time_dim = getattr(args, 'time_dim', 16)
        self.abl_no_density = getattr(args, 'abl_no_density', False)
        self.abl_no_mnar_bias = getattr(args, 'abl_no_mnar_bias', False)
        self.abl_no_cross_attn = getattr(args, 'abl_no_cross_attn', False)
        self.abl_no_mnar_cls = getattr(args, 'abl_no_mnar_cls', False)
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.time_encoder = TimeEncoder(self.time_dim)
        self.mnar_encoder = MissingPatternEncoder(args.d_model)
        if hasattr(args, 'var_order_idx') and hasattr(args, 'inv_order_idx'):
            self.mnar_encoder.set_variable_order(args.var_order_idx, args.inv_order_idx)
        if not self.abl_no_mnar_bias:
            self.mnar_cooccur_encoder = MNARCooccurrenceEncoder()
        if not self.abl_no_density:
            self.obs_density_embedder = ObsDensityEmbedder(
                args.d_model, window_size=getattr(args, 'obs_density_window', 5)
            )
        self.blocks = nn.ModuleList([
            SMILEv2FiLMBasicBlock(
                dim=args.d_model, num_heads=args.n_heads, time_dim=self.time_dim,
                mlp_ratio=4., qkv_bias=False, proj_drop=args.dropout,
                abl_no_mnar_bias=self.abl_no_mnar_bias,
                abl_no_cross_attn=self.abl_no_cross_attn,
            )
            for _ in range(args.e_layers)
        ])
        if not self.abl_no_mnar_cls:
            self.mnar_time_pool = nn.Linear(args.d_model, 1)
            nn.init.zeros_(self.mnar_time_pool.weight)
            nn.init.zeros_(self.mnar_time_pool.bias)
            self.mnar_cls_proj = nn.Linear(args.d_model, args.d_model)
            nn.init.normal_(self.mnar_cls_proj.weight, std=0.01)
            nn.init.zeros_(self.mnar_cls_proj.bias)
        if not self.abl_no_density:
            self.cls_density_proj = nn.Linear(args.d_model, args.d_model)
            nn.init.zeros_(self.cls_density_proj.weight)
            nn.init.zeros_(self.cls_density_proj.bias)

    def forward(self, x, lens, mask, time=None, original_mask=None, **kwargs):
        x = self.embedder(x, mask)                              # (B, V, T, d)

        mnar_emb = None
        mnar_emb_padded = None
        mnar_cooccur = None
        obs_density_emb = None
        if original_mask is not None:
            mnar_emb = self.mnar_encoder(original_mask)         # (B, V, T, d)
            if not self.abl_no_mnar_bias:
                mnar_cooccur = self.mnar_cooccur_encoder(original_mask)  # (B, V, V)
            if not self.abl_no_density:
                obs_density_emb = self.obs_density_embedder(original_mask)  # (B, V, T, d)
                x = x + obs_density_emb
            cls_pad = torch.zeros(
                mnar_emb.shape[0], mnar_emb.shape[1], 1, mnar_emb.shape[3],
                device=mnar_emb.device, dtype=mnar_emb.dtype
            )
            mnar_emb_padded = torch.cat([cls_pad, mnar_emb], dim=2)  # (B, V, T+1, d)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)
        x = self.position_enc(x)

        # Inject global observation density into CLS token (zero-init -> no-op at init)
        if obs_density_emb is not None:
            global_density = self.cls_density_proj(
                obs_density_emb.mean(dim=2, keepdim=True)       # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_density, x[:, :, 1:]], dim=2)

        # Initial MNAR injection after PE (positional geometry intact)
        if mnar_emb_padded is not None:
            x = x + mnar_emb_padded

        lens_mask = length_to_mask(lens + 1)
        mask_full = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask_full = mask_full.transpose(1, 2).float()

        # Build time encoding including CLS position (t=0)
        time_enc = None
        if time is not None:
            cls_time = torch.zeros(x.shape[0], 1, device=time.device, dtype=time.dtype)
            time_full = torch.cat([cls_time, time], dim=1)      # (B, T+1)
            time_enc = self.time_encoder(time_full)              # (B, T+1, time_dim)

        for block in self.blocks:
            x = block(x, mask_full, lens, lens_mask, mnar_cooccur, mnar_emb_padded,
                      time_enc)

        # Attention-pooled global MNAR -> CLS token injection
        if (not self.abl_no_mnar_cls) and mnar_emb is not None:
            attn_scores = self.mnar_time_pool(mnar_emb)         # (B, V, T, 1)
            attn_w = torch.softmax(attn_scores, dim=2)
            global_mnar = self.mnar_cls_proj(
                (attn_w * mnar_emb).sum(dim=2, keepdim=True)   # (B, V, 1, d)
            )
            x = torch.cat([x[:, :, :1] + global_mnar, x[:, :, 1:]], dim=2)

        return x


class SMILELeanEncoder(nn.Module):
    """Lean encoder: MNAR co-occurrence bias + VarAtt FiLM + local obs density.

    Removes from SMILEv2FiLMEncoder:
      - MissingPatternEncoder (1D/2D CNN) -- heavy, redundant with MNAR cooccur bias
      - MNARCrossAttention per block      -- per-layer overhead removed
      - TimeFiLMMLPBlock -> MLPBlock      -- FFN needs no time conditioning
      - Global density injection to CLS   -- mean(T) destroys temporal dynamics
      - Attention-pooled MNAR to CLS      -- manual injection fights Self-Attention

    Keeps / improves vs. original SMILELean:
      - MNARCooccurrenceEncoder           -- zero-param co-occurrence bias
      - MNARBiasFiLMVarAttBlock           -- MNAR bias + FiLM + time-dynamic MNAR scale
      - DensityMLPEmbedder                -- density as 3rd input feature (not additive)
      - TimeEncoder (dual-track PE)       -- physical time added to x after index PE
      - PositionalEncoding                -- sequence-index sinusoidal PE

    Ablation switches (via args):
      - abl_no_density:    use MLPEmbedder instead of DensityMLPEmbedder
      - abl_no_mnar_bias:  disable MNAR co-occurrence attention bias
      - abl_no_film:       disable time-conditional FiLM on VarAtt
      - abl_no_time_mnar:  disable time-dynamic MNAR scaling only
      - abl_no_time_pe:    disable physical-time positional encoding
    """

    def __init__(self, args):
        super().__init__()
        self.time_dim = getattr(args, 'time_dim', 16)
        self.obs_density_window = getattr(args, 'obs_density_window', 5)
        # Ablation flags
        self.abl_no_density = getattr(args, 'abl_no_density', False)
        self.abl_no_mnar_bias = getattr(args, 'abl_no_mnar_bias', False)
        self.abl_no_film = getattr(args, 'abl_no_film', False)
        self.abl_no_time_mnar = getattr(args, 'abl_no_time_mnar', False)
        self.abl_no_time_pe = getattr(args, 'abl_no_time_pe', False)
        # Embedder: density-aware or plain
        if self.abl_no_density:
            self.embedder = MLPEmbedder(args.d_model)
        else:
            self.embedder = DensityMLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        # Sinusoidal PE for sequence index
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        # Learnable time encoder for FiLM conditioning AND dual-track physical-time PE
        self.time_encoder = TimeEncoder(self.time_dim)
        # Project time_enc -> d_model for additive physical-time PE (zero-init)
        if not self.abl_no_time_pe:
            self.time_pe_proj = nn.Linear(self.time_dim, args.d_model)
            nn.init.zeros_(self.time_pe_proj.weight)
            nn.init.zeros_(self.time_pe_proj.bias)
        # MNAR co-occurrence encoder (zero-param)
        if not self.abl_no_mnar_bias:
            self.mnar_cooccur_encoder = MNARCooccurrenceEncoder()
        self.blocks = nn.ModuleList([
            SMILELeanBasicBlock(
                dim=args.d_model, num_heads=args.n_heads, time_dim=self.time_dim,
                mlp_ratio=4., qkv_bias=False, proj_drop=args.dropout,
                abl_no_film=self.abl_no_film,
                abl_no_mnar_bias=self.abl_no_mnar_bias,
                abl_no_time_mnar=self.abl_no_time_mnar,
            )
            for _ in range(args.e_layers)
        ])

    def forward(self, x, lens, mask, time=None, original_mask=None, **kwargs):
        # Compute local observation density for the embedder
        if self.abl_no_density:
            x = self.embedder(x, mask)                          # (B, V, T, d)
        else:
            if original_mask is not None:
                B_m, T_m, V_m = original_mask.shape
                ws = self.obs_density_window
                m = original_mask.float().permute(0, 2, 1).reshape(B_m * V_m, 1, T_m)
                d = F.avg_pool1d(m, kernel_size=ws, stride=1, padding=ws // 2)
                density = d.reshape(B_m, V_m, T_m).permute(0, 2, 1)  # (B, T, V)
            else:
                density = torch.zeros_like(mask)
            x = self.embedder(x, mask, density)                 # (B, V, T, d)

        mnar_cooccur = None
        if not self.abl_no_mnar_bias and original_mask is not None:
            mnar_cooccur = self.mnar_cooccur_encoder(original_mask)  # (B, V, V)

        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                        # (B, V, T+1, d)

        # Dual-track positional encoding:
        #   1. sinusoidal index PE (tells SeqAtt "this is the k-th obs in the sequence")
        x = self.position_enc(x)

        lens_mask = length_to_mask(lens + 1)
        mask_full = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask_full = mask_full.transpose(1, 2).float()

        time_enc = None
        if time is not None:
            cls_time = torch.zeros(x.shape[0], 1, device=time.device, dtype=time.dtype)
            time_full = torch.cat([cls_time, time], dim=1)      # (B, T+1)
            time_enc = self.time_encoder(time_full)              # (B, T+1, time_dim)
            #   2. physical-time PE (zero-init -> no contribution at init)
            if not self.abl_no_time_pe:
                time_pe = self.time_pe_proj(time_enc).unsqueeze(1)  # (B, 1, T+1, d_model)
                x = x + time_pe

        for block in self.blocks:
            x = block(x, mask_full, lens, lens_mask, mnar_cooccur, time_enc)

        return x


# ============================================================
# SMILE-Lean V2: Dynamic MNAR + Variable Policy + Dual-Head
# ============================================================


class PolicyDensityEmbedder(nn.Module):
    """Embedder with variable-level observation policy tokens.

    Replaces DensityMLPEmbedder. Combines continuous (value, mask, density)
    projection with learnable policy embeddings that distinguish observed,
    recently-missing, and long-missing states per variable.

    The recency gate is learned (not a fixed threshold), allowing the model
    to discover variable-adaptive density-to-recency mappings.

    Input:
        x             (B, T, V) -- observed values
        mask          (B, T, V) -- observation mask (possibly with dropout)
        density       (B, T, V) -- local obs density from avg_pool1d
        original_mask (B, T, V) or None -- clean mask before dropout
    Output: (B, V, T, d_model)
    """

    def __init__(self, d_model, use_policy_tokens=True):
        super().__init__()
        self.use_policy_tokens = use_policy_tokens
        self.embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Linear(d_model, d_model),
        )
        if use_policy_tokens:
            # Learnable policy tokens (small random init)
            self.embed_observed = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            self.embed_recent_missing = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            self.embed_long_missing = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            # Learned recency gate: density -> soft weight (variable-adaptive)
            self.recency_gate = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, mask, density, original_mask=None):
        # DataLoader/collate may surface masks as integer tensors; keep the
        # policy-density pathway explicitly in floating point for Linear layers.
        x = x.float()
        mask = mask.float()
        density = density.float()
        # Continuous feature projection
        inp = torch.stack((x, mask, density), dim=-1)   # (B, T, V, 3)
        out = self.embed(inp)                            # (B, T, V, d_model)
        if not self.use_policy_tokens:
            return out.permute(0, 2, 1, 3)              # (B, V, T, d_model)

        # Determine missing status from clean mask if available
        if original_mask is not None:
            is_missing = (original_mask < 0.5).float()   # (B, T, V)
        else:
            is_missing = (mask < 0.5).float()

        # Learned recency: density -> [0, 1] interpolation weight
        recency = self.recency_gate(density.unsqueeze(-1))  # (B, T, V, 1)

        # Policy embedding: observed vs (recent_missing <-> long_missing)
        is_obs = 1.0 - is_missing                        # (B, T, V)
        policy_emb = (
            is_obs.unsqueeze(-1) * self.embed_observed
            + is_missing.unsqueeze(-1) * (
                recency * self.embed_recent_missing
                + (1.0 - recency) * self.embed_long_missing
            )
        )                                                 # (B, T, V, d_model)

        return (out + policy_emb).permute(0, 2, 1, 3)    # (B, V, T, d_model)


class DynamicMNARCooccurrenceEncoder(nn.Module):
    """Compute per-frame MNAR co-occurrence using causal sliding windows.

    Replaces global static MNARCooccurrenceEncoder. Produces per-timestep
    (B, T, V, V) co-occurrence capturing local, time-varying measurement
    co-missingness patterns via causal windowed computation.

    Uses F.pad + unfold for memory-efficient windowed computation.
    Output is padded to (B, T+1, V, V) for CLS token alignment.

    Args:
        window_size: Causal window size for local co-occurrence (default 5).
    """

    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def forward(self, original_mask):
        # original_mask: (B, T, V), 1=observed, 0=missing
        B, T, V = original_mask.shape
        ws = self.window_size
        missing = 1.0 - original_mask.float()             # (B, T, V)

        # Causal padding: look at current + past ws-1 steps
        padded = F.pad(missing, (0, 0, ws - 1, 0))       # (B, T+ws-1, V)

        # Extract causal windows via unfold
        windows = padded.unfold(1, ws, 1)                 # (B, T, V, ws)

        # Batch matmul for co-occurrence per frame
        w = windows.reshape(B * T, V, ws)                 # (B*T, V, ws)
        co_occur = torch.bmm(w, w.transpose(1, 2)) / ws  # (B*T, V, V)
        co_occur = co_occur.reshape(B, T, V, V)

        # Subtract independent expectation (correlation form)
        marginal = windows.mean(dim=-1)                   # (B, T, V)
        expected = marginal.unsqueeze(-1) * marginal.unsqueeze(-2)  # (B, T, V, V)
        co_occur = co_occur - expected

        # Pad for CLS token alignment: (B, T+1, V, V)
        cls_pad = torch.zeros(B, 1, V, V, device=co_occur.device, dtype=co_occur.dtype)
        return torch.cat([cls_pad, co_occur], dim=1)


class DynamicMNARBiasVarAttention(nn.Module):
    """V2.1 VarAttention: Static Base Features + Dynamic MNAR Graph.
    
    Restores the robust temporal-mean Q/K mechanism from V1 to compute a stable 
    base graph, then expands it to inject the local, frame-by-frame dynamic MNAR bias.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mnar_bias_scale = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, var_decay=None):
        B, N, P, C = x.shape

        # 1. 提取 QKV
        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 2, 4, 1, 5)          # (3, B, P, heads, N, head_dim)
        q_all, k_all, v_all = qkv.unbind(0)          # each: (B, P, heads, N, head_dim)

        # =========================================================
        # 2. 恢复 V1 极其强壮的“全局 Q/K 机制” (计算静态基座)
        # =========================================================
        # Q: 仅使用 CLS token
        q = q_all[:, 0]                              # (B, heads, N, head_dim)
        
        # K: 使用全序列时间均值
        mask_r = mask.reshape(B, N, P, 1, 1).repeat(1, 1, 1, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        k = k_all.masked_fill(~mask_r.bool(), 0).sum(dim=1) / (mask_r.sum(dim=1) + 1e-6) # (B, heads, N, head_dim)

        # 计算稳定的基准注意力 Logits
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # =========================================================
        # 3. 展开到时间维度，并注入 V2 局域动态 MNAR
        # =========================================================
        # 将静态 Logits 复制 P 份： (B, heads, P, N, N)
        logits = logits.unsqueeze(2).expand(-1, -1, P, -1, -1).clone()

        if mnar_cooccur is not None:
            # mnar_cooccur: (B, P, N, N)
            scale = self.mnar_bias_scale.view(1, -1, 1, 1, 1)
            bias = mnar_cooccur.unsqueeze(1) * scale          # (B, heads, P, N, N)
            
            if var_decay is not None:
                # var_decay: (B, heads, N, N) -> (B, heads, 1, N, N)
                bias = bias * var_decay.unsqueeze(2)
            
            # 在稳定基座上打上动态医疗策略的补丁！
            logits = logits + bias

        # =========================================================
        # 4. Softmax 并逐帧作用于 Value
        # =========================================================
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.proj_drop(attn_weights)           # (B, heads, P, N, N)

        # 调整 V 的形状以匹配批量矩阵乘法
        v = v_all.permute(0, 2, 1, 3, 4)                      # (B, heads, P, N, head_dim)
        
        # 逐帧进行矩阵相乘：(N, N) @ (N, head_dim) -> (N, head_dim)
        x_out = torch.matmul(attn_weights, v)                 # (B, heads, P, N, head_dim)

        # =========================================================
        # 5. 还原形状并输出
        # =========================================================
        x_out = x_out.permute(0, 3, 2, 1, 4).reshape(B, N, P, C)
        
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class DynamicMNARBiasFiLMVarAttBlock(nn.Module):
    """VarAttBlock with per-frame dynamic MNAR bias, variable-level time decay,
    and post-attention FiLM modulation.

    Combines:
    - DynamicMNARBiasVarAttention: per-frame cross-variable attention + MNAR bias
    - Variable-level time decay: per-variable, per-head temporal modulation
    - Post-attention FiLM: time-conditional scaling of attention output

    All new parameters zero-initialized for safe integration.
    """

    def __init__(self, dim, num_heads, time_dim, qkv_bias=False, proj_drop=0.,
                 norm_layer=nn.LayerNorm, use_mnar_bias=True, use_time_mnar=True,
                 use_film=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.use_mnar_bias = use_mnar_bias
        self.use_time_mnar = use_time_mnar
        self.use_film = use_film
        self.attn_var = DynamicMNARBiasVarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop
        )
        if use_film:
            # Post-attention FiLM (zero-init = identity)
            self.film_gen = nn.Linear(time_dim, 2 * dim)
            nn.init.zeros_(self.film_gen.weight)
            nn.init.zeros_(self.film_gen.bias)
        if use_time_mnar:
            # Variable-level time decay projection (zero-init)
            self.var_time_proj = nn.Linear(time_dim, num_heads)
            nn.init.zeros_(self.var_time_proj.weight)
            nn.init.zeros_(self.var_time_proj.bias)

    def compute_var_time_decay(self, time_enc, mask):
        """Compute variable-level pairwise time decay for MNAR bias modulation.

        Args:
            time_enc: (B, T+1, time_dim)
            mask:     (B, V, T+1) -- observation mask used as weight

        Returns:
            pair_decay: (B, heads, V, V)
        """
        # Exclude CLS token at t=0; only real physical timesteps should shape
        # variable time statistics.
        w = mask[:, :, 1:].float()                             # (B, V, T)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (B, V, 1)
        t_enc = time_enc[:, 1:, :]                              # (B, T, time_dim)
        var_time = torch.bmm(w, t_enc) / w_sum                  # (B, V, time_dim)
        var_decay = torch.sigmoid(self.var_time_proj(var_time))  # (B, V, num_heads)
        var_decay = var_decay.permute(0, 2, 1)                  # (B, num_heads, V)
        pair_decay = (var_decay.unsqueeze(-1) + var_decay.unsqueeze(-2)) / 2
        return pair_decay                                        # (B, heads, V, V)

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, time_enc=None):
        var_decay = None
        if self.use_time_mnar and time_enc is not None and mnar_cooccur is not None:
            var_decay = self.compute_var_time_decay(time_enc, mask)

        mnar_bias = mnar_cooccur if self.use_mnar_bias else None
        attn_out = self.attn_var(self.norm1(x), mask, lens, lens_mask,
                                  mnar_bias, var_decay)
        if self.use_film and time_enc is not None:
            film = self.film_gen(time_enc)              # (B, T+1, 2*dim)
            gamma, beta = film.chunk(2, dim=-1)
            attn_out = (1.0 + gamma.unsqueeze(1)) * attn_out + beta.unsqueeze(1)
        return x + attn_out


class SMILELeanV2BasicBlock(nn.Module):
    """V2 transformer block: SeqAtt + DynamicMNARBiasFiLMVarAtt + standard MLP.

    Upgrades from SMILELeanBasicBlock:
    - Per-frame dynamic MNAR bias (replaces static global co-occurrence)
    - Variable-level time decay (replaces sample-level head gate)
    - Per-frame cross-variable attention (replaces CLS-only variable attention)
    """

    def __init__(self, dim, num_heads, time_dim, mlp_ratio=4., qkv_bias=False,
                 proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 abl_no_mnar_bias=False, abl_no_time_mnar=False, abl_no_film=False):
        super().__init__()
        self.seq_att_block = SeqAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.var_att_block = DynamicMNARBiasFiLMVarAttBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            qkv_bias=qkv_bias, proj_drop=proj_drop, norm_layer=norm_layer,
            use_mnar_bias=not abl_no_mnar_bias,
            use_time_mnar=not abl_no_time_mnar,
            use_film=not abl_no_film,
        )
        self.mlp = MLPBlock(
            dim=dim, mlp_ratio=mlp_ratio,
            proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer,
        )

    def forward(self, x, mask, lens, lens_mask, mnar_cooccur=None, time_enc=None):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        x = self.var_att_block(x, mask, lens, lens_mask, mnar_cooccur, time_enc)
        x = self.mlp(x)
        return x


class SMILELeanV2Encoder(nn.Module):
    """SMILE-Lean V2 encoder with dynamic MNAR and variable policy embeddings.

    Key changes from SMILELeanEncoder (V1):
    - PolicyDensityEmbedder: variable-level observation policy tokens replace
      scalar density, providing signed missingness representation.
    - DynamicMNARCooccurrenceEncoder: causal windowed (B, T+1, V, V) co-occurrence
      replaces global static (B, V, V), capturing local temporal dynamics.
    - DynamicMNARBiasVarAttention: per-frame cross-variable attention with
      dynamic MNAR bias, replacing CLS-only attention with static bias.
    - Variable-level time decay: per-variable per-head temporal modulation of
      MNAR bias, replacing sample-level head gate.
    - Time-PE removed by default (ablation showed it hurts multi-task).
      Kept as toggle (v2_use_time_pe) for backward compatibility.

    Compatible with EmbeddingDecoder for pretraining (same output shape).
    """

    def __init__(self, args):
        super().__init__()
        self.time_dim = getattr(args, 'time_dim', 16)
        self.obs_density_window = getattr(args, 'obs_density_window', 5)
        self.use_time_pe = getattr(args, 'v2_use_time_pe', False)
        self.abl_no_density = getattr(args, 'abl_no_density', False)
        self.abl_no_policy = getattr(args, 'abl_no_policy', False)
        self.abl_no_mnar_bias = getattr(args, 'abl_no_mnar_bias', False)
        self.abl_no_dynamic_mnar = getattr(args, 'abl_no_dynamic_mnar', False)
        self.abl_no_time_mnar = getattr(args, 'abl_no_time_mnar', False)
        self.abl_no_film = getattr(args, 'abl_no_film', False)

        # PolicyDensityEmbedder replaces DensityMLPEmbedder
        self.embedder = PolicyDensityEmbedder(
            args.d_model,
            use_policy_tokens=not self.abl_no_policy,
        )
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.time_encoder = TimeEncoder(self.time_dim)

        # Optional time PE (default off for v2)
        if self.use_time_pe:
            self.time_pe_proj = nn.Linear(self.time_dim, args.d_model)
            nn.init.zeros_(self.time_pe_proj.weight)
            nn.init.zeros_(self.time_pe_proj.bias)

        # Dynamic MNAR co-occurrence encoder
        if not self.abl_no_mnar_bias:
            if self.abl_no_dynamic_mnar:
                self.mnar_cooccur_encoder = MNARCooccurrenceEncoder()
            else:
                self.mnar_cooccur_encoder = DynamicMNARCooccurrenceEncoder(
                    window_size=self.obs_density_window
                )

        # V2 transformer blocks
        self.blocks = nn.ModuleList([
            SMILELeanV2BasicBlock(
                dim=args.d_model, num_heads=args.n_heads, time_dim=self.time_dim,
                mlp_ratio=4., qkv_bias=False, proj_drop=args.dropout,
                abl_no_mnar_bias=self.abl_no_mnar_bias,
                abl_no_time_mnar=self.abl_no_time_mnar,
                abl_no_film=self.abl_no_film,
            )
            for _ in range(args.e_layers)
        ])

    def forward(self, x, lens, mask, time=None, original_mask=None, **kwargs):
        # Compute local observation density
        if (not self.abl_no_density) and original_mask is not None:
            B_m, T_m, V_m = original_mask.shape
            ws = self.obs_density_window
            m = original_mask.float().permute(0, 2, 1).reshape(B_m * V_m, 1, T_m)
            d = F.avg_pool1d(m, kernel_size=ws, stride=1, padding=ws // 2)
            density = d.reshape(B_m, V_m, T_m).permute(0, 2, 1)  # (B, T, V)
        else:
            density = torch.zeros_like(mask, dtype=x.dtype)

        # Embed with policy tokens
        x = self.embedder(x, mask, density, original_mask)    # (B, V, T, d)

        # Dynamic MNAR co-occurrence
        mnar_cooccur = None
        if (not self.abl_no_mnar_bias) and original_mask is not None:
            if self.abl_no_dynamic_mnar:
                static_mnar = self.mnar_cooccur_encoder(original_mask)     # (B, V, V)
                num_frames = x.shape[2] + 1
                mnar_cooccur = static_mnar.unsqueeze(1).repeat(1, num_frames, 1, 1)
                mnar_cooccur[:, 0] = 0.0
            else:
                mnar_cooccur = self.mnar_cooccur_encoder(original_mask)    # (B, T+1, V, V)

        # CLS token
        x = torch.cat(
            (self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2
        )                                                      # (B, V, T+1, d)

        # Sinusoidal PE (sequence index)
        x = self.position_enc(x)

        # Mask setup
        lens_mask = length_to_mask(lens + 1)
        mask_full = torch.cat(
            (torch.ones(mask.shape[0], 1, mask.shape[-1],
                        device=mask.device, dtype=mask.dtype), mask),
            dim=1,
        )
        mask_full = mask_full.transpose(1, 2).float()          # (B, V, T+1)

        # Time encoding
        time_enc = None
        if time is not None:
            cls_time = torch.zeros(x.shape[0], 1, device=time.device, dtype=time.dtype)
            time_full = torch.cat([cls_time, time], dim=1)     # (B, T+1)
            time_enc = self.time_encoder(time_full)            # (B, T+1, time_dim)
            # Optional time PE (default off in v2)
            if self.use_time_pe:
                time_pe = self.time_pe_proj(time_enc).unsqueeze(1)  # (B, 1, T+1, d)
                x = x + time_pe

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask_full, lens, lens_mask, mnar_cooccur, time_enc)

        return x


class DualHeadClassifier(nn.Module):
    """Dual-head classifier: CLS representation + missingness temporal summary.

    Branch 1: Standard CLS token processing (same as Classifier).
    Branch 2: Temporal missingness summary -- splits original_mask into
              early/mid/late segments and pools each, providing the classifier
              with explicit observation-policy information aligned with T3/T4
              temporal dynamics evidence.

    When original_mask is None, branch 2 receives zeros (graceful fallback).

    Args:
        args: Namespace with d_model, input_dim, num_class, dropout.
    """

    def __init__(self, args):
        super().__init__()
        d = args.d_model
        V = args.input_dim
        # Branch 1: CLS tokens
        self.cls_mlp = MLPBlock(
            dim=d, mlp_ratio=4, proj_drop=args.dropout,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        )
        # Branch 2: missingness temporal summary (early/mid/late -> 3*V features)
        self.mask_proj = nn.Sequential(
            nn.Linear(V * 3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # Fusion: CLS (V*d) + mask_summary (d) -> num_class
        self.out = nn.Linear(d * V + d, args.num_class)

    def forward(self, h, original_mask=None, **kwargs):
        B, V, T_plus1, H = h.shape
        # Branch 1: CLS tokens -> MLP -> flatten
        cls_token = h[:, :, 0]                              # (B, V, H)
        cls_token = self.cls_mlp(cls_token)                 # (B, V, H)
        cls_flat = cls_token.reshape(B, -1)                 # (B, V*H)

        # Branch 2: temporal missingness summary (early/mid/late)
        if original_mask is not None:
            # original_mask: (B, T, V)
            T_m = original_mask.shape[1]
            t1 = T_m // 3
            t2 = 2 * T_m // 3
            early = original_mask[:, :t1].float().mean(dim=1)     # (B, V)
            mid = original_mask[:, t1:t2].float().mean(dim=1)     # (B, V)
            late = original_mask[:, t2:].float().mean(dim=1)      # (B, V)
            mask_summary = torch.cat([early, mid, late], dim=-1)  # (B, 3*V)
        else:
            mask_summary = torch.zeros(B, V * 3, device=h.device, dtype=h.dtype)

        mask_emb = self.mask_proj(mask_summary)             # (B, H)

        # Fusion
        fused = torch.cat([cls_flat, mask_emb], dim=-1)     # (B, V*H + H)
        return self.out(fused)
