import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

from einops import rearrange, repeat
from mamba_ssm.modules.block import Block
import torch.utils.checkpoint as checkpoint
from torchvision.transforms import functional as TF

# 0) MambaConfig
#########################################
class MambaConfig:
    def __init__(self):
        # 1) Mamba2
        self.d_model = 1024
        self.d_state = 256
        self.d_conv = 4
        self.expand = 2
        self.headdim = 128
        self.ngroups = 1
        self.A_init_range = (1 , 16)
        self.dt_min=0.001
        self.dt_max=0.02
        self.dt_init_floor=1e-4
        self.dt_limit=(0.0, float("inf"))
        self.learnable_init_states=False
        self.activation="swish"
        self.mamba_bias=False
        self.mamba_conv_bias=True
        self.chunk_size=256
        self.use_mem_eff_path=True

        # 2) Policy (多相机+lowdim)
        self.camera_names = ['head_camera']
        self.embed_dim=1024    # each camera output
        self.lowdim_dim=14       # state_dim
        self.action_dim=14
        self.num_blocks=4

#########################################
# 1)  Mamba2
import math


class SinusoidalPosEmb(nn.Module):
    """时间位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if x.dim() == 1:
            emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        else:
            emb = x.unsqueeze(-1) * emb.view(1, 1, -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SymplecticHead(nn.Module):
    """
    [TRO Architecture]
    Gated-Cross-Symmetry 动力学头：结合 Cross-Attention 与 GLU 门控。
    """
    def __init__(self, action_dim, cond_dim, hidden_dim=512, future_steps=100):
        super().__init__()
        self.future_steps = future_steps
        self.action_dim = action_dim // future_steps # 14
        
        # 1. 物理状态投影
        self.phase_proj = nn.Linear(self.action_dim * 2, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        # 2. 空间锚定：Cross-Attention (Q:动作, KV:环境)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 3. 门控演化层 (GLU 变体)：实现跨模态筛选
        # 显式构造门控结构，无需调用 nn.GLU
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2) 
        self.cond_adapter = nn.Linear(cond_dim, hidden_dim)
        
        # 4. 最终输出
        self.final_proj = nn.Linear(hidden_dim, self.action_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, q, v, t, cond):
        """
        q, v: [B, L, 224]
        cond: [B, L, 2048]
        t: [B] (采样时刻)
        """
        B, L, _ = q.shape
        # 维度还原为序列
        q_seq = q.view(B * L, self.future_steps, self.action_dim)
        v_seq = v.view(B * L, self.future_steps, self.action_dim)
        
        # 1. 嵌入
        phase_feat = self.phase_proj(torch.cat([q_seq, v_seq], dim=-1)) # [NM, 16, H]
        if t.dim() == 1:
            t = t.view(B, 1).expand(B, L).reshape(-1)
        t_emb = self.time_mlp(t).unsqueeze(1) # [NM, 1, H]
        
        # 2. 跨模态空间对齐
        # 这里的 cond 需要适配 [NM, 1, D]
        ctx = self.cond_adapter(cond.view(B * L, -1)).unsqueeze(1) # [NM, 1, H]
        x = phase_feat + t_emb
        
        # 动作检索环境特征
        attn_out, _ = self.cross_attn(x, ctx, ctx)
        x = self.norm1(x + attn_out)
        
        # 3. 门控提纯 (GLU 逻辑)
        # 让环境特征 ctx 动态地为动作序列打分
        gate_input = self.gate_proj(x)
        val, gate = gate_input.chunk(2, dim=-1)
        x = val * torch.sigmoid(gate + ctx) # 环境特征直接干预门控开关
        
        # 4. 输出力场
        acc = self.final_proj(x) # [NM, 100, 14]
        return acc.view(B, L, -1)   

        
class CNNSymplecticHead(nn.Module):
    """
    [T-RO 薛定谔逆动力学头] 
    接收条件: [x_fused (1024) + IMLE预测的未来目标 Z_goal (512)] = 1536
    作为拉扯动作质点的“未来引力势场”。
    """
    def __init__(self, action_dim, visual_dim=1024, hidden_dim=1024, future_steps=32):
        super().__init__()
        self.future_steps = future_steps
        self.action_dim = action_dim // future_steps 
        
        self.phase_proj = nn.Sequential(
            nn.Linear(self.action_dim * 2, 196), 
            nn.Mish(inplace=True),
            nn.Linear(196, hidden_dim), 
            nn.LayerNorm(hidden_dim) 
        )
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(hidden_dim), 
                                      nn.Linear(hidden_dim, hidden_dim), 
                                      nn.Mish())
        
        # 接收 1024 维度的强力物理锚定特征
        self.visual_adapter = nn.Sequential(nn.Linear(visual_dim, hidden_dim), 
                                            nn.LayerNorm(hidden_dim))
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.Mish(inplace=True)
        
        self.final_proj = nn.Linear(hidden_dim, self.action_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, q, v, t, visual_cond):
        # 统一维度识别
        if q.dim() == 2: 
            B = q.shape[0]
            L = 1
            q = q.unsqueeze(1)
            v = v.unsqueeze(1)
        else:
            B, L, _ = q.shape

        q_seq = q.view(B * L, self.future_steps, self.action_dim)
        v_seq = v.view(B * L, self.future_steps, self.action_dim)
        
        x = self.phase_proj(torch.cat([q_seq, v_seq], dim=-1)) 
        x = x.transpose(1, 2) 
        
        if t.dim() == 1 and t.shape[0] == B:
            t = t.view(B, 1).expand(B, L).reshape(-1)
        elif t.dim() == 1 and t.shape[0] == B * L:
            pass 
            
        v_flat = visual_cond.reshape(B * L, -1) # 此时为 [B*L, 1024]
        c = self.time_mlp(t) + self.visual_adapter(v_flat) # [B*L, H]
        c = c.unsqueeze(-1) # [B*L, H, 1]
        
        res = x
        x = self.act(self.conv1(x + c))
        x = x + res
        
        res = x
        x = self.act(self.conv2(x + c))
        x = x + res
        
        x = x.transpose(1, 2) 
        acc = self.final_proj(x) 
        
        return acc.view(B, L, -1)


class SwiGLU(nn.Module):
    """
    [引入大语言模型 LLaMA 的核心算子]
    比传统 ReLU/GELU 具有更强的高维特征表达能力，专门处理复杂的物理与几何特征。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4 * 2 / 3) # LLaMA 黄金比例
        
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PointCloudMambaFusion(nn.Module):
    """
    1D CNN + ReLU + BN + MaxPool，极简即是极强！
    增加双路输出：(Mamba意图输入, 瞬时物理高保真特征)
    """
    def __init__(self, embed_dim=1024, use_rgb=True, proprio_dim=16):
        super().__init__()
        self.use_rgb = use_rgb
        self.embed_dim = embed_dim
        
        pc_dim = 6 if use_rgb else 3 
        
        # 1. 外部世界认知器 (经典 4 层 CNN + ReLU)
        self.point_net = nn.Sequential(
            nn.Conv1d(pc_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True), 

            nn.Conv1d(64, 128, kernel_size=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True), 

            nn.Conv1d(128, 256, kernel_size=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace=True), 

            nn.Conv1d(256, 512, kernel_size=1), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), 
        )

        # 2. 内部自我感知器 
        self.proprio_projector = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True), # 保持全网络激活函数的一致性
            nn.Linear(128, 512),
            nn.LayerNorm(512) 
        )
        
        # 3. 终极消化回路
        self.pre_norm = nn.LayerNorm(512 + 512)
        
    def forward(self, point_cloud, proprio_embed):

        if self.use_rgb:
            processed_pc = point_cloud 
        else:
            processed_pc = point_cloud[..., :3]

        # 1. 模态独立提取
        pc_t = processed_pc.transpose(1, 2).contiguous()
        point_features = self.point_net(pc_t) # [B, 512, N]
        
        # 绝对宏观几何
        global_pc_feature = torch.max(point_features, dim=-1)[0] # [B, 512]
        
        # 2. 本体感特征升维 
        high_dim_proprio = self.proprio_projector(proprio_embed) # [B, 512]

        # 3. 模态融合
        concat_state = torch.cat([global_pc_feature, high_dim_proprio], dim=-1)
        
        # 【关键节点】：保存融合后的瞬时物理真值
        x_fused = self.pre_norm(concat_state) # [B, 1024]

        # 返回：(给Mamba的序列输入, 给WM)
        return x_fused


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class CrossAttention(nn.Module):

    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)  # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]

        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, horizon]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]

        return attn_output
    
import einops
from einops.layers.torch import Rearrange   

class ConditionalResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        condition_type="film",
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == "film":  # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "add":
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "cross_attention_add":
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == "cross_attention_film":
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == "mlp_film":
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond=None):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        if cond is not None:
            if self.condition_type == "film":
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == "add":
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == "cross_attention_add":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)  # [batch_size, out_channels, horizon]
                out = out + embed
            elif self.condition_type == "cross_attention_film":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == "mlp_film":
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class IMLE_Generator(nn.Module):
    """
    [T-RO 终极形态] 纯血1D-UNet
    输入: Z 序列 (纯随机噪声的轨迹形状)
    条件: Mamba 意图 (纯净的宏观因果指令)
    """
    def __init__(
        self,
        action_dim=16,
        future_steps=16,
        mamba_dim=1024,
        obs_dim=1024,       # 【修改处】：接收 x_fused 的 1024 维
        down_dims=[512, 1024, 2048], # 4090 极速收敛版通道数
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.action_dim = action_dim
        
        # 1. 无损全频段广播 1024 + 1024 = 2048
        global_cond_dim = mamba_dim + obs_dim
        # 【修改处】：极简 MLP，防止 2048 维产生过拟合，去掉了 Mish 和冗余的 Linear
        self.cond_mlp = nn.Sequential(
            nn.Linear(global_cond_dim, global_cond_dim),
            nn.LayerNorm(global_cond_dim)
        )
        
        # 2. 初始特征卷积：把 14 维的 Z 噪声序列，映射到 128 维特征空间
        start_dim = down_dims[0]
        self.init_conv = Conv1dBlock(action_dim, start_dim, kernel_size=kernel_size)
        
        # 3. U-Net 结构 (Down -> Mid -> Up)
        in_out = list(zip([start_dim] + down_dims[:-1], down_dims))
        
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
                    ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                ])
            )
            
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
        ])
        
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
                    ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=global_cond_dim, kernel_size=kernel_size, n_groups=n_groups, condition_type="film"),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                ])
            )
            
        # 4. 最终还原为轨迹
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, action_dim, 1),
        )

    def forward(self, mamba_cond, z_seq, obs_cond):
        """
        mamba_cond: [B, 1024]
        z_seq: [B, action_dim, future_steps] <- 【核心变革：Z 直接作为输入序列】
        """
        B = mamba_cond.shape[0]
        
        # 1. 提取全局条件
        # 暴力拼接 2048 维
        combined_cond = torch.cat([mamba_cond, obs_cond], dim=-1)
        global_feature = self.cond_mlp(combined_cond) # [B, 2048]
        
        # 2. 从 Z 噪声序列起步
        x = self.init_conv(z_seq) # [B, 128, 16]
        
        # 3. U-Net 雕刻过程
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
            
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
            
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1) 
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
            
        # 4. 输出打平
        x = self.final_conv(x) # [B, action_dim, 16]
        x = einops.rearrange(x, "b d t -> b t d").contiguous() 
        return x.view(B, -1) # [B, 224]

class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=512,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.02,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # # Conv step
        # if causal_conv1d_update is None:
        #     conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        #     conv_state[:, :, -1] = xBC
        #     xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
        #     if self.conv1d.bias is not None:
        #         xBC = xBC + self.conv1d.bias
        #     xBC = self.act(xBC).to(dtype=dtype)
        # else:
        #     xBC = causal_conv1d_update(
        #         xBC,
        #         conv_state,
        #         rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #         self.conv1d.bias,
        #         self.activation,
        #     )
        # === 强制对齐类型，防止混合精度导致的算子报错 ===
        # === [修正版] Conv step ===
        # 1. 确保输入 xBC 的类型与权重一致 (BF16/FP16)，防止算子报错
        # # Conv step
        # # 1. 强制类型对齐：以卷积层权重的精度为准 (通常是 BF16)
        # # 1. 确保类型对齐 (防止 BF16/FP32 冲突)
        # target_dtype = self.conv1d.weight.dtype
        # if xBC.dtype != target_dtype:
        #     xBC = xBC.to(target_dtype)
        # if conv_state.dtype != target_dtype:
        #     conv_state = conv_state.to(target_dtype)

        # # 2. 强制走进 Python 分支 (注释掉 causal_conv1d_update 的判断)
        # # if causal_conv1d_update is None: <--- 原来的判断
        if True: # 强制为 True，绕过底层算子
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            # 这里的 rearrange 也是纯 PyTorch 操作，非常安全
            weight_reshaped = rearrange(self.conv1d.weight, "d 1 w -> d w")
            xBC = torch.sum(conv_state * weight_reshaped, dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            # 调用算子：确保所有输入都是 Contiguous 且 Dtype 一致
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                conv_weight,
                conv_bias,
                self.activation,
            )
            # === [修复结束] ===
        # ===end change ===
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

#########################################
# 2) MambaPolicy (相机 + lowdim -> Mamba2 -> action)
#########################################
import torch
# import torch.nn as nn
from torchvision import models

class MambaPolicy(nn.Module):
    """
    多相机 + lowdim -> backbone -> concat/sum -> in_proj -> Block (with Mamba2) -> action
    """
    def __init__(
        self,
        camera_names,
        embed_dim=2048,
        lowdim_dim=14,
        d_model=2048,
        action_dim=14,
        num_blocks=4,  # 支持多个 Block
        block_cfg=None,  # Block 的配置
        mamba_cfg=None,  # Mamba2 的配置
        future_steps=16,  # 预测未来16步，可调
    ):
        super().__init__()
        self.camera_names = camera_names
        self.future_steps = future_steps
        self.lowdim_dim = lowdim_dim
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.action_dim = action_dim
        if mamba_cfg is None or not isinstance(mamba_cfg, MambaConfig):
            mamba_cfg = MambaConfig()
        self.mamba_cfg = mamba_cfg
        self.flat_action_dim = action_dim * self.future_steps

        self.num_cameras = len(camera_names)
        # 植入 3D 引擎
        self.fusion_engine = PointCloudMambaFusion(
            embed_dim=self.embed_dim,       # 2048
            use_rgb=True,
            proprio_dim=self.lowdim_dim     # 14
        )

        self.num_imle_samples = 5 

        self.imle_generator = IMLE_Generator(
            action_dim=self.action_dim,         
            future_steps=self.future_steps,     
            mamba_dim=self.d_model,             # 1024
            obs_dim=1024,                     # 视觉维度
            down_dims=[512, 1024, 2048]
        )
         
        self.sb_head = CNNSymplecticHead(
            action_dim=self.flat_action_dim, # 224
            visual_dim=1024,                 
            hidden_dim=1024,                
            future_steps=self.future_steps      
        )
        # 融合后的维度已经是 2048，直接透传
        self.in_proj = nn.Identity()
        
        # # 跨相机交叉注意力模块
        # Block 配置
        if block_cfg is None:
            block_cfg = {}

        # Mamba2 配置
        def mixer_fn(dim):
            return Mamba2(
                d_model=dim,
                d_state=self.mamba_cfg.d_state,
                d_conv=self.mamba_cfg.d_conv,
                expand=self.mamba_cfg.expand,
                headdim=self.mamba_cfg.headdim,
                ngroups=self.mamba_cfg.ngroups,
                A_init_range=self.mamba_cfg.A_init_range,
                dt_min=self.mamba_cfg.dt_min,
                dt_max=self.mamba_cfg.dt_max,
                dt_init_floor=self.mamba_cfg.dt_init_floor,
                dt_limit=self.mamba_cfg.dt_limit,
                chunk_size=self.mamba_cfg.chunk_size,
                use_mem_eff_path=self.mamba_cfg.use_mem_eff_path,
            )

        def mlp_fn(dim):
            hidden_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
        # 构建多个 Block
        self.blocks = nn.ModuleList([
            Block(
                dim=self.d_model,
                mixer_cls=mixer_fn,
                mlp_cls=mlp_fn,
                norm_cls=nn.LayerNorm,
                fused_add_norm=block_cfg.get("fused_add_norm", False),
                residual_in_fp32=block_cfg.get("residual_in_fp32", False),
            )
            for _ in range(num_blocks)
        ])
        # 3. Phase Space Noise
        # 训练时在相空间注入的噪声强度 (Quantum Perturbation)
        self.sigma_pos = 0.02
        self.sigma_vel = 0.02

    # def init_hidden_states(self, batch_size, device=None):
    #     """
    #     For single-step inference: gather each block's Mamba2 => allocate_inference_cache
    #     """
    #     if device is None:
    #         device = next(self.parameters()).device
    #     hidden_list = []
    #     for blk in self.blocks:
    #         if hasattr(blk.mixer, "allocate_inference_cache"):
    #             conv_st, ssm_st = blk.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=None)
    #         else:
    #             conv_st, ssm_st = None, None
    #         hidden_list.append((conv_st, ssm_st))
    #     return hidden_list
    
    def init_hidden_states(self, batch_size, device=None):
        """
        Episode 级冷启动初始化：
        每次控制器 reset 时调用，强制将二阶意图 (z_pos, z_vel) 刷新为纯净的标准高斯分布，
        彻底斩断跨 Episode 污染！
        """
        if device is None:
            device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
            
        hidden_list = []
        # 1. 正常分配每个 Block 的 Mamba2 缓存
        for blk in self.blocks:
            if hasattr(blk.mixer, "allocate_inference_cache"):
                conv_st, ssm_st = blk.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=None)
            else:
                conv_st, ssm_st = None, None
            hidden_list.append((conv_st, ssm_st))
            
        # 2. 在列表最末尾塞入本回合冷启动的初始意图与初始速度
        z_pos = torch.randn(batch_size, self.action_dim, self.future_steps, device=device, dtype=dtype)
        z_vel = torch.randn(batch_size, self.action_dim, self.future_steps, device=device, dtype=dtype)
        hidden_list.append((z_pos, z_vel))
        
        return hidden_list

    # === [辅助函数：三次样条插值 (Cubic Spline)] ===
    def cubic_spline(self, t, q0, q1):
        """
        计算连接 (q0, v0=0) 和 (q1, v1=0) 的三次样条轨迹。
        Boundary Conditions:
        x(0)=q0, x'(0)=0
        x(1)=q1, x'(1)=0
        
        Polynomial: x(t) = a + bt + ct^2 + dt^3
        v0=0 => b=0
        v1=0 => 2c + 3d = 0
        x(1)=q1 => q0 + c + d = q1
        
        Solution:
        a = q0
        b = 0
        c = 3(q1 - q0)
        d = -2(q1 - q0)
        """
        t2 = t * t
        t3 = t2 * t
        
        delta = q1 - q0
        
        a = q0
        b = torch.zeros_like(q0)
        c = 3 * delta
        d = -2 * delta
        
        # Position q_t
        q_t = a + c * t2 + d * t3
        # Velocity v_t
        v_t = 2 * c * t + 3 * d * t2
        # Acceleration a_t (Target Force)
        a_t = 2 * c + 6 * d * t
        
        return q_t, v_t, a_t
    
    @torch.no_grad()
    def step(self, x_fused_step, hidden_states, sample_steps=5):
        # 适用于实机环境的高频单步自回归展开
        B = x_fused_step.shape[0]
        device = x_fused_step.device
        dtype = x_fused_step.dtype
        
        x_t = self.in_proj(x_fused_step)
        
        # --- Mamba 状态更新  ---
        new_states = []
        current_hidden = x_t
        current_residual = None
        for i, blk in enumerate(self.blocks):
            conv_st, ssm_st = hidden_states[i]
            if current_residual is None: current_residual = current_hidden
            else: current_residual = current_residual + current_hidden
            hidden_ln = blk.norm(current_residual.to(dtype=blk.norm.weight.dtype))
            
            if hasattr(blk.mixer, "step"):
                y_t, new_conv_st, new_ssm_st = blk.mixer.step(hidden_ln.unsqueeze(1), conv_st, ssm_st)
                y_t = y_t.squeeze(1)
            else:
                y_t = blk.mixer(hidden_ln.unsqueeze(1)).squeeze(1)
                new_conv_st, new_ssm_st = conv_st, ssm_st
                
            new_states.append((new_conv_st, new_ssm_st))
            current_hidden = y_t
            
            if blk.mlp is not None:
                current_residual = current_residual + current_hidden
                r2 = blk.norm2(current_residual.to(dtype=blk.norm2.weight.dtype))
                current_hidden = blk.mlp(r2)
                
        mamba_cond_step_updated = current_hidden # [B, 1024]
        
        # alpha = 0.9
        # if not hasattr(self, '_current_z') or self._current_z is None:
        #     self._current_z = torch.randn(B, self.action_dim, self.future_steps, device=device)
        # else:
        #     self._current_z = alpha * self._current_z + math.sqrt(1 - alpha**2) * torch.randn_like(self._current_z)
            
        # q_curr = self.imle_generator(mamba_cond_step_updated, self._current_z, x_fused_step)

        # [推理创新]: 二阶潜动力学 (Second-Order Latent Dynamics / AR(2))
        # 物理隐喻：给意图 Z 赋予“质量”，使用旋转变换严格保持 N(0,I) 分布
        # =====================================================================
        
        # 超参数定义 (可作为类属性 self.latent_theta 等)
        theta = 0.1  # 意图位置更新速度 (类似 dt，越大意图改变越快)
        alpha = 0.9  # 意图速度的阻尼 (类似摩擦力，0.9 表示速度惯性很大)

        # --- 鲁棒型二阶潜动力学 (从 hidden_states 尾部安全提取) ---
        # ---  跨步自回归 AR(2) 潜动力学演化 ---
        z_pos, z_vel = hidden_states[-1]
        
        # 弹性自适应保护（防止 Batch Size 突变或推断初期未对齐）
        if (z_pos.shape[0] != B or z_pos.device != device or z_pos.dtype != dtype):
            z_pos = torch.randn(B, self.action_dim, self.future_steps, device=device, dtype=dtype)
            z_vel = torch.randn(B, self.action_dim, self.future_steps, device=device, dtype=dtype)

        # 驱动速度演化 (带有惯性阻尼的一阶马尔可夫链)
        noise = torch.randn_like(z_vel)
        z_vel_new = alpha * z_vel + math.sqrt(1 - alpha**2) * noise
        
        # 驱动位置演化 (正交旋转积分)
        raw_new_pos = math.cos(theta) * z_pos + math.sin(theta) * z_vel_new
        
        # 【方差纠偏】：消除 AR 长期迭代带来的流形协方差漂移，维持标准的各向同性高斯流形
        z_pos_new = (raw_new_pos - raw_new_pos.mean(dim=(1, 2), keepdim=True)) / (raw_new_pos.std(dim=(1, 2), keepdim=True) + 1e-6)

        # 将演化后的新状态追加到末尾，由 MambaController 自动带入下一控制步
        new_states.append((z_pos_new, z_vel_new))

        # 喂给生成器，输出带有平滑多模态特征的粗动作起点
        q_curr = self.imle_generator(mamba_cond_step_updated, z_pos_new, x_fused_step)


        v_curr = torch.zeros_like(q_curr)
        
        dt = 1.0 / sample_steps
        for i in range(sample_steps):
            t_input = torch.full((B,), i / sample_steps, device=device)
            acc = self.sb_head(q_curr.unsqueeze(1), v_curr.unsqueeze(1), t_input, x_fused_step.unsqueeze(1)).squeeze(1)
            v_curr = v_curr + acc * dt
            q_curr = q_curr + v_curr * dt
                        
        actions = q_curr.view(B, self.future_steps, self.action_dim)
        return actions, new_states

    def process_vision_chunk(self, dino_feats):
        """
        辅助函数：处理 Vision Encoder 出来的特征块。
        dino_feats: [N_chunk, 1024, H, W] (DINO output)
        returns: [N_chunk, embed_dim]
        """
        return dino_feats

    def forward_features(self, mamba_in_seq):
            """
            接收的已经是融合完毕的 [B, L, 1024]
            """
            x = self.in_proj(mamba_in_seq)
            
            residual = None
            for blk in self.blocks:
                x, residual = blk(x, residual)
                
            return x # 返回 Mamba 凝练的历史意图 (mamba_cond)
 
    
    def compute_loss(self, x_fused_seq, gt_actions):
        """
        [EE 16维 极简物理求解版]
        x_fused_seq: [B, L, 1024] (包含视觉512与本体512的绝对瞬时物理特征)
        gt_actions: [B, L, 16, 16] (未来16步的末端位姿目标)
        """
        B, L, n_future, D_act = gt_actions.shape 
        device = gt_actions.device
        BL = B * L
        
        # 1. 提取 Mamba 宏观意图与瞬时状态
        mamba_cond = self.forward_features(x_fused_seq) # [B, L, 1024]
        
        mamba_cond_flat = mamba_cond.reshape(BL, -1)
        x_fused_flat = x_fused_seq.reshape(BL, -1)
        gt_actions_flat = gt_actions.reshape(BL, n_future, D_act)
        
        # =======================================================
        # 阶段 A：IMLE 多模态生成与挑选
        # =======================================================
        with torch.no_grad():
            z_samples = torch.randn(BL, self.num_imle_samples, D_act, n_future, device=device)
            mamba_exp = mamba_cond_flat.unsqueeze(1).expand(BL, self.num_imle_samples, -1).reshape(BL * self.num_imle_samples, -1)
            xfused_exp = x_fused_flat.unsqueeze(1).expand(BL, self.num_imle_samples, -1).reshape(BL * self.num_imle_samples, -1)
            z_flat = z_samples.reshape(BL * self.num_imle_samples, D_act, n_future)
            
            # 传入 xfused_exp 作为第三个参数！
            gen_actions = self.imle_generator(mamba_exp, z_flat, xfused_exp)
            gen_actions = gen_actions.view(BL, self.num_imle_samples, n_future, D_act)
            
            gt_act_expanded = gt_actions_flat.unsqueeze(1).expand_as(gen_actions)
            
            # 完全依赖 EE 动作距离挑选最佳模态
            distances = F.mse_loss(gen_actions, gt_act_expanded, reduction='none').mean(dim=(2, 3))
            min_dist, min_idx = torch.min(distances, dim=1) 
            
        best_z = z_samples[torch.arange(BL), min_idx] 
        
        # 传入 x_fused_flat 作为第三个参数！
        q_0_flat = self.imle_generator(mamba_cond_flat, best_z, x_fused_flat)
        
        # 移除了所有拒绝域，强行逼近极限精度
        loss_imle_act = F.mse_loss(q_0_flat, gt_actions_flat.reshape(BL, -1), reduction='none').mean(dim=-1)

        # =======================================================
        # 阶段 B：薛定谔桥物理纠偏 (以 x_fused 为绝对锚点)
        # =======================================================
        q_0 = q_0_flat.detach()
        q_1 = gt_actions_flat.reshape(BL, -1)
        t = torch.rand(BL, device=device)
        t_v = t.unsqueeze(-1)
        q_target, v_target, a_target = self.cubic_spline(t_v, q_0, q_1)
        
        sigma_peak = 0.02
        sigma_t = 16.0 * sigma_peak * ((t_v * (1.0 - t_v)) ** 2)
        sigma_dot_t = 16.0 * sigma_peak * (2.0 * t_v * (1.0 - t_v) * (1.0 - 2.0 * t_v))
        eps = torch.randn_like(q_target)
        q_noisy = q_target + sigma_t * eps
        v_noisy = v_target + sigma_dot_t * eps
        
        # 仅接收 1024 维的当前物理绝对空间特征 x_fused
        a_pred_3d = self.sb_head(q_noisy.unsqueeze(1), v_noisy.unsqueeze(1), t, x_fused_flat.unsqueeze(1))
        a_pred = a_pred_3d.squeeze(1)
        
        k_p, k_d = 4.0, 4.0
        force_target = a_target + k_p * (q_target - q_noisy) + k_d * (v_target - v_noisy)
        loss_force = F.mse_loss(a_pred, force_target, reduction='none').mean(dim=-1)

        total_loss_flat = 1.0 * loss_imle_act + 0.1 * loss_force
        return total_loss_flat.view(B, L)
    
    
    @torch.no_grad()
    def sample_actions(self, x_fused_seq, steps=5):
        B, L, _ = x_fused_seq.shape
        device = x_fused_seq.device
        
        mamba_cond = self.forward_features(x_fused_seq).reshape(B*L, -1)
        x_fused_flat = x_fused_seq.reshape(B*L, -1)

        z_global = torch.randn(B, 1, self.action_dim, self.future_steps, device=device)
        z_sample = z_global.expand(B, L, self.action_dim, self.future_steps).reshape(B*L, self.action_dim, self.future_steps)
        
        q_curr = self.imle_generator(mamba_cond, z_sample, x_fused_flat)
        q_curr = q_curr.view(B, L, -1)
        v_curr = torch.zeros_like(q_curr)
        
        # 辛流积分：在当前物理锚点的约束下丝滑下落
        x_fused_seq_exp = x_fused_flat.view(B, L, -1) 
        
        dt = 1.0 / steps
        for i in range(steps):
            t_input = torch.full((B,), i / steps, device=device)
            acc = self.sb_head(q_curr, v_curr, t_input, x_fused_seq_exp)
            v_curr = v_curr + acc * dt
            q_curr = q_curr + v_curr * dt
            
        return q_curr.view(B, L, self.future_steps, self.action_dim)
