"""
MoE-Nanjin 模型
Post-Norm Transformer

1. ShortTermExpert: GRU（短期依赖专家）
2. LongTermExpert: Transformer（长期依赖专家，Post-Norm架构）
3. StatisticExpert: MLP（统计特征专家，仅处理目标变量统计特征）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShortTermExpert(nn.Module):
    """短期依赖专家：使用GRU架构处理序列依赖"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 输出层
        if output_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim)
            )
        else:
            self.output_proj = None
    
    def forward(self, x):
        """x: [B, T, F] -> output: [B, output_dim] 或 [B, hidden_dim]"""
        gru_out, hidden = self.gru(x)  # gru_out: [B, T, hidden_dim]
        last_hidden = gru_out[:, -1, :]  # [B, hidden_dim]
        
        if self.output_proj:
            output = self.output_proj(last_hidden)
        else:
            output = last_hidden
        return output


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, d_model = x.shape
        
        Q = self.w_q(x)  # [B, T, d_model]
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 分割为多头
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T, d_k]
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, n_heads, T, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)  # [B, n_heads, T, d_k]
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, T, d_model)  # [B, T, d_model]
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """FFN层"""
    def __init__(self, d_model, d_ff, dropout=0.2):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层（Post-Norm架构）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super().__init__()
        # Post-Norm: 归一化在子层之后
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Post-Norm: attention -> norm -> dropout -> residual
        attn_out = self.self_attn(x, mask)
        x = x + self.dropout1(self.norm1(attn_out))
        
        # Post-Norm: ffn -> norm -> dropout -> residual
        ffn_out = self.ffn(x)
        x = x + self.dropout2(self.norm2(ffn_out))
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码（可学习）"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        
    def forward(self, x):
        T = x.size(1)
        if T <= self.pe.size(1):
            pos = self.pe[:, :T, :]
        else:
            n_repeat = (T + self.pe.size(1) - 1) // self.pe.size(1)
            pos = self.pe.repeat(1, n_repeat, 1)[:, :T, :]
        return x + pos


class LongTermExpert(nn.Module):
    """长期依赖专家：使用Transformer架构（Post-Norm）"""
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.2, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # 输入dropout
        self.input_dropout = nn.Dropout(dropout)
        
        # Transformer编码器层（Post-Norm架构）
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # 输出层
        if output_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, output_dim)
            )
        else:
            self.output_proj = None
    
    def forward(self, x):
        """x: [B, T, F] -> output: [B, output_dim] 或 [B, d_model]"""
        # 输入投影
        x = self.input_proj(x)  # [B, T, d_model]
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 输入dropout
        x = self.input_dropout(x)
        
        # Transformer编码（Post-Norm架构）
        for layer in self.layers:
            x = layer(x)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 时间聚合（平均池化）
        x = x.mean(dim=1)  # [B, d_model]
        
        # 输出投影
        if self.output_proj:
            output = self.output_proj(x)
        else:
            output = x
        
        return output


class StatisticExpert(nn.Module):
    """残差/分布专家（Residual Expert）：MLP 处理目标变量分布统计特征"""
    def __init__(self, input_dim, n_targets=20, hidden_dim=128, output_dim=None, dropout=0.2,
                 enhanced_stats=True):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.enhanced_stats = enhanced_stats
        
        # mean, std, max (+ skew, kurt, Q25, Q75 when enhanced)
        stats_per_target = 7 if enhanced_stats else 3
        target_stats_dim = n_targets * stats_per_target
        self.target_stats_proj = nn.Sequential(
            nn.Linear(target_stats_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        if output_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim)
            )
        else:
            self.output_proj = None
    
    def forward(self, x):
        """
        Args:
            x: [B, T, F] 输入序列，F = n_targets（仅目标变量，无外生变量）
        Returns:
            output: [B, output_dim] 或 [B, hidden_dim]
        """
        # 目标变量统计特征
        target_features = x  # [B, T, n_targets]
        
        target_mean = target_features.mean(dim=1)  # [B, n_targets]
        target_std = target_features.std(dim=1).clamp_min(1e-6)   # [B, n_targets]
        target_max = target_features.max(dim=1)[0]  # [B, n_targets]
        
        if self.enhanced_stats:
            centered = target_features - target_mean.unsqueeze(1)
            skew = (centered ** 3).mean(dim=1) / (target_std ** 3 + 1e-8)
            kurt = (centered ** 4).mean(dim=1) / (target_std ** 4 + 1e-8)
            q25 = torch.quantile(target_features, 0.25, dim=1)
            q75 = torch.quantile(target_features, 0.75, dim=1)
            target_stats = torch.cat(
                [target_mean, target_std, target_max, skew, kurt, q25, q75], dim=1
            )
        else:
            target_stats = torch.cat([target_mean, target_std, target_max], dim=1)
        
        target_encoded = self.target_stats_proj(target_stats)  # [B, hidden_dim]
        
        # 输出投影
        if self.output_proj:
            output = self.output_proj(target_encoded)
        else:
            output = target_encoded
        
        return output


class GatingNetwork(nn.Module):
    """场景感知门控网络：Flatten(历史序列) + 日历场景向量 c_t"""
    def __init__(self, input_dim, lookback, num_experts, hidden_dim=128, dropout=0.2, scene_dim=0):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.lookback = lookback
        self.scene_dim = scene_dim
        
        gate_in_dim = lookback * input_dim + scene_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 门控决策层
        self.gating = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
    
    def forward(self, x, scene_c=None):
        """x: [B, T, D], scene_c: [B, scene_dim] -> weights, logits"""
        B, T, D = x.shape
        flat_input = x.reshape(B, -1)
        if self.scene_dim > 0:
            if scene_c is None:
                scene_c = torch.zeros(B, self.scene_dim, device=x.device, dtype=x.dtype)
            flat_input = torch.cat([flat_input, scene_c], dim=-1)
        
        features = self.feature_extractor(flat_input)
        logits = self.gating(features)  # [B, num_experts]
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        return weights, logits


class MoENanjin(nn.Module):
    """MoE-Nanjin模型"""
    def __init__(
        self,
        input_dim,
        output_dim,
        lookback,
        num_experts=3,
        hidden_dim=128,
        expert_config=None,
        dropout=0.2,
        ablation_mode='baseline',
        use_scene_gating=True,
        enhanced_statistic=True,
        scene_dim=8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ablation_mode = ablation_mode
        self.use_scene_gating = use_scene_gating
        self.enhanced_statistic = enhanced_statistic
        self.scene_dim = scene_dim if use_scene_gating else 0
        
        # 默认专家配置
        if expert_config is None:
            expert_config = {
                'short_term': {'hidden_dim': hidden_dim, 'num_layers': 3},
                'long_term': {'d_model': hidden_dim, 'n_heads': 8, 'n_layers': 4, 'd_ff': hidden_dim * 4},
                'statistic': {'hidden_dim': hidden_dim}
            }
        
        # 根据ablation_mode决定创建哪些专家
        self.experts = nn.ModuleList()
        expert_types = []  # 记录专家类型，用于调试
        
        if ablation_mode == 'baseline':
            # 基线：1×GRU + 1×Transformer + 1×MLP = 3个专家
            self.experts.append(ShortTermExpert(input_dim, output_dim=output_dim, **expert_config['short_term'], dropout=dropout))
            expert_types.append('ShortTerm')
            self.experts.append(LongTermExpert(input_dim, output_dim=output_dim, **expert_config['long_term'], dropout=dropout))
            expert_types.append('LongTerm')
            self.experts.append(StatisticExpert(
                input_dim, n_targets=input_dim, output_dim=output_dim,
                enhanced_stats=enhanced_statistic, **expert_config['statistic'], dropout=dropout
            ))
            expert_types.append('Residual')
        elif ablation_mode == 'no_statistic':
            # 移除StatisticExpert：1×GRU + 1×Transformer = 2个专家
            self.experts.append(ShortTermExpert(input_dim, output_dim=output_dim, **expert_config['short_term'], dropout=dropout))
            expert_types.append('ShortTerm')
            self.experts.append(LongTermExpert(input_dim, output_dim=output_dim, **expert_config['long_term'], dropout=dropout))
            expert_types.append('LongTerm')
        elif ablation_mode == 'no_longterm':
            # 移除LongTermExpert：1×GRU + 1×MLP = 2个专家
            self.experts.append(ShortTermExpert(input_dim, output_dim=output_dim, **expert_config['short_term'], dropout=dropout))
            expert_types.append('ShortTerm')
            self.experts.append(StatisticExpert(
                input_dim, n_targets=input_dim, output_dim=output_dim,
                enhanced_stats=enhanced_statistic, **expert_config['statistic'], dropout=dropout
            ))
            expert_types.append('Residual')
        elif ablation_mode == 'no_shortterm':
            # 移除ShortTermExpert：1×Transformer + 1×MLP = 2个专家
            self.experts.append(LongTermExpert(input_dim, output_dim=output_dim, **expert_config['long_term'], dropout=dropout))
            expert_types.append('LongTerm')
            self.experts.append(StatisticExpert(
                input_dim, n_targets=input_dim, output_dim=output_dim,
                enhanced_stats=enhanced_statistic, **expert_config['statistic'], dropout=dropout
            ))
            expert_types.append('Residual')
        else:
            raise ValueError(f"未知的ablation_mode: {ablation_mode}. 支持的模式: 'baseline', 'no_statistic', 'no_longterm', 'no_shortterm'")
        
        # 门控网络：专家数量根据ablation_mode动态调整
        self.num_experts = len(self.experts)
        self.gating = GatingNetwork(
            input_dim, lookback, self.num_experts, hidden_dim, dropout,
            scene_dim=self.scene_dim
        )
        
        # 打印专家配置信息
        print(f"\n[模型配置] 消融模式: {ablation_mode}")
        print(f"  专家数量: {self.num_experts}")
        print(f"  专家类型: {expert_types}")
    
    def forward(self, x, scene_c=None):
        """
        Args:
            x: [B, T, F] 输入序列
            scene_c: [B, scene_dim] 日历场景向量（可选）
        """
        gate_weights, gate_logits = self.gating(x, scene_c)
        
        # 各专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [B, output_dim]
            expert_outputs.append(expert_out)
        
        # 加权组合
        output = torch.zeros_like(expert_outputs[0])  # [B, output_dim]
        for i, expert_out in enumerate(expert_outputs):
            output += gate_weights[:, i:i+1] * expert_out
        
        # Residual connection
        if x.shape[2] == output.shape[1]:
            output = output + x[:, -1, :]
        
        return output, expert_outputs, gate_weights, gate_logits

