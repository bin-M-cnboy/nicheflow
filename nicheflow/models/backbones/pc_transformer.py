import torch
from jaxtyping import Float
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.LeakyReLU()

    def forward(
        self, x: Float[Tensor, "... {self.input_dim}"]
    ) -> Float[Tensor, "... {self.input_dim}"]:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(input_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Float[Tensor, "B N_cells embed_dim"],
        mask: Float[Tensor, "B N_cells n_cells"] | None = None,
    ) -> Float[Tensor, "B N_cells embed_dim"]:
        key_padding_mask: Tensor | None = None
        if mask is not None:
            # The mask contains 1s for cells to be considered and 0s for cells
            # to be ignored. MultiHeadAttention expects True for tokens to be
            # ignored, and False for tokens to be kept. Therefore, we invert the mask.
            key_padding_mask = ~mask.bool()

        # Self attention
        attn_output, _ = self.attn.forward(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_output
        x = self.ln_1(x)

        # Feedforward
        x = self.ff(x)
        x = self.ln_2(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Feed forward
        self.ff = FeedForward(input_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.ln_3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Float[Tensor, "B N_cells_dec embed_dim"],
        enc_output: Float[Tensor, "B N_cells_enc embed_dim"],
        self_mask: Float[Tensor, "B N_cells_dec"] | None = None,
        cross_mask: Float[Tensor, "B N_cells_enc"] | None = None,
    ) -> Float[Tensor, "B N_cells_dec embed_dim"]:
        # The mask contains 1s for cells to be considered and 0s for cells
        # to be ignored. MultiHeadAttention expects True for tokens to be
        # ignored, and False for tokens to be kept. Therefore, we invert the mask.
        self_key_padding_mask = None
        if self_mask is not None:
            self_key_padding_mask = ~self_mask.bool()

        cross_key_padding_mask = None
        if cross_mask is not None:
            cross_key_padding_mask = ~cross_mask.bool()

        # Self attention + res. connection + layer norm
        attn_output, _ = self.self_attn(
            x, x, x, key_padding_mask=self_key_padding_mask, need_weights=False
        )
        x = x + attn_output
        x = self.ln_1(x)

        # Cross attention + res. connection + layer norm
        cross_attn_output, _ = self.cross_attn(
            x,
            enc_output,
            enc_output,
            key_padding_mask=cross_key_padding_mask,
            need_weights=False,
        )
        x = x + cross_attn_output
        x = self.ln_2(x)

        # Feed forward
        x = self.ff(x)
        x = self.ln_3(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, out_dim: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.out_dim = out_dim
        self.out_linear = nn.Linear(2 * time_emb_dim, out_dim)

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B {self.out_dim}"]:
        if t.ndim == 1:
            t = t.unsqueeze(1)

        freqs = torch.arange(self.time_emb_dim, device=t.device, dtype=t.dtype)
        freqs = freqs.unsqueeze(0)
        t_freq = t * freqs
        # (B, 2 * D)
        t_emb = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        return self.out_linear(t_emb)


class PointCloudTransformer(nn.Module):
    """
    PointCloudTransformer 用于处理空间转录组数据的点云 Transformer 主干网络。
    
    它采用了典型的 Encoder-Decoder 架构：
    - Encoder: 接收条件时间点(t_0)的微环境点云(细胞表达式+坐标+时间=0)，利用 Self-Attention 提取源微环境的上下文特征。
    - Decoder: 接收目标时间点(t_1)的带噪点云(细胞表达式+坐标+目标时间 t)，并通过 Cross-Attention 获取 Encoder 的历史特征，
               最终输出用于流匹配(Flow Matching)计算的预测向量(预测细胞表达式和坐标的导数/流向量)。
    """
    def __init__(
        self,
        pca_dim: int,
        ohe_dim: int,
        coord_dim: int,
        output_dim: int,
        embed_dim: int = 128,
        ff_hidden_dim: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pca_dim = pca_dim
        self.ohe_dim = ohe_dim

        # Recalculate the embedding dimension
        embed_dim = (3 * num_heads) * (embed_dim // (3 * num_heads))
        # Split it equally for gene expressions, time and positions
        concat_dim = embed_dim // 3

        self.x_emb = nn.Linear(pca_dim + ohe_dim, concat_dim)
        self.pos_emb = nn.Linear(coord_dim, concat_dim)
        self.time_emb = TimeEmbedding(embed_dim, concat_dim)

        self.enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.out_linear = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        # 条件阶段 (t_0 源微环境) 的输入
        pcs_cond: Float[Tensor, "B N_points_t1 D_in"],      # 条件细胞基因表达 PCA 成分
        pos_cond: Float[Tensor, "B N_points_t1 D_coord"],   # 条件细胞的空间坐标
        ohe_cond: Float[Tensor, "B N_points_t1 D_ohe"],     # 条件时间点的独热编码 (One-Hot)
        # 目标阶段 (当前采样的状态 x_t) 的输入
        pcs_target: Float[Tensor, "B N_points_t2 D_in"],    # 目标细胞基因表达 PCA 成分 (带噪)
        pos_target: Float[Tensor, "B N_points_t2 D_coord"], # 目标细胞的空间坐标 (带噪)
        ohe_target: Float[Tensor, "B N_points_t1 D_ohe"],   # 目标时间点的独热编码
        t_target: Float[Tensor, "B ..."],                   # 连续流的当前时间步 t (用于流匹配中的ODE演化)
        # 掩码 (用于处理不同大小的微环境点云 batch 填充)
        mask_condition: Float[Tensor, "B N_points_t1"] | None = None,
        mask_target: Float[Tensor, "B N_points_t2"] | None = None,
    ) -> tuple[Float[Tensor, "B N_points_t2 D_in"], Float[Tensor, "B N_points_t2 D_coord"]]:
        # 1. 拼接细胞表达和时间相关的 One-Hot 编码特征
        pcs_cond = torch.cat([pcs_cond, ohe_cond], dim=-1)
        pcs_target = torch.cat([pcs_target, ohe_target], dim=-1)

        # 2. 条件微环境 (源) 特征嵌入
        x_cond = self.x_emb(pcs_cond)
        pos_cond = self.pos_emb(pos_cond)
        # 条件点云通常认为时间 t=0，使用全0向量作为占位
        t_cond = torch.zeros_like(x_cond, device=x_cond.device, dtype=x_cond.dtype)

        # 3. Encoder 计算 (源微环境上下文特征提取)
        # 沿着特征维度拼接表达、时间和空间坐标的 Embedding (三者维度各占 1/3，正好等于 embed_dim)
        enc_output = torch.cat([x_cond, t_cond, pos_cond], dim=-1)
        for block in self.enc_blocks:
            enc_output = block(x=enc_output, mask=mask_condition)

        # 4. 目标微环境 (当前中间态) 特征嵌入
        x_target = self.x_emb(pcs_target)
        pos_target = self.pos_emb(pos_target)
        # 提取连续时间步的傅里叶编码
        t_target = self.time_emb(t_target)[:, None, :].expand(-1, x_target.size(1), -1)

        # 5. Decoder 计算 (融合源微环境条件指导目标态预测)
        dec_output = torch.cat([x_target, t_target, pos_target], dim=-1)
        for block in self.dec_blocks:
            dec_output = block(
                x=dec_output,
                enc_output=enc_output, # Cross Attention 指向 Encoder 输出
                self_mask=mask_target,
                cross_mask=mask_condition,
            )

        # 6. 输出映射：还原回 表达维度(pca_dim) + 坐标维度(coord_dim)
        out = self.out_linear(dec_output)

        x_pred = out[:, :, : self.pca_dim]
        pos_pred = out[:, :, self.pca_dim :]
        return x_pred, pos_pred
