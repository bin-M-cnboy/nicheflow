import torch
from jaxtyping import Float
from torch import Tensor, nn

from nicheflow.models.backbones.pc_transformer import TimeEmbedding


class SinglePointMLP(nn.Module):
    """A feedforward MLP for modeling single-cell trajectories
    用于单细胞轨迹建模的前馈多层感知机 (MLP)
    """

    def __init__(
        self,
        pca_dim: int = 50,
        coord_dim: int = 2,
        ohe_dim: int = 3,
        time_emb_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 52,
        use_layer_norm: bool = True,
    ) -> None:
        """
            pca_dim: 基因表达的 PCA 特征维度
            coord_dim: 空间坐标特征维度 (x, y)
            ohe_dim: 类别 (如时间步) 的 One-hot 编码维度
            time_emb_dim: 连续时间 t 的特征嵌入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征的总维度
            use_layer_norm: 是否使用 LayerNorm 正则化
        """
        super().__init__()
        self.pca_dim = pca_dim
        self.ohe_dim = ohe_dim

        # Embedding layers (特征嵌入层：分别处理基因表达、空间坐标和类别标签)
        self.emb_x = nn.Linear(pca_dim, hidden_dim)
        self.emb_coord = nn.Linear(coord_dim, hidden_dim)
        self.emb_ohe = nn.Linear(ohe_dim, hidden_dim)

        # 时间t的编码网络
        self.time_embedding = TimeEmbedding(time_emb_dim=time_emb_dim, out_dim=hidden_dim)

        # [cond + target] + time 
        # 拼接后的总维度 = 2批次(条件特征 + 目标特征) * 3类特征(基因 + 坐标 + 标签) * hidden_dim + 时间特征(hidden_dim)
        concat_dim = 2 * (hidden_dim * 3) + hidden_dim

        # 构建主干多层感知机网络
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(concat_dim))
        layers += [
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, output_dim),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_cond: Float[Tensor, "B N D_pca"],
        pos_cond: Float[Tensor, "B N D_coord"],
        ohe_cond: Float[Tensor, "B N D_ohe"],
        x_target: Float[Tensor, "B N D_pca"],
        pos_target: Float[Tensor, "B N D_coord"],
        ohe_target: Float[Tensor, "B N D_ohe"],
        t: Float[Tensor, "B"],
    ) -> tuple[Float[Tensor, "B N D_pca"], Float[Tensor, "B N D_coord"]]:
        """
        网络的前向传播逻辑。基于前一时刻状态 (cond)、当前假定目标状态 (target) 以及时间步 (t)，
        预测状态的演化速度向量或下一时刻特征。
        """
        # Condition embeddings (条件状态特征嵌入)
        x_emb_cond = self.emb_x(x_cond)
        coord_emb_cond = self.emb_coord(pos_cond)
        ohe_emb_cond = self.emb_ohe(ohe_cond)

        # Target embeddings (目标状态特征嵌入)
        x_emb_target = self.emb_x(x_target)
        coord_emb_target = self.emb_coord(pos_target)
        ohe_emb_target = self.emb_ohe(ohe_target)

        # Time embedding (连续时间t特征提取，并扩展到所有 N 个单细胞节点维度以方便求余计算)
        t_emb = self.time_embedding(t)
        t_emb = t_emb[:, None, :].expand(-1, x_emb_cond.size(1), -1)

        # Concatenate all embeddings (在特征维度 dim=-1 拼接以上所有嵌入)
        z = torch.cat(
            [
                x_emb_cond,
                coord_emb_cond,
                ohe_emb_cond,
                x_emb_target,
                coord_emb_target,
                ohe_emb_target,
                t_emb,
            ],
            dim=-1,
        )

        # 通过 MLP 网络得到预测输出
        out = self.mlp(z)
        # 将输出分离成对应 PCA 的预测部分和对应的空间坐标预测部分
        return out[..., : self.pca_dim], out[..., self.pca_dim :]
