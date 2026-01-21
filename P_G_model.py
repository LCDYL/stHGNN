import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import ZINB
from typing import Union
from torch_geometric.nn import BatchNorm, HypergraphConv, GCNConv
from utils import *
from config import Config as Config
from threadpoolctl import threadpool_limits


# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
device = torch.device("cpu")
torch.set_default_device(device)

config_file = 'DLPFC.ini'
config = Config(config_file)

class PGConfig:
    g_hidden_F_1: int = 256  # G特征 第1层
    g_hidden_F_2: int = 256  # G特征 第2层
    g_hidden_T_1: int = 256  # G空间 第1层
    g_hidden_T_2: int = 256  # G空间 第2层
    out_dim: int = 128
    zinb_hid: int = 64

    dropout: float = 0
    alpha = 0.5
    beta = 0.1
    gamma = 0.001

    reg_on = "spatial"       # "spatial" | "feature" | "both"


class G_HGNN_F(nn.Module):
    """
    输入:
        x_g: [Ng, Fg]
        hyperedge_index: [2, E]
    输出:
        emb1: [Ng, 128]
    """
    def __init__(self, in_dim:int, cfg:PGConfig):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, cfg.g_hidden_F_1)
        self.bn1 = BatchNorm(cfg.g_hidden_F_1)
        self.conv2 = HypergraphConv(cfg.g_hidden_F_1, cfg.g_hidden_F_2)
        self.bn2 = BatchNorm(cfg.g_hidden_F_2)
        self.conv3 = HypergraphConv(cfg.g_hidden_F_2, cfg.out_dim)
        self.bn3 = BatchNorm(cfg.out_dim)
        self.drop = cfg.dropout

    def forward(self, x_g:torch.Tensor, H_F:torch.LongTensor):
        h = self.conv1(x_g, H_F)
        h = self.bn1(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = self.conv2(h, H_F)
        h = self.bn2(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = self.conv3(h, H_F)
        h = self.bn3(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        return h


class G_HGNN_T(nn.Module):
    """
    输入:
        x_g: [Ng, Fg]
        H_T: [2, E]
    输出:
        emb2: [Ng, 128]
    """
    def __init__(self, in_dim:int, cfg:PGConfig):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, cfg.g_hidden_T_1)
        self.bn1 = BatchNorm(cfg.g_hidden_T_1)
        self.conv2 = HypergraphConv(cfg.g_hidden_T_1, cfg.g_hidden_T_2)
        self.bn2 = BatchNorm(cfg.g_hidden_T_2)
        self.conv3 = HypergraphConv(cfg.g_hidden_T_2, cfg.out_dim)
        self.bn3 = BatchNorm(cfg.out_dim)
        self.drop = cfg.dropout

    def forward(self, x_g:torch.Tensor, H_T:torch.LongTensor):
        h = self.conv1(x_g, H_T)
        h = self.bn1(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = self.conv2(h, H_T)
        h = self.bn2(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = self.conv3(h, H_T)
        h = self.bn3(h); h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        return h


class Our_Super_Plus_Pro_Max_Ultra_Model(nn.Module):
    def __init__(
            self,
            cfg:PGConfig,
            in_dim:int,
            nfeat_genes: int,
            k_clusters: int,
            ridge_lambda: float = 0.0):
        super().__init__()
        self.convF = G_HGNN_F(in_dim, cfg)
        self.convT = G_HGNN_T(in_dim, cfg)
        self.drop = cfg.dropout
        self.dec = ZINB_decoder(nfeat=nfeat_genes, nhid1=cfg.zinb_hid, nhid2=cfg.out_dim)
        self.ridge_lambda = ridge_lambda

        # === 动态权重的单模置信度 MLP（Φ_F, Φ_T）===
        self.phi_F = nn.Sequential(nn.Linear(cfg.out_dim, cfg.out_dim//2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(cfg.out_dim//2, 1), nn.Sigmoid())
        self.phi_T = nn.Sequential(nn.Linear(cfg.out_dim, cfg.out_dim//2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(cfg.out_dim//2, 1), nn.Sigmoid())
        # 损失函数组分的权重
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        # self.omiga = cfg.omiga
        self.reg_on = cfg.reg_on

        # === 双向交叉注意力（F <-> T）===
        self.lnF = nn.LayerNorm(cfg.out_dim)
        self.lnT = nn.LayerNorm(cfg.out_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.out_dim, num_heads=4, dropout=cfg.dropout, batch_first=True)
        # 轻量 FFN + 残差规范化（稳定训练）
        self.fuse_ffn = nn.Sequential(
            nn.Linear(cfg.out_dim, 2 * cfg.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.out_dim, cfg.out_dim),
        )
        self.fuse_ln = nn.LayerNorm(cfg.out_dim)

        self.S_beta = nn.Parameter(torch.tensor(0.0))
        self._row_softmax = nn.Softmax(dim=1)

        # ===== DEC =====
        self.k_clusters = k_clusters
        K = int(self.k_clusters)
        d = cfg.out_dim

        self.dec_centroids_T = nn.Parameter(torch.empty(K, d))  # for embT
        self.dec_centroids_F = nn.Parameter(torch.empty(K, d))  # for embF
        self.dec_centroids_E = nn.Parameter(torch.empty(K, d))  # for EMB

        self.register_buffer("_dec_init_T", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_dec_init_F", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_dec_init_E", torch.tensor(0, dtype=torch.long))
        self.w_dec_T = nn.Parameter(torch.tensor(1.0))
        self.w_dec_F = nn.Parameter(torch.tensor(1.0))
        self.w_dec_E = nn.Parameter(torch.tensor(1.0))


    def _dec_loss_single(self, Z: torch.Tensor, C: torch.nn.Parameter,
                         init_flag: torch.Tensor, emb_max, v: float = 1.0):
        device = Z.device
        N, d = Z.shape
        K = C.shape[0]

        with torch.no_grad():
            Z_np = Z.detach().cpu().numpy()
            # , n_init=10, random_state=config.seed, algorithm="lloyd"
            km = KMeans(n_clusters=K)
            with threadpool_limits(limits=1):
                km.fit(Z_np)
            centers = torch.from_numpy(km.cluster_centers_).to(device=device, dtype=Z.dtype)
            C.data.copy_(centers)
            init_flag.fill_(1)

        z2 = (Z * Z).sum(dim=1, keepdim=True)  # [N,1]
        c2 = (C * C).sum(dim=1, keepdim=True).t()  # [1,K]
        dist2 = z2 + c2 - 2.0 * (Z @ C.t())  # [N,K]

        q = (1.0 + dist2 / max(1e-6, v)).pow(-0.5 * (v + 1.0))
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)

        fk = q.sum(dim=0, keepdim=True)  # [1,K]
        p = (q * q) / fk.clamp_min(1e-12)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p = p.detach()

        kl = (p * (p.add(1e-12).log() - q.add(1e-12).log())).sum() / float(N)
        return kl, q

    def forward(
            self,
            adata,
            x_g: torch.Tensor,
            H_F: torch.LongTensor,
            H_T: torch.LongTensor,
            epoch: int,
            emb_max,
            scale_factor: Union[torch.Tensor, float] = 1.0):
        emb_max = torch.Tensor(emb_max)

        # adata.to(device)
        x_g = x_g.to(device)
        H_F = H_F.to(device)
        H_T = H_T.to(device)

        embF = self.convF(x_g, H_F)
        embT = self.convT(x_g, H_T)

        # ===== Cross-Attention 融合 =====
        # F 作为 Query，从 T 中取信息
        qF = self.lnF(embF).unsqueeze(0)  # [1, N, d]
        kT = self.lnT(embT).unsqueeze(0)  # [1, N, d]
        attn_F, _ = self.cross_attn(query=qF, key=kT, value=kT, need_weights=False)
        yF = embF + attn_F.squeeze(0)  # 残差到 F
        # T 作为 Query，从 F 中取信息
        qT = self.lnT(embT).unsqueeze(0)  # [1, N, d]
        kF = self.lnF(embF).unsqueeze(0)  # [1, N, d]
        attn_T, _ = self.cross_attn(query=qT, key=kF, value=kF, need_weights=False)
        yT = embT + attn_T.squeeze(0)  # 残差到 T
        # 双向对齐后的平均
        y = 0.5 * (yF + yT)
        # 轻量 FFN + 残差 LN
        EMB = self.fuse_ln(y + self.fuse_ffn(y))  # [N, d]

        ZL = EMB  # ZL := EMB  (N,d)
        # 归一化自相关矩阵 S
        Sim = ZL @ ZL.t()  # (N,N)
        S = self._row_softmax(Sim)
        ZG = S @ ZL
        EMB = self.S_beta * ZG + ZL


        # 计算reg_loss
        reg_loss = EMB.new_tensor(0.)
        A_T_pos = adata.obsm["A_T_pos"]
        A_F_pos = adata.obsm["A_F_pos"]
        A_T_pos = A_T_pos.to(device, dtype=EMB.dtype, non_blocking=True)
        A_F_pos = A_F_pos.to(device, dtype=EMB.dtype, non_blocking=True)
        if self.reg_on in ("spatial", "both"):
            reg_loss = reg_loss + pair_smooth_loss(EMB, A_T_pos)
        if self.reg_on in ("feature", "both"):
            reg_loss = reg_loss + pair_smooth_loss(EMB, A_F_pos)
        if epoch < 30:
            beta = float(self.beta) * (epoch / 30)
        else:
            beta = float(self.beta)


        # ZINB
        pi, theta, mu = self.dec(EMB)  # [Ng, out_dim] x3
        zinb = ZINB(pi=pi, theta=theta, scale_factor=scale_factor, ridge_lambda=self.ridge_lambda)
        zinb_loss = zinb.loss(y_true=x_g, y_pred=mu, mean=True)

        # DEC（Student-t + KL）
        # embT（空间分支）
        kl_T, q_T = self._dec_loss_single(
            Z=embT,
            C=self.dec_centroids_T,
            init_flag=self._dec_init_T,
            emb_max=emb_max,
            v=1.0
        )
        # embF（特征分支）
        kl_F, q_F = self._dec_loss_single(
            Z=embF,
            C=self.dec_centroids_F,
            init_flag=self._dec_init_F,
            emb_max=emb_max,
            v=1.0
        )
        # EMB（融合/共识分支）
        kl_E, q_E = self._dec_loss_single(
            Z=EMB,
            C=self.dec_centroids_E,
            init_flag=self._dec_init_E,
            emb_max=emb_max,
            v=1.0
        )
        # 组合：带权平均, softplus
        wT = F.softplus(self.w_dec_T)  # log(1+e^x)
        wF = F.softplus(self.w_dec_F)
        wE = F.softplus(self.w_dec_E)

        wsum = (wT + wF + wE).clamp_min(1e-6)
        dec_loss = (wT * kl_T + wF * kl_F + wE * kl_E) / wsum
        loss = self.alpha*zinb_loss + beta*reg_loss + self.gamma*dec_loss

        return EMB, pi, theta, mu, loss, q_E


class ZINB_decoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        """
        ZINB decoder network
        :param nfeat: number of features (genes)
        :param nhid1: number of hidden units in the first layer
        :param nhid2: number of hidden units in the second layer, same as the embedding dimension
        """
        super(ZINB_decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)

        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        """
        Forward pass
        :param emb: embedding matrix
        :return: pi
        :return: dispersion
        :return: mean [*, nfeat]
        """
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


def pair_smooth_loss(Z, A_pos):
    """
    Z: [N, d] on (cuda/cpu)
    A_pos: [N, N] 0/1 dense tensor (same device & dtype as Z is best)
    Return: scalar
    """
    if A_pos is None or A_pos.numel() == 0:
        return Z.new_tensor(0.)
    if A_pos.device != Z.device or A_pos.dtype != Z.dtype:
        A_pos = A_pos.to(device=Z.device, dtype=Z.dtype, non_blocking=True)

    deg = A_pos.sum()
    if deg.item() == 0:
        return Z.new_tensor(0.)

    n2 = (Z * Z).sum(dim=1)                # [N]
    G  = Z @ Z.t()                         # [N,N]

    deg_i = A_pos.sum(dim=1)               # [N]
    term1 = 2.0 * (deg_i * n2).sum()
    term2 = 2.0 * (A_pos * G).sum()
    num = term1 - term2

    return num / deg.clamp_min(1)


def incidence_to_adj(H: torch.LongTensor,
                     num_nodes: int,
                     keep_self_loops: bool = True,
                     binary: bool = True,
                     dtype: torch.dtype = torch.float32):
    """
    超图 incidence -> 节点-节点邻接矩阵 A
    H: [2, E]，(row=node_idx, col=hyperedge_idx)
    num_nodes: 节点个数 N
    keep_self_loops: 是否保留自环
    binary: True 则把权重压成 0/1（只要同超边即为相邻）；False 则用共享超边计数
    """
    N = int(num_nodes)

    if H.numel() == 0:
        A = torch.zeros((N, N), dtype=dtype, device=device)
        if not keep_self_loops:
            A.fill_diagonal_(0)
        A_neg = torch.ones_like(A) - A
        if not keep_self_loops:
            A_neg.fill_diagonal_(0)
        return A, A_neg

    M = int(H[1].max().item()) + 1
    S = torch.zeros((N, M), dtype=dtype, device=device)
    S.index_put_((H[0], H[1]), torch.ones(H.size(1), dtype=dtype, device=device), accumulate=True)
    S.clamp_(0, 1)

    A = S @ S.t()
    if binary:
        A = (A > 0).to(dtype)

    if not keep_self_loops:
        A.fill_diagonal_(0)

    A_neg = (torch.ones_like(A) - A)
    if not keep_self_loops:
        A_neg.fill_diagonal_(0)
    return A, A_neg



