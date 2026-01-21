import anndata as ad
from P_G_model import *


def load_g_inputs(
        h5ad_path: str,
        spatial_radius: int = 550,
        k_feat_hyper: int = 14,
    ):
    adata = ad.read_h5ad(h5ad_path, backed=None)
    

    # 1) 特征矩阵 x_g
    X = adata.X  # numpy.ndarray 或 scipy.sparse
    # 防出错，统一转 torch.float32 的 dense
    if "scipy" in str(type(X)):
        X = X.toarray()
    x_g = torch.tensor(np.asarray(X), dtype=torch.float32)  # [Ng, Fg]
    Ng = int(x_g.size(0))


    # 2) 超图结构 edge_index
    # === 空间半径超边 H_T ===
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)  # 坐标，(Ng, 2)
    nbrs = NearestNeighbors(radius=spatial_radius, metric="euclidean", n_jobs=1).fit(coords)
    indices = nbrs.radius_neighbors(coords, return_distance=False)
    rows_T, cols_T = [], []
    for e in range(Ng):
        members = indices[e]
        if members is None or len(members) == 0:
            members = np.array([e], dtype=np.int64)
        else:
            members = np.unique(np.append(members.astype(np.int64), e))
        rows_T.extend(members.tolist())
        cols_T.extend([e] * len(members))
    H_T = torch.tensor([rows_T, cols_T], dtype=torch.long)  # [2, E_T]

    # === 特征 cosine-KNN 超边 H_F ===
    x_norm = F.normalize(x_g, p=2, dim=1)  # [Ng, Fg]
    sim = torch.matmul(x_norm, x_norm.t())  # [Ng, Ng]
    topk_idx = torch.topk(sim, k=min(k_feat_hyper + 1, Ng), dim=1).indices.cpu().numpy()
    rows_F, cols_F = [], []
    for e in range(Ng):
        members = np.unique(topk_idx[e].astype(np.int64))
        rows_F.extend(members.tolist())
        cols_F.extend([e] * len(members))
    H_F = torch.tensor([rows_F, cols_F], dtype=torch.long)  # [2, E_F]


    # 3) 附带信息
    meta = {
        "obs_names": np.asarray(adata.obs_names),  # 条形码顺序
        "var_names": np.asarray(adata.var_names),  # 基因名顺序（已是 HVG 列）
        "spatial": np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }

    # 4) 评价
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']

    N = x_g.size(0)
    A_T_pos, _ = incidence_to_adj(torch.LongTensor(H_T), N)
    A_F_pos, _ = incidence_to_adj(torch.LongTensor(H_F), N)
    adata.obsm["A_T_pos"] = A_T_pos
    adata.obsm["A_F_pos"] = A_F_pos

    return x_g, meta, adata, features, labels, H_T, H_F



if __name__ == '__main__':
    # 测试数据加载函数
    x_g, meta, adata, features, labels, H_T, H_F = load_g_inputs(
        h5ad_path="../generate_data/DLPFC/151507/151507.h5ad",
        spatial_radius = 550,
        k_feat_hyper = 14
    )
    print(x_g.shape)
    print(meta)
    print(adata)
    print(features)
    print(labels)
    print(H_T)

    print(H_F)
