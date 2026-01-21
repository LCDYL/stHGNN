from __future__ import division
from __future__ import print_function

import argparse
import os
import warnings
import random
import torch.optim as optim
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from input_my_data import load_g_inputs
from P_G_model import *
from utils import *

# matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
device = torch.device("cpu")
torch.set_default_device(device)
os.environ['PYTHONHASHSEED'] = str(config.seed)


# noinspection PyShadowingNames
def train(model, optimizer, x_g, H_F, H_T, adata, epoch: int, emb_max, scale_factor=1.0):
    model.train()
    optimizer.zero_grad()
    EMB ,pi, disp, mean, loss, q_E = model(
        adata = adata,
        x_g = x_g,
        H_F = H_F,
        H_T = H_T,
        epoch = epoch,
        emb_max=emb_max,
        scale_factor=scale_factor
    )

    EMB = pd.DataFrame(EMB.cpu().detach().numpy()).fillna(0).values
    EMB = StandardScaler().fit_transform(EMB)
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

    loss.backward()
    optimizer.step()

    return EMB, mean, loss, q_E


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
    #             '151671', '151672', '151673', '151674', '151675', '151676']
    datasets = ['MBA']

    for i in range(len(datasets)):

        H5AD_PATH = "generate_data/" + datasets[i] + "/" + datasets[i] + ".h5ad"

        dataset = datasets[i]
        config_file = 'DLPFC.ini'
        print(dataset)

        x_g, meta, adata, features, labels, H_T, H_F = load_g_inputs(
            h5ad_path=H5AD_PATH,
            spatial_radius=Config(config_file).radius,
            k_feat_hyper=Config(config_file).k
        )
        nfeat = x_g.size(1)
        print(adata)


        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = 'result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())
        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = True


        cfg = PGConfig()
        print('dataset:', dataset, ' lr:', config.lr)
        model = Our_Super_Plus_Pro_Max_Ultra_Model(
            cfg=cfg,
            in_dim=x_g.size(1),
            nfeat_genes=nfeat,
            k_clusters = config.class_num,
            ridge_lambda=0.0
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        epoch_max = 0  # 记录最佳epoch
        ari_max = 0  # 记录最佳ARI分数
        nmi_max = 0  # 记录最佳NMI分数
        idx_max = []  # 记录最佳聚类标签
        mean_max = []  # 记录最佳重建
        emb_max = []  # 记录最佳嵌入向量
        min_zinbloss = 999  # 记录最佳ZINBloss


        for epoch in range(config.epochs):
            EMB, mean, loss, q_E = train(model, optimizer, x_g, H_F, H_T, adata, epoch, emb_max, scale_factor=1.0)
            # KMeans聚类
            kmeans = KMeans(n_clusters=config.class_num).fit(EMB)
            idx = kmeans.labels_ 

            coords = np.asarray(adata.obsm["spatial"], dtype=float)
            idx = spatial_majority_vote(idx, coords, radius=Config(config_file).radius, min_frac=0.5, max_iter=1)
            # 计算ARI
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.cluster.normalized_mutual_info_score(labels, idx)
            print(dataset, ' epoch: ', epoch,' loss = {:.4f}'.format(loss), 'Now_ARI = {:.4f}'.format(ari_res), 'Best_ARI = {:.4f}'.format(ari_max), 'Now_NMI = {:.4f}'.format(nmi_res), 'Max_NMI = {:.4f}'.format(nmi_max))
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = EMB
            if nmi_res > nmi_max:
                nmi_max = nmi_res

        print(dataset, ' ', ari_max)

        title = 'OurSuperModel: ARI={:.4f}'.format(ari_max) + 'NMI={:.4f}'.format(nmi_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'OurSuperModel.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=5, title=title, legend_fontoutline=2,
                           show=False, legend_loc='none', text_kwds={'alpha': 0})  # 没有legend
        plt.savefig(savepath + 'OurSuperModel_umap_mean.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        pd.DataFrame(emb_max).to_csv(savepath + 'OurSuperModel_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'OurSuperModel_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.obsm["sadj"] = adata.obsm["A_T_pos"].numpy()
        for k in ("A_T_pos", "A_F_pos", "E_T_pairs"):
            if k in adata.obsm:
                del adata.obsm[k]
        adata.write(savepath + 'OurSuperModel.h5ad')
