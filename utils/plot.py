from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FIGURES_DIR = 'figures/'
plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_pca(features, labels=None):
    plt.figure()
    proj = PCA(n_components=2).fit_transform(features)

    if labels is not None:
        plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="Paired")
        plt.colorbar()
        plt.savefig(FIGURES_DIR + 'Figure_pca-preds' + '.png')
    else:
        plt.scatter(proj[:, 0], proj[:, 1], cmap='Paired')
        plt.colorbar()
        plt.savefig(FIGURES_DIR + 'Figure_pca-data' + '.png')
    plt.show()


def plot_tsne(features, labels=None):
    ax = Axes3D(plt.figure())

    tsne = TSNE(n_components=3)
    Z = tsne.fit_transform(features)
    if labels is not None:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=labels, cmap='Paired')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title(f'T-SNE on the data')
        plt.savefig(FIGURES_DIR + f'Figure_tsne-preds' + '.png')
    else:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], cmap='Paired')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title(f'T-SNE on the data')
        plt.savefig(FIGURES_DIR + f'Figure_tsne-data' + '.png')

    plt.show()
