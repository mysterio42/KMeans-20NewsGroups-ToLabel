import argparse

import numpy as np

from utils.data import get_data, get_data_sources
from utils.model import LabelMe
from utils.plot import plot_pca, plot_tsne


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, choices=get_data_sources().keys(),
                        help='Data Source filenames'
                        )
    parser.add_argument('--clusters', type=int,
                        help='How many Label we want')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1)

    args = parse_args()

    data = get_data(get_data_sources()[args.fname])

    labelme = LabelMe(data, args.clusters)
    embeds = labelme.embed()

    plot_pca(embeds.toarray())
    plot_tsne(embeds.toarray())

    labelme.train(embeds)
    labelme.clusterize(args.fname)

    plot_pca(embeds.toarray(), labelme.model.labels_)
    plot_tsne(embeds.toarray(), labelme.model.labels_)
