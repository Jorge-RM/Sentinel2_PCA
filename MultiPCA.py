import os

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import PCA


class MultiPCA:
    """Compute PCs of a database.

    Args:
        container (str): Folder with a list of folders with hyperspectral images.
        bands (int): Number of bands to analyse.

    """

    def __init__(self, container, bands):

        self.n_bands = bands
        self.pca_list = []
        # List of folders in Main container
        self.folders = os.listdir(container)

        self.folders_path = [
            os.path.join(container, folder)
            for folder in self.folders
            if os.path.isdir(os.path.join(container, folder))
        ]
        self.eigvalues_array = np.empty((len(self.folders_path), self.n_bands))

        for f, folder in enumerate(self.folders_path):
            if os.path.isdir(folder):
                pca = PCA.OwnPCA(folder, bands)
                eigvals = pca.eigvals * 100 / sum(pca.eigvals)
                self.eigvalues_array[f, :] = np.round(eigvals, 4)
                self.pca_list.append(pca)

    def save_data(self, out_container):
        """Save PCA images and graphs.

        Args:
            out_container (str): Path of output folder.

        """
        pca_folder = os.path.join(out_container, "PCS")
        folder_names=[]
        for pca, folder in zip(self.pca_list, self.folders_path):
            folder_name = os.path.basename(folder)
            local_pca_folder = os.path.join(pca_folder, folder_name)
            if not os.path.isdir(local_pca_folder):
                os.makedirs(local_pca_folder)
            folder_names.append(folder_name)
            pca.get_projection(local_pca_folder)
            pca.get_eigvecs(out_container, folder_name)
            pca.get_variance(out_container, folder_name)
            pca.get_correlation(out_container, folder_name)
            pca.get_best_eigvecs(out_container, folder_name)
            excel_file = os.path.join(out_container, folder_name) + ".xlsx"
            pca.get_excel(excel_file)
        self.get_weighting(out_container)

    def get_weighting(self, out_container):
        """Save a graph with weightings of every principal component computed. """
        fig, ax = plt.subplots()
        ax.grid()
        plt.ylim([0, 100])
        plt.xticks(range(1, self.n_bands + 1), fontsize=13)
        plt.yticks(fontsize=13)
        ax.set_xlabel("Principal Component", fontsize=13)
        ax.set_ylabel("Weighting (%)", fontsize=13)
        ax.set_title("Principal Components weighting", fontsize=15)
        for ev, legend_name in zip(self.eigvalues_array, self.folders):
            ax.plot(
                range(1, self.n_bands + 1),
                ev,
                "o",
                linestyle="dotted",
                label=legend_name,
                markersize=9,
            )
        ax.legend()
        fig.savefig(
            out_container + "\\" + os.path.basename(out_container) + ".png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute PCA from multiple folders.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder container of folders with hyperspectral images.",
    )
    parser.add_argument("-o", "--output", type=str, help="Output folder to save data.")
    parser.add_argument("-b", "--bands", type=int, help="Number of bands.")

    if len(sys.argv) == 7:
        args = parser.parse_args()
        out_container = args.output
        in_container = args.input
        bands = args.bands
        if os.path.isdir(in_container):
            pca = MultiPCA(in_container, bands)
            pca.save_data(out_container)
        else:
            print("Missing folder: %s", in_container)
    else:
        print("Invalid number of arguments")
