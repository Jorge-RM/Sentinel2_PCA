import os

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd

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

        self.folders = [
            os.path.join(container, folder)
            for folder in self.folders
            if os.path.isdir(os.path.join(container, folder))
        ]
        self.eigvalues_array = np.empty((len(self.folders), self.n_bands))

        for f, folder in enumerate(self.folders):
            if os.path.isdir(folder):
                pca = PCA.PCA(folder, bands)
                eigvals = pca.eigvals * 100 / sum(pca.eigvals)
                self.eigvalues_array[f, :] = np.round(eigvals, 4)
                self.pca_list.append(pca)

    def save_data(self, out_container):
        """Save PCA images, graphs and Excels.

        Args:
            out_container (str): Path of output folder.

        """
        pca_folder = os.path.join(out_container, "PCS")
        rgb_folder = os.path.join(out_container, "PCS_RGB")
        excel_folder = os.path.join(out_container, "Excel")
        if not os.path.isdir(excel_folder):
            os.makedirs(excel_folder)

        for pca, folder in zip(self.pca_list, self.folders):
            folder_name = os.path.basename(folder)
            local_pca_folder = os.path.join(pca_folder, folder_name)
            local_rgb_folder = os.path.join(rgb_folder, folder_name)
            if not os.path.isdir(local_pca_folder):
                os.makedirs(local_pca_folder)
            if not os.path.isdir(local_rgb_folder):
                os.makedirs(local_rgb_folder)

            pca.calculate_pca(local_pca_folder, local_rgb_folder)
            excel_file = os.path.join(excel_folder, folder_name) + ".xlsx"
            pca.write_pcs(excel_file)

        fig, ax = plt.subplots()
        ax.set(
            xlabel="Principal Component",
            ylabel="Weighting (%)",
            title="Principal Components weighting",
        )
        ax.grid()
        plt.ylim([0, 100])
        plt.xticks(range(self.n_bands))
        for ev in self.eigvalues_array:
            ax.plot(range(self.n_bands), ev, "o", linestyle="dotted")
        fig.savefig(
            out_container + "\\" + os.path.basename(out_container) + ".png"
        )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute PCA.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder container of folders with hyperspectral images.",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output folder to save data."
    )
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
