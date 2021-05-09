import glob
import os
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns
from pandas import DataFrame
from PIL import Image, ImageOps

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler


class OwnPCA:
    """PCA computing

    Args:
        in_folder (str): Folder with input images.
        bands (int): Number of bands.
    """

    def __init__(self, in_folder, bands):
        self.n_bands = bands

        self.image_list = []
        self.in_folder = in_folder

        # Get images
        self.image_names = os.listdir(in_folder)
        self.image_names = natsort.natsorted(self.image_names)  # Sort images

        for image_name in self.image_names:
            img = cv2.imread(in_folder + "/" + image_name, cv2.IMREAD_ANYDEPTH)
            self.image_list.append(img)

        # Save original shape
        self.img_shape = self.image_list[0].shape

        self.matrix = np.zeros(
            (self.image_list[0].size, self.n_bands), dtype=np.float32
        )
        # Data mean
        m_mean = np.array(self.image_list).mean()
        # Data standard deviation
        m_std = np.array(self.image_list).std()

        for i, element in enumerate(self.image_list):
            # 2D to 1D
            element = element.flatten()
            # Image standardization
            element = (element - m_mean) / m_std
            self.matrix[:, i] = element

        cov_matrix = np.cov(self.matrix.transpose())
        self.eigvals, self.eigvecs = np.linalg.eig(cov_matrix)
        order = self.eigvals.argsort()[::-1]  # Descending order
        self.eigvals = self.eigvals[order]
        self.eigvecs = self.eigvecs[:, order]

        for i, eigvec in enumerate(self.eigvecs.transpose()):
            p = np.max(eigvec)
            if np.max(eigvec) < 0.1:
                self.eigvecs[:, i] = -eigvec

        # pca_out = PCA().fit(self.matrix)
        # self.eigvals = pca_out.explained_variance_
        # self.eigvecs = pca_out.components_

        print(
            "Principal Components Computing finished for images at <"
            + in_folder
            + ">"
        )

    def get_projection(self, pc_folder, rgb_folder):
        """Input images are projected over eigenvectors (principal component images).

        Args:
            pc_folder (str): Path to save PCA images.
            rgb_folder (str): Path to save false RGB images.
        """

        pc_img = np.zeros(
            (self.img_shape[0], self.img_shape[1]), dtype=np.float32
        )
        # PCs computing
        self.pcs = np.matmul(self.matrix, self.eigvecs)

        for i, pc in enumerate(np.transpose(self.pcs)):
            img_path = pc_folder + "/" + self.image_names[i]
            # Resize PC from 1-Dimension to 2-Dimension with original images shape
            resized_pc = pc.reshape(-1, self.img_shape[1])
            # Normalize data from 0 to 1
            pc_img = cv2.normalize(
                resized_pc, pc_img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            cv2.imwrite(img_path, pc_img)

            # Get the 3 bands with greater weighting for each PC
            best_bands = abs(np.transpose(self.eigvecs)[i]).argsort()[::-1][:3]
            # Create a false RGB image with them
            self.save_rgb(self.image_names[i], rgb_folder, best_bands)

        print("Save false RGB images at <" + rgb_folder + ">")

    def get_variance(self, path, title):
        """Plot eigenvectors on a heatmap.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        var_set = [np.var(a) for a in self.matrix.transpose()]
        fig, ax = plt.subplots()
        ax.set(xlabel="Bandas", ylabel="Varianza")
        ax.grid()
        plt.xticks(range(1, self.n_bands + 1))
        ax.plot(
            range(1, self.n_bands + 1),
            var_set,
            "o",
            linestyle="dotted",
        )
        ax.tick_params(direction="out", length=10)
        fig.savefig(path + "/" + title + "_var.png")
        plt.close()
        print("Save variance graph at <" + path + ">")

    def get_contributions(self):
        """Plot the contribution of each PC."""
        cum_eigvals = np.cumsum(self.eigvals * 100 / sum(self.eigvals))
        plt.step(range(len(cum_eigvals)), cum_eigvals)
        name = os.path.basename(self.in_folder)
        plt.savefig(name + ".png")
        print("Save Principal Components weight graph at <" + name + ">")

    def get_heatmap(self, path, title):
        """Plot eigenvectors on a heatmap.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        col_df = ["PC" + str(i + 1) for i in range(self.n_bands)]
        row_df = ["B" + str(i + 1) for i in range(self.n_bands)]
        vecs = np.round(np.array(self.eigvecs), 2)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=row_df)
        ax = sns.heatmap(eigvecs_df, annot=True, cmap="Spectral")
        ax.set_title(title)
        figure = ax.get_figure()
        figure.set_size_inches(9, 6)
        figure.savefig(path + "/" + title + ".png")
        figure.clf()
        print("Save Principal Components heatmap at <" + path + ">")

    def save_rgb(self, name, pc_folder, bands):
        """Create an RGB image with 3 selected bands.

        Args:
            name (str): Name of image.
            pc_folder (str): Output folder.
            bands (list): list of 3 bands to create an image with 3 channels (B, G, R)
        """
        if len(bands) != 3:
            print("Error: you should choose only 3 bands [B, G, R]")
        else:
            # Creates a matrix with dimensions of original images
            img = np.zeros(
                [self.img_shape[0], self.img_shape[1], 3], dtype=np.uint8
            )
            for b, band in enumerate(bands):
                img[:, :, b] = self.image_list[band] * 255

            cv2.imwrite(pc_folder + "/" + name, img)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute PCA.")
    parser.add_argument(
        "-i", "--input", type=str, help="Folder with images to be analysed."
    )
    parser.add_argument(
        "-o",
        "--pcOut",
        type=str,
        help="Output folder where PCA images will be stored.",
    )
    parser.add_argument(
        "-rgb",
        "--rgbOut",
        type=str,
        help="Output folder where false RGB images composed with"
        "the 3 highest weighted images of each Principal Component.",
    )
    parser.add_argument("-b", "--bands", type=int, help="Number of bands.")

    if len(sys.argv) == 11:
        args = parser.parse_args()
        pc_folder = args.pcout
        rgb_folder = args.rgbout
        in_folder = args.input
        bands = args.bands
        pca = OwnPCA(in_folder, bands)
        pca.get_projection(pc_folder, rgb_folder)
        pca.get_contributions()

    else:
        print("Invalid number of arguments")
