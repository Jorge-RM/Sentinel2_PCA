import glob
import os
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import scipy.linalg as la
from pandas import DataFrame
from PIL import Image, ImageOps


class PCA:
    """PCA computing

    Args:
        in_folder (str): Folder with input images.
        pc_folder (str): Folder where PCs will be stored.
        rgb_folder (str): Folder where false RGB images will be stored.
        bands (int): Number of bands.
        excel_name (str): Name of Excel File.
    """

    def __init__(self, in_folder, bands):
        self.n_bands = bands

        self.image_list = []

        # Get images
        self.image_names = os.listdir(in_folder)
        self.image_names = natsort.natsorted(self.image_names)  # Sort images

        for image_name in self.image_names:
            img = cv2.imread(in_folder + "/" + image_name, 0)
            resize_img = img
            self.image_list.append(resize_img)

        # Save original shape
        self.img_shape = self.image_list[0].shape

        self.matrix = np.zeros((self.image_list[0].size, self.n_bands))
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
        self.eigvals, self.eigvecs = la.eig(cov_matrix)
        order = self.eigvals.argsort()[::-1]  # Descending order
        self.eigvals = self.eigvals[order]
        self.eigvecs = self.eigvecs[:, order]

    def calculate_pca(self, pc_folder, rgb_folder):
        """PCA calculation.

        Args:
            pc_folder (str): Path to save PCA images.
            rgb_folder (str): Path to save false RGB images.

        Return:
            pc_list (list): List os computed Principal Components.

        """

        pc_img = np.zeros((self.img_shape[0], self.img_shape[1]))
        # PCs computing
        self.pcs = np.matmul(self.matrix, self.eigvecs)

        for i, pc in enumerate(np.transpose(self.pcs)):
            img_path = pc_folder + "/" + self.image_names[i]
            # Resize PC from 1-Dimension to 2-Dimension with original images shape
            resized_pc = pc.reshape(-1, self.img_shape[1])
            # Normalize data from 0 to 255
            pc_img = cv2.normalize(resized_pc, pc_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(img_path, pc_img)

            # Get the 3 bands with greater weighting for each PC
            best_bands = np.transpose(self.eigvecs)[i].argsort()[::-1][:3]
            # Create a false RGB image with them
            self.save_rgb(self.image_names[i], rgb_folder, best_bands)

        return self.pcs

    def write_pcs(self, excel_name):
        """Write sorted eigenvectors in Excel file and get false RGB images."""
        sheet = "PCs"
        sheet_2 = "Sorted_PCs"
        data_init = [0, 0]
        writer = pd.ExcelWriter(excel_name, engine="xlsxwriter")
        wb = writer.book
        writer.sheets = {sheet: wb.add_worksheet(sheet)}
        ws_pc = writer.sheets[sheet]

        # Colors
        top_blue = "#92CDDC"
        top_green = "#C4D79B"
        mid_blue = "#B7DEE8"
        mid_green = "#D8E4BC"
        bot_blue = "#D5EEF3"
        bot_green = "#EBF1DE"

        # Format for titles at Sorted_PCs sheet
        merge_format = wb.add_format({"bold": 1, "border": 1, "align": "center"})

        # WRITE EIGENVECTORS
        col_df = ["Comp. " + str(i) for i in range(self.n_bands)]
        row_df = ["Band " + str(i) for i in range(self.n_bands)]
        vecs = np.round(np.array(self.eigvecs), 4)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=row_df)
        eigvecs_df.to_excel(writer, sheet_name=sheet)

        # Fit columns to longest cell text.
        max_length = max(
            [
                max([len(str(s)) for s in eigvecs_df[col].values] + [len(str(col))])
                for col in eigvecs_df.columns
            ]
        )
        ws_pc.set_column(0, self.n_bands + 1, max_length + 2)

        # WRITE SORTED EIGENVECTORS
        sorted_vecs = np.transpose(vecs)
        index = list(range(self.n_bands))

        for i, vec in enumerate(sorted_vecs):
            sorted_df = DataFrame(data=vec, columns=["Eigenvectors"], index=index)
            sorted_df = sorted_df.sort_values(by=["Eigenvectors"], ascending=False)
            sorted_df = sorted_df.reset_index()
            sorted_df = sorted_df.rename(columns={"index": "Bands"})

            sorted_df.to_excel(writer, sheet_name=sheet_2, startrow=1, startcol=i * 2, index=False)
            ws_sort = writer.sheets[sheet_2]
            ws_sort.merge_range(0, i * 2, 0, i * 2 + 1, "PC " + str(i), merge_format)

            # Find the maximum length of the column names and values
            max_length = max(
                [
                    max([len(str(s)) for s in sorted_df[col].values] + [len(str(col))])
                    for col in sorted_df.columns
                ]
            )
            ws_sort.set_column(i * 2, i * 2 + 1, max_length)

            # If the cell is not blank, then the format is applied
            if i % 2:
                cell_format = wb.add_format({"bg_color": top_blue})
                ws_sort.conditional_format(
                    0, i * 2, 0, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )

                cell_format = wb.add_format({"bg_color": mid_blue})
                ws_sort.conditional_format(
                    1, i * 2, 1, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )

                cell_format = wb.add_format({"bg_color": bot_blue})
                ws_sort.conditional_format(
                    2, i * 2, 14, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )
            else:
                cell_format = wb.add_format({"bg_color": top_green})
                ws_sort.conditional_format(
                    0, i * 2, 0, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )

                cell_format = wb.add_format({"bg_color": mid_green})
                ws_sort.conditional_format(
                    1, i * 2, 1, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )

                cell_format = wb.add_format({"bg_color": bot_green})
                ws_sort.conditional_format(
                    2, i * 2, 14, i * 2 + 1, {"type": "no_blanks", "format": cell_format}
                )

        writer.save()

    def show_pca_contributions(self):
        """Plot the contribution of each PC"""
        sum_eigvals = sum(self.eigvals)
        eigvals = self.eigvals * 100 / sum_eigvals
        cum_eigvals = np.cumsum(eigvals)
        plt.bar(range(len(eigvals)), eigvals)
        plt.step(range(len(cum_eigvals)), cum_eigvals)
        plt.savefig("PCA_Contributions.png")

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
            img = np.zeros([self.img_shape[0], self.img_shape[1], 3], dtype=np.uint8)
            img[:, :, 0] = self.image_list[bands[0]]
            img[:, :, 1] = self.image_list[bands[1]]
            img[:, :, 2] = self.image_list[bands[2]]
            cv2.imwrite(pc_folder + "/" + name, img)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute PCA.")
    parser.add_argument("-i", "--input", type=str, help="Folder with images to be analysed.")
    parser.add_argument(
        "-o", "--pcOut", type=str, help="Output folder where PCA images will be stored."
    )
    parser.add_argument(
        "-rgb",
        "--rgbOut",
        type=str,
        help="Output folder where false RGB images composed with"
        "the 3 highest weighted images of each Principal Component.",
    )
    parser.add_argument("-b", "--bands", type=int, help="Number of bands.")
    parser.add_argument("-e", "--excel", type=str, help="Excel file name.")

    if len(sys.argv) == 11:
        args = parser.parse_args()
        pc_folder = args.pcout
        rgb_folder = args.rgbout
        in_folder = args.input
        bands = args.bands
        excel_name = args.excel
    elif len(sys.argv) == 2:
        args = parser.parse_args()
    else:
        pc_folder = "PCS/Etna_fire"
        rgb_folder = "PCS_RGB/Etna_fire"
        in_folder = "crop/Etna_fire"
        bands = 13
        excel_name = "Etna_fire.xlsx"

    pca = PCA(in_folder, bands)
    pca.calculate_pca(pc_folder, rgb_folder)
    pca.show_pca_contributions()
    pca.write_pcs(excel_name)
