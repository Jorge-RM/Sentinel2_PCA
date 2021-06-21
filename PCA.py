import os
import time

import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r (%r, %r) %2.5f sec" % (method.__name__, args, kw, te - ts))
        return result

    return timed


class OwnPCA:
    """PCA computing

    Args:
        in_folder (str): Folder with input images.
        bands (int): Number of bands.
    """

    @timeit
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

        # 2D to 1D
        for i, element in enumerate(self.image_list):
            element = element.flatten()
            self.matrix[:, i] = element

        self.cov_mat = np.cov(self.matrix.transpose())
        # self.correlation = np.corrcoef(self.matrix.transpose())
        self.eigvals, self.eigvecs = np.linalg.eig(self.cov_mat)
        # self.eigvals, self.eigvecs = np.linalg.eig(self.correlation)
        order = self.eigvals.argsort()[::-1]  # Descending order
        self.eigvals = self.eigvals[order]
        self.eigvecs = self.eigvecs[:, order]
        for i, eigvec in enumerate(self.eigvecs.transpose()):
            p = np.max(eigvec)
            if np.max(eigvec) < 0.1:
                self.eigvecs[:, i] = -eigvec

        # PCs computing
        self.pcs = np.matmul(self.matrix, self.eigvecs)

        print(
            "Principal Components Computing finished for images at <"
            + in_folder
            + ">"
        )

    @timeit
    def get_projection(self, pc_folder):
        """Input images are projected over eigenvectors (principal component images).

        Args:
            pc_folder (str): Path to save PCA images.
        """
        pc_img = np.zeros(
            (self.img_shape[0], self.img_shape[1]), dtype=np.float32
        )
        for i, pc in enumerate(np.transpose(self.pcs)):
            img_path = pc_folder + "/" + self.image_names[i]
            # Resize PC from 1-Dimension to 2-Dimension with original images shape
            resized_pc = pc.reshape(-1, self.img_shape[1])
            # Normalize data from 0 to 1
            pc_img = cv2.normalize(
                resized_pc, pc_img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            cv2.imwrite(img_path, pc_img)
        print("Save PC images at <" + img_path + ">")

    @timeit
    def get_variance(self, path, title):
        """Get variance of input images.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        self.variances = [np.var(a) for a in self.image_list]
        label = ["B" + str(i) for i in range(1, self.n_bands + 1)]
        fig, ax = plt.subplots()
        ax.grid()
        plt.xticks(range(self.n_bands), label, fontsize=13)
        plt.yticks(fontsize=13)
        ax.set_xlabel("Bands", fontsize=13)
        ax.set_ylabel("Variance", fontsize=13)
        ax.plot(
            range(self.n_bands),
            self.variances,
            "o",
            linestyle="dotted",
            markersize=9,
        )
        ax.set_title("Variance of " + title + " images", fontsize=15)
        ax.tick_params(direction="out", length=10)
        fig.savefig(path + "/" + title + "_var.png", bbox_inches="tight")
        plt.close()
        print("Save variance graph at <" + path + ">")

    @timeit
    def get_excel(self, excel_name):
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
        merge_format = wb.add_format(
            {"bold": 1, "border": 1, "align": "center"}
        )

        # WRITE EIGENVECTORS
        col_df = ["Comp. " + str(i) for i in range(1, self.n_bands + 1)]
        row_df = ["Band " + str(i) for i in range(1, self.n_bands + 1)]
        vecs = np.round(np.array(self.eigvecs), 3)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=row_df)
        eigvecs_df.to_excel(writer, sheet_name=sheet)

        # Fit columns to longest cell text.
        max_length = max(
            [
                max(
                    [len(str(s)) for s in eigvecs_df[col].values]
                    + [len(str(col))]
                )
                for col in eigvecs_df.columns
            ]
        )
        ws_pc.set_column(0, self.n_bands + 1, max_length + 2)

        # WRITE SORTED EIGENVECTORS
        sorted_vecs = np.transpose(vecs)
        index = list(range(self.n_bands))

        for i, vec in enumerate(sorted_vecs):
            sorted_df = DataFrame(
                data=vec, columns=["Eigenvectors"], index=index
            )
            sorted_df = sorted_df.sort_values(
                by=["Eigenvectors"], ascending=False
            )
            sorted_df = sorted_df.reset_index()
            sorted_df = sorted_df.rename(columns={"index": "Bands"})

            sorted_df.to_excel(
                writer,
                sheet_name=sheet_2,
                startrow=1,
                startcol=i * 2,
                index=False,
            )
            ws_sort = writer.sheets[sheet_2]
            ws_sort.merge_range(
                0, i * 2, 0, i * 2 + 1, "PC " + str(i), merge_format
            )

            # Find the maximum length of the column names and values
            max_length = max(
                [
                    max(
                        [len(str(s)) for s in sorted_df[col].values]
                        + [len(str(col))]
                    )
                    for col in sorted_df.columns
                ]
            )
            ws_sort.set_column(i * 2, i * 2 + 1, max_length)

            # If the cell is not blank, then the format is applied
            if i % 2:
                cell_format = wb.add_format({"bg_color": top_blue})
                ws_sort.conditional_format(
                    0,
                    i * 2,
                    0,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )

                cell_format = wb.add_format({"bg_color": mid_blue})
                ws_sort.conditional_format(
                    1,
                    i * 2,
                    1,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )

                cell_format = wb.add_format({"bg_color": bot_blue})
                ws_sort.conditional_format(
                    2,
                    i * 2,
                    14,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )
            else:
                cell_format = wb.add_format({"bg_color": top_green})
                ws_sort.conditional_format(
                    0,
                    i * 2,
                    0,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )

                cell_format = wb.add_format({"bg_color": mid_green})
                ws_sort.conditional_format(
                    1,
                    i * 2,
                    1,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )

                cell_format = wb.add_format({"bg_color": bot_green})
                ws_sort.conditional_format(
                    2,
                    i * 2,
                    14,
                    i * 2 + 1,
                    {"type": "no_blanks", "format": cell_format},
                )

        writer.save()

    @timeit
    def get_eigvecs(self, path, title):
        """Plot eigenvectors on a heatmap.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        row_df = ["B" + str(i) for i in range(1, self.n_bands + 1)]
        col_df = ["PC" + str(i + 1) for i in range(self.n_bands)]
        vecs = np.round(np.array(self.eigvecs), 2)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=row_df)
        plt.subplots(figsize=(9, 6))
        ax = sns.heatmap(
            eigvecs_df, annot=True, cmap="bwr", linewidths=0.5, vmin=-1, vmax=1
        )
        ax.set_title("Eigenvectors of " + title + " images")
        plt.savefig(
            path + "/" + title + "_eigvecs.png", bbox_inches="tight", dpi=100
        )
        plt.close()
        print("Save Principal Components heatmap at <" + path + ">")

    def get_best_eigvecs(self, path, title):
        """Plot eigenvectors on a heatmap.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        label = ["B" + str(i) for i in range(1, self.n_bands + 1)]
        vals = []
        eigvals_weighting = self.eigvals * 100 / sum(self.eigvals)
        # 98% Criteria
        for n, v in enumerate(eigvals_weighting):
            vals.append(v)
            if sum(vals) >= 98:
                break

        n_vals = len(vals)
        col_df = ["PC" + str(i + 1) for i in range(n_vals)]
        vecs = np.round(np.array(self.eigvecs[:, :n_vals]), 2)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=label)
        plt.subplots(figsize=(6, 9))
        ax = sns.heatmap(
            eigvecs_df,
            annot=True,
            cmap="bwr",
            vmin=-1,
            vmax=1,
            annot_kws={"fontsize": 15},
        )
        ax.collections[0].colorbar.ax.tick_params(labelsize=15)
        ax.set_title("MWE of " + title + " images", fontsize=14)
        ax.tick_params(labelsize=15)
        plt.savefig(
            path + "/" + title + "_weigvecs.png", bbox_inches="tight", dpi=100
        )
        plt.close()
        print("Save Principal Components heatmap at <" + path + ">")

    @timeit
    def get_correlation(self, path, title):
        """Plot eigenvectors on a heatmap.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        label = ["B" + str(i) for i in range(1, self.n_bands + 1)]
        data = DataFrame(data=self.matrix)
        self.correlations = data.corr()
        self.correlations = np.round(np.array(self.correlations), 2)
        mask = np.triu(np.ones_like(self.correlations, dtype=bool))
        sheet = "Correlation Matrix"
        data_init = [0, 0]
        plt.subplots(figsize=(9, 6))
        ax = sns.heatmap(
            self.correlations,
            xticklabels=label,
            yticklabels=label,
            mask=mask,
            annot=True,
            cmap="viridis_r",
            vmin=-1,
            vmax=1,
        )
        ax.collections[0].colorbar.ax.tick_params(labelsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        ax.set_title("Correlation matrix of " + title + " images", fontsize=15)
        plt.savefig(
            path + "/" + title + "_corr.png", bbox_inches="tight", dpi=100
        )
        plt.close()
        print("Save Correlation matrix heatmap at <" + path + ">")

    def get_rgb(self, pc_folder, name, rgb_vec):
        """Create an RGB image.

        Args:
            pc_folder (str): Output folder.
            name (str): Name of image.
            rgb_vec (list): vector of bands
        """
        # Creates a matrix with dimensions of original images
        img = np.ones(
            [self.img_shape[0], self.img_shape[1], 3], dtype=np.float32
        )
        finalImg = np.zeros(
            [self.img_shape[0], self.img_shape[1], 3], dtype=np.uint8
        )
        cv2.merge(
            [
                self.image_list[rgb_vec[0]],
                self.image_list[rgb_vec[1]],
                self.image_list[rgb_vec[2]],
            ],
            img,
        )
        cv2.normalize(img, finalImg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.resize(
            finalImg, (300, int(finalImg.shape[0] * 300 / finalImg.shape[1]))
        )
        cv2.imwrite(pc_folder + "/" + name + "_rgb.jpg", finalImg)
        print("RGB image saved at <" + pc_folder + ">")


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
    parser.add_argument("-b", "--bands", type=int, help="Number of bands.")

    if len(sys.argv) == 6:
        args = parser.parse_args()
        pc_folder = args.pcout
        in_folder = args.input
        bands = args.bands
        pca = OwnPCA(in_folder, bands)
        pca.get_projection(pc_folder)

    else:
        print("Invalid number of arguments")
