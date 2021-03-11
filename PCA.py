import glob
import os
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
from pandas import DataFrame
from PIL import Image, ImageOps


class PCA:
    def __init__(self, input_folder, out_folder, bands, truncate, std, excel_name):
        self.n_bands = bands

        self.image_list = []
        self.truncate = truncate
        self.excel_name = excel_name
        # Get images
        images = os.listdir(input_folder)
        images = sorted([int(os.path.splitext(img)[0]) for img in images])
        images = [str(i) + ".png" for i in images]

        for image_name in images:
            img = cv2.imread(input_folder + "/" + image_name, 0)
            resize_img = img
            self.image_list.append(resize_img)

        self.img_shape = self.image_list[0].shape

        self.matrix = self.flat_dimension(standarization=std)

        cov_matrix = np.cov(self.matrix.transpose())
        self.eigvals, self.eigvecs = la.eig(cov_matrix)
        order = self.eigvals.argsort()[::-1]  # Descending order
        self.eigvals = self.eigvals[order]
        self.eigvecs = self.eigvecs[:, order]

        self.pca = self.calculate_pca(out_folder, truncate)

    def flat_dimension(self, standarization=1):
        """2d to 1d and optional data standarization."""
        matrix = np.zeros((self.image_list[0].size, self.n_bands))
        for i, element in enumerate(self.image_list):
            element = element.flatten()
            if standarization:
                element = (element - element.mean()) / element.std()
            matrix[:, i] = element
        return matrix

    def calculate_pca(self, out_folder, truncate):
        """PCA calculation.

        Args:
            out_folder (str): Output folder for PCA images.
            truncate (int): The number of bands considered at image representation.

        """
        deleted_bands = []
        pc_list = []

        pc_img = np.zeros((self.img_shape[0], self.img_shape[1]))

        for i, vec in enumerate(np.transpose(self.eigvecs)):
            if truncate < self.n_bands:
                vec_min_loc = vec.argsort()[:-truncate]
                truncated_vec = [v if j not in vec_min_loc else 0 for j, v in enumerate(vec)]
                pc = np.matmul(self.matrix, np.transpose(truncated_vec))
            else:
                pc = np.matmul(self.matrix, np.transpose(vec))

            pc_list.append(pc)
            resized_pc = pc.reshape(-1, self.img_shape[1])
            pc_img = cv2.normalize(resized_pc, pc_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            if self.show:
                cv2.namedWindow("PCA " + str(i), cv2.WINDOW_NORMAL)
                cv2.imshow("PCA " + str(i), pc_img)
                cv2.waitKey(0)

            cv2.imwrite(out_folder + "/" + str(i) + ".png", pc_img)

        return pc_list

    def write_pcs(self):
        """Write sorted eigenvectors in Excel file."""
        excel_name = self.excel_name
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

        merge_format = wb.add_format({"bold": 1, "border": 1, "align": "center"})

        col_df = ["Comp. " + str(i + 1) for i in range(self.n_bands)]
        row_df = ["Band " + str(i + 1) for i in range(self.n_bands)]
        vecs = np.round(np.array(self.eigvecs), 4)
        eigvecs_df = DataFrame(data=vecs, columns=col_df, index=row_df)
        eigvecs_df.to_excel(writer, sheet_name=sheet)
        max_length = max(
            [
                max([len(str(s)) for s in eigvecs_df[col].values] + [len(str(col))])
                for col in eigvecs_df.columns
            ]
        )
        ws_pc.set_column(0, self.n_bands + 1, max_length + 2)

        # WRITE SORTED VECTORS
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
        # Plot the contribution of each PC
        sum_eigvals = sum(self.eigvals)
        eigvals = self.eigvals * 100 / sum_eigvals
        cum_eigvals = np.cumsum(eigvals)
        plt.bar(range(len(eigvals)), eigvals)
        plt.step(range(len(cum_eigvals)), cum_eigvals)

        plt.show()

    def crop_input_images(self, in_folder, out_folder):
        """Delete the white area surrounding input images."""
        images = os.listdir(in_folder)

        images = sorted([int(os.path.splitext(img)[0]) for img in images])
        images = [str(i) + ".png" for i in images]
        save_images = 1
        if save_images:
            for i, image_name in enumerate(images):

                img = Image.open(in_folder + "/" + image_name)

                crop_img = ImageOps.grayscale(img.crop((90, 96, 649, 542)))
                crop_img.save(out_folder + str(i) + ".png")

    def save_false_rgb(self, out_folder):
        i = np.zeros([self.img_shape[0], self.img_shape[1], 3], dtype=np.uint8)
        i[:, :, 0] = self.image_list[10]
        i[:, :, 1] = self.image_list[11]
        i[:, :, 2] = self.image_list[12]
        cv2.imshow("0 1 2", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/0 1 2.png", i)

        i[:, :, 0] = self.image_list[10]
        i[:, :, 1] = self.image_list[12]
        i[:, :, 2] = self.image_list[11]
        cv2.imshow("0 2 1", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/0 2 1.png", i)

        i[:, :, 0] = self.image_list[11]
        i[:, :, 1] = self.image_list[10]
        i[:, :, 2] = self.image_list[12]
        cv2.imshow("1 0 2", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/1 0 2.png", i)

        i[:, :, 0] = self.image_list[12]
        i[:, :, 1] = self.image_list[10]
        i[:, :, 2] = self.image_list[11]
        cv2.imshow("2 0 1", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/2 0 1..png", i)

        i[:, :, 0] = self.image_list[11]
        i[:, :, 1] = self.image_list[12]
        i[:, :, 2] = self.image_list[10]
        cv2.imshow("1 2 0", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/1 2 0.png", i)

        i[:, :, 0] = self.image_list[12]
        i[:, :, 1] = self.image_list[11]
        i[:, :, 2] = self.image_list[10]
        cv2.imshow("2 1 0", i)
        cv2.imwrite(out_folder + "False color (10, 11, 12)/2 1 0.png", i)

        cv2.waitKey(0)
        cv2.destroyAllWindows_pc()

    def save_rgb(self, name, out_folder):
        i = np.zeros([self.img_shape[0], self.img_shape[1], 3], dtype=np.uint8)
        i[:, :, 0] = self.image_list[1]
        i[:, :, 1] = self.image_list[2]
        i[:, :, 2] = self.image_list[3]
        # cv2.imshow("RGB", i)
        cv2.imwrite(out_folder + name, i)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Compare groundtruth annotation bounding boxes " "with algorithm detections."
    )
    parser.add_argument("input", type=str, help="Folder with images to be analysed.")
    parser.add_argument("output", type=str, help="Output folder where PCA images will be saved.")
    parser.add_argument("bands", type=int, help="Number of bands.")
    parser.add_argument("truncate", type=int, help="Number of images applied on PCA visualization")
    parser.add_argument("std", type=int, help="1 or 0, enable or disable.")
    parser.add_argument("excel", type=str, help="Excel file name.")

    if len(sys.argv) == 6:
        args = parser.parse_args()
        out_folder = args.output
        in_folder = args.input
        bands = args.bands
        truncate = args.truncate
        std = args.std

        if std != 1 or std != 0:
            raise Exception('"std" should be 1 or 0')

        excel_name = args.excel
    else:
        out_folder = "Saved/pc_gc/"
        in_folder = "gc/"
        bands = 13
        truncate = bands
        std = 1
        excel_name = "pc_gc.xlsx"

    pca = PCA(in_folder, out_folder, bands, truncate, std, excel_name)
    # pca.show_pca_contributions()
    pca.write_pcs()
    # pca.save_rgb("RGB.png", "")
