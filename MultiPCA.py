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
        rgb_folder = os.path.join(out_container, "PCS_RGB")
        folder_names=[]
        for pca, folder in zip(self.pca_list, self.folders_path):
            folder_name = os.path.basename(folder)
            local_pca_folder = os.path.join(pca_folder, folder_name)
            local_rgb_folder = os.path.join(rgb_folder, folder_name)
            if not os.path.isdir(local_pca_folder):
                os.makedirs(local_pca_folder)
            if not os.path.isdir(local_rgb_folder):
                os.makedirs(local_rgb_folder)
            folder_names.append(folder_name)
            pca.get_projection(local_pca_folder, local_rgb_folder)
            pca.get_eigvecs(out_container, folder_name)
            pca.get_variance(out_container, folder_name)
            pca.get_correlation(out_container, folder_name)
            pca.get_best_eigvecs(out_container, folder_name)
            #excel_file = os.path.join(out_container, folder_name) + ".xlsx"
            #pca.get_excel(excel_file)
        mean_vars = [np.mean(pca.variances) for pca in self.pca_list]
        mean_cors = [np.mean(pca.correlations)[1] for pca in self.pca_list]
        self.get_variances(mean_vars, mean_cors, folder_names, out_container)
        self.get_weighting(out_container)

    def get_variances(self, mean_vars, mean_cors, folder_names, out_container):
        """Get variance of input images.

        Args:
            path (str): Output folder.
            title (str): Name of image.
        """
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        w=0.4

        bar1=np.arange(len(folder_names))
        bar2 = [i+w for i in bar1]

        ax.set_ylabel("Variance", fontsize=13)
        ax2.set_ylabel("Correlation", fontsize=13)

        ax.bar(bar1, mean_vars, w, color='b', align='center')
        ax2.bar(bar2, mean_cors, w, color='r', align='center')
        plt.xticks(range(len(folder_names)), folder_names, fontsize=10, rotation='vertical')
        plt.yticks(fontsize=13)

        ax.set_title("Mean variances of " + os.path.basename(out_container) + " images", fontsize=15)
        ax.tick_params(direction="out", length=10)
        fig.savefig(out_container + "/" + os.path.basename(out_container) + "_meanvar.png", bbox_inches="tight", dpi=100)
        plt.close()
        print("Save variance graph at <" + os.path.basename(out_container) + ">")


        """
        x = np.arange(10)
        ax1 = plt.subplot(1,1,1)
        w = 0.3
        #plt.xticks(), will label the bars on x axis with the respective country names.
        plt.xticks(x + w /2, datasort['country'], rotation='vertical')
        pop =ax1.bar(x, datasort['population']/ 10 ** 6, width=w, color='b', align='center')
        #The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method.
        ax2 = ax1.twinx()
        #We have calculated GDP by dividing gdpPerCapita to population.
        gdp =ax2.bar(x + w, datasort['gdpPerCapita'] * datasort.population / 10**9, width=w,color='g',align='center')
        #Set the Y axis label as GDP.
        plt.ylabel('GDP')
        #To set the legend on the plot we have used plt.legend()
        plt.legend([pop, gdp],['Population in Millions', 'GDP in Billions'])
        #To show the plot finally we have used plt.show().
        plt.show()
        """



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

    parser = argparse.ArgumentParser(description="Compute PCA.")
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
