import os

import MultiPCA

bands = 13

output = "D:/Teleco/TFG/Python/Galicia_Out"
container = "D:/Teleco/TFG/Python/Fuegos_Galicia_crop"

folders = os.listdir(container)
for folder in folders:
    in_path = os.path.join(container, folder)
    out_path = os.path.join(output, folder)
    pca = MultiPCA.MultiPCA(in_path, bands)
    pca.save_data(out_path)

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute MultiPCA.")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-b", "--bands", type=int)

    if len(sys.argv) == 6:
        args = parser.parse_args()
        out_container = args.output
        in_container = args.input
        bands = args.bands

        folders = os.listdir(in_container)
        for folder in folders:
            in_path = os.path.join(in_container, folder)
            out_path = os.path.join(out_container, folder)
            pca = MultiPCA.MultiPCA(in_path, bands)
            pca.save_data(out_path)
