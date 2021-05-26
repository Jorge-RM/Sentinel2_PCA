import os
import time

import MultiPCA

if __name__ == "__main__":
    import argparse
    import sys
    ti=time.time()
    parser = argparse.ArgumentParser(description="Compute MultiPCA.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="A Main Folder with this structure: MainFolder/Folders/ImageFolders/Images",
    )
    parser.add_argument("-o", "--output", type=str, help="Output folder")
    parser.add_argument("-b", "--bands", type=int, help="Number of bands")

    if len(sys.argv) == 7:
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
    te=time.time()
    print("Total time elapsed: %2.2f" % (te-ti))

