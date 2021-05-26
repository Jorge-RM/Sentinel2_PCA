import os

import cv2
import natsort

if __name__ == "__main__":
    """Delete no-images data from Sentinel images."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Crop input images. Example coordinates [x1, y1, x2, y2]: Etna (fire)[298, 96, 562, 620],"
        "Etna (no fire)[90, 96, 563, 620, Vesuvio[90, 96, 648, 620], Gran Canaria[90, 96, 648, 542],"
        "New GC: [0, 0, 607, 1355], New Etna [0, 193, 518, 496], New Etna [0, 219, 518, 496]"
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Folder with images to be cropped."
    )
    parser.add_argument("-o", "--output", type=str, help="Storage folder.")
    parser.add_argument("x1", type=str, help="Starting X coordinate.")
    parser.add_argument("y1", type=str, help="Starting Y coordinate.")
    parser.add_argument("x2", type=str, help="Ending X coordinate.")
    parser.add_argument("y2", type=str, help="Ending X coordinate.")

    args = parser.parse_args()

    in_folder = args.input
    out_folder = args.output
    x1 = int(args.x1)
    y1 = int(args.y1)
    x2 = int(args.x2)
    y2 = int(args.y2)
    # images = os.listdir(in_folder)

    main_folders = os.listdir(in_folder)
    for main_folder in main_folders:
        main_folder_path = os.path.join(in_folder, main_folder)
        folders = os.listdir(main_folder_path)
        for folder in folders:
            folder_path = os.path.join(main_folder_path, folder)
            imgs = os.listdir(folder_path)
            for img_name in imgs:
                img = cv2.imread(
                    folder_path + "/" + img_name, cv2.IMREAD_ANYDEPTH
                )
                crop_img = img[x1:x2, y1:y2]
                out_path = out_folder + "/" + main_folder + "/" + folder
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)
                cv2.imwrite(out_path + "/" + img_name, crop_img)
                print(out_path + "\n")
