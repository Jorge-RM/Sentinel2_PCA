import os
from PIL import Image, ImageOps

def crop_images(in_folder, out_folder):
    """Delete the white area surrounding Sentinel images."""
    images = os.listdir(in_folder)

    images = sorted([int(os.path.splitext(img)[0]) for img in images])
    images = [str(i) + ".png" for i in images]
    save_images = 1
    if save_images:
        for i, image_name in enumerate(images):

            img = Image.open(in_folder + "/" + image_name)

            crop_img = ImageOps.grayscale(img.crop((90, 96, 649, 542)))
            crop_img.save(out_folder + "/" + str(i) + ".png")