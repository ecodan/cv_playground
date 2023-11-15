from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def side_by_side_from_paths(imgpath1:Path, imgpath2:Path, outfilepath:Path):
    # Open the two images
    image1 = Image.open(imgpath1)
    image2 = Image.open(imgpath2)
    return side_by_side_from_images(image1, image2, outfilepath)

def side_by_side_from_arrays(img1:np.ndarray, img2:np.ndarray, outfilepath:Path):
    # Create the two images
    # i1:np.ndarray = (img1*255).astype('uint8')
    # imgX = np.zeros([100, 100], dtype=np.uint8)
    # imageX = Image.fromarray(imgX)
    image1 = Image.fromarray((img1.reshape((img1.shape[0],img1.shape[1]))*255).astype('uint8'))
    image2 = Image.fromarray((img2.reshape((img2.shape[0],img2.shape[1]))*255).astype('uint8'))
    return side_by_side_from_images(image1, image2, outfilepath)

def side_by_side_from_images(img1:Image, img2:Image, outfilepath:Path):

    # Get the width and height of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with the combined width of both images
    combined_width = width1 + width2
    combined_height = max(height1, height2)
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the first image into the combined image
    combined_image.paste(im=img1, box=(0, 0))

    # Paste the second image next to it
    combined_image.paste(im=img2, box=(width1, 0))

    # Save the new combined image
    combined_image.save(outfilepath)


