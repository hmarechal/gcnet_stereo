#!/usr/bin/env python
# Convert the given input file to a groundtruth depth image
# File must be in PFM format with .pfm extension

import string
import argparse
import numpy as np
from PIL import Image
from pfm_reader import PFMReader


def main(filename):
    if not filename.endswith('.pfm'):
        return 1

    pfm_reader = PFMReader()
    try:
        data, scale = pfm_reader.readPFM(filename)
    except:
        print('Error opening file', filename)
        raise

    data = data / np.max(data) * 256

    u8data = data.astype(np.uint8)

    img = Image.fromarray(u8data, 'L')

    img_filename = string.rstrip(filename, '.pfm')
    img_filename  += '.png'
    img.save(img_filename)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', help='Filename with .pfm extension')

    args = parser.parse_args()

    filename = args.filename

    if main(filename) != 0:
        print ("Error")
