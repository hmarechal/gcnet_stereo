import re
import numpy as np
from PIL import Image


class PFMReader():
    def readPFM(self, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip().decode()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip().decode())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data


if __name__ == '__main__':
    pfm_reader = PFMReader()
    data, scale = pfm_reader.readPFM('../dataset/disparity/0400.pfm')

    data = data / np.max(data) * 256

    u8data = data.astype(np.uint8)

    print ('data', type(data), data, 'max', np.max(data), 'min', np.min(data))
    print('u8data', type(u8data), u8data, 'max', np.max(u8data), 'min', np.min(u8data))
    print ('scale', type(scale), scale)

    img = Image.fromarray(u8data, 'L')
    img.save('groundtruth.png')
