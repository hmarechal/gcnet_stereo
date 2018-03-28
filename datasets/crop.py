from PIL import Image

filename = 'left/left_0800.png'
img = Image.open(filename)
cropped = img.crop((0, 0, 512, 256))
cropped.save('left/0800_cropped.png')

filename = 'right/right_0800.png'
img = Image.open(filename)
cropped = img.crop((0, 0, 512, 256))
cropped.save('right/0800_cropped.png')

