import imageio

# scipy.misc.imread is deprecated.

img = imageio.imread("./example_img.jpg")

print("Image Shape: {}.".format(img.shape))

print("Number of Channels: {}.".format(img.shape[2]))

print("Image data type: {}.".format(img.dtype))

print(img[100: 102, 100: 102, :])
