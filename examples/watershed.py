#
# Watershed segmentation
#

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
from edt import edt
from skimage.segmentation import watershed
# from skimage.morphology import ball, disk, square, cube
from skimage.measure import block_reduce
import eilnn

import skimage.morphology as morph


def extendedmin(img, h):
    mask = img.copy()
    marker = mask + h
    hmin = morph.reconstruction(marker, mask, method="erosion")
    return morph.local_minima(hmin)


def vol_frac(im, phase):
    return np.sum(im == phase) / np.size(im)


data_dir = os.path.join(eilnn.IMAGES_DIR, "pore_spy_tests\\label_export")
# data_dir = "C:\\Users\\tom\\Documents\\label_export\label_export"
files = os.listdir(data_dir)
files = [f.split("_")[1] for f in files]
files = [f.split(".")[0] for f in files]
files = np.asarray(files).astype(int)
files = sorted(files)
ims = []
for i, num in enumerate(files):
    fname = "label_" + str(num) + ".tiff"
    fpath = os.path.join(data_dir, fname)
    ims.append(io.imread(fpath))
    print("File", i, "loaded")

im = np.asarray(ims)
im = im == 255
print("Loaded images")

# Downsample the image
im = block_reduce(im, block_size=(2, 2, 2), func=np.min)

# Get slice indices
x, y, z = np.ceil(np.array(im.shape) / 2).astype(int)

# Raw image
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(im[x, :, :])
axes[1].imshow(im[:, y, :])
axes[2].imshow(im[:, :, z])

# # Binary opening
# im = spim.binary_opening(im, ball(1), 10)
# fig, axes = plt.subplots(1, 3, figsize=(15, 10))
# axes[0].imshow(im[x, :, :])
# axes[1].imshow(im[:, y, :])
# axes[2].imshow(im[:, :, z])
# print('Done Binary Opening')

# Distance Transform
D = -edt(im)
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(D[x, :, :])
axes[1].imshow(D[:, y, :])
mappable = axes[2].imshow(D[:, :, z])
plt.colorbar(mappable)
print("Done Distance Transform")

# Extended minima
mask = extendedmin(D, 2)
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(mask[x, :, :])
axes[1].imshow(mask[:, y, :])
axes[2].imshow(mask[:, :, z])
print("Done Extended Minima")

# Labelled minima - new step for python

mlabel, N = morph.label(mask, return_num=True)
# Randomize - to make clearer image

mlabel_cpy = mlabel.copy()
rnd_lab = np.arange(N - 1) + 1
np.random.shuffle(rnd_lab)
for i, r in enumerate(rnd_lab):
    mlabel[mlabel_cpy == r] = i
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(mlabel[x, :, :])
axes[1].imshow(mlabel[:, y, :])
axes[2].imshow(mlabel[:, :, z])
print("Done labelling and shuffling minima")

# Watershed
W = watershed(D, markers=mlabel, mask=im).astype(float)
# make background white
W[W == 0] = np.nan
# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(W[x, :, :])
axes[1].imshow(W[:, y, :])
axes[2].imshow(W[:, :, z])
print("Done Watershed")
