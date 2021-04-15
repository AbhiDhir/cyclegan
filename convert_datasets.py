import glob
import skimage as sk
from skimage import io, transform, util
import json
import numpy as np
import cv2
import random
import os
import math
REAL_IMAGE_GLOB = r"F:\ravenml\datasets\cygnus_real_1388\*.jpg"
SYN_IMAGE_GLOB = r"F:\ravenml\datasets\cygnus_20k_re_norm_mix_drb\test\*.png"
REAL_OUT_DIR = r"F:\lab-seeker-gan\seeker\trainB"
SYN_OUT_DIR = r"F:\lab-seeker-gan\seeker\trainA"
SIZE = 256
os.makedirs(REAL_OUT_DIR, exist_ok=True)
os.makedirs(SYN_OUT_DIR, exist_ok=True)
# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)
    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5
    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0],
    ]
    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]
    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]
    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)
    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])
    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    # Apply the transform
    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return result
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)
def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    if width > image_size[0]:
        width = image_size[0]
    if height > image_size[1]:
        height = image_size[1]
    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]
def rotate(image, i):
    image_height, image_width = image.shape[:2]
    image_rotated = rotate_image(image, i)
    return crop_around_center(image_rotated, *largest_rotated_rect(image_width, image_height, math.radians(i)))
def random_crop(img):
    img = rotate(img, random.uniform(0, 360))
    h, w, _ = img.shape
    ndim = random.randint(80, min(h, w))
    img = img[:h, :w]
    return img

real_cnt = 0
for img_fn in glob.iglob(REAL_IMAGE_GLOB):
    json_fn = img_fn.replace(".jpg", ".json").replace("image", "meta")
    with open(json_fn, "r") as f:
        meta = json.loads(f.read())
    bbox = meta["bboxes"]["cygnus"]
    img_original = io.imread(img_fn)
    ymin, ymax, xmin, xmax = int(bbox["ymin"]), int(bbox["ymax"]), int(bbox["xmin"]), int(bbox["xmax"])
    r = int(max(ymax - ymin, xmax - xmin) / 2 * 1.3)
    cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2
    temp = min(cy - r, cy + r, cx - r, cx + r)
    if temp < 0:
        r += temp
    cropped = img_original[cy - r : cy + r, cx - r : cx + r]
    real_cnt += 1
    img_norm = transform.resize(cropped, (SIZE, SIZE))
    io.imsave(os.path.join(REAL_OUT_DIR, str(real_cnt).zfill(4) + ".png"), util.img_as_ubyte(img_norm))
    real_cnt += 1
    img_flip = random.choice([cropped.copy(), cropped[:, ::-1]])
    img_flip = rotate(img_flip, random.uniform(0, 360))
    img_flip = transform.resize(img_flip, (SIZE, SIZE))
    io.imsave(os.path.join(REAL_OUT_DIR, str(real_cnt).zfill(4) + ".png"), util.img_as_ubyte(img_flip))
for img_fn in glob.iglob(r"F:\ravenml\datasets\nasa_images_filtered\*.jpg"):
    img_original = io.imread(img_fn)
    real_cnt += 1
    img_norm = transform.resize(img_original, (SIZE, SIZE))
    io.imsave(os.path.join(REAL_OUT_DIR, str(real_cnt).zfill(4) + ".png"), util.img_as_ubyte(img_norm))
syn_cnt = 0
for img_fn in glob.iglob(SYN_IMAGE_GLOB):
    json_fn = img_fn.replace(".png", ".json").replace("image", "meta")
    with open(json_fn, "r") as f:
        meta = json.loads(f.read())
    try:
        bbox = meta["bboxes"]["cygnus"]
    except:
        continue
    img_original = io.imread(img_fn)
    ymin, ymax, xmin, xmax = bbox["ymin"], bbox["ymax"], bbox["xmin"], bbox["xmax"]
    r = int(max(ymax - ymin, xmax - xmin) / 2 * 1.2)
    cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2
    temp = min(cy - r, cy + r, cx - r, cx + r)
    if temp < 0:
        r += temp
    cropped = img_original[cy - r : cy + r, cx - r : cx + r]
    syn_cnt += 1
    img_norm = transform.resize(cropped, (SIZE, SIZE))
    io.imsave(os.path.join(SYN_OUT_DIR, str(syn_cnt).zfill(4) + ".png"), util.img_as_ubyte(img_norm))
    syn_cnt += 1
    img_flip = random.choice([cropped.copy(), cropped[:, ::-1]])
    img_flip = rotate(img_flip, random.uniform(0, 360))
    img_flip = transform.resize(img_flip, (SIZE, SIZE))
    io.imsave(os.path.join(SYN_OUT_DIR, str(syn_cnt).zfill(4) + ".png"), util.img_as_ubyte(img_flip))
    if syn_cnt > real_cnt:
        break