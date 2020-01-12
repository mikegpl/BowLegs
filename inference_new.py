# https://pydicom.github.io/pydicom/stable/getting_started.html
import os.path  # from os.path import exists

import cv2
import imutils as imutils
import numpy as np
from keras.models import model_from_json
from keras.optimizers import RMSprop
from skimage import img_as_ubyte, exposure
from skimage import morphology
from skimage import transform, io

from model.losses import bce_dice_loss, dice_coeff

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

dir_img_mask = 'results/new/'
dataset_bow_legs_dir = 'uploads'
img_names = []
mask_names = []
im_shape = (512, 256)


# TODO refactor these funcs
def load_imgs(im_names):
    _X = []
    for im_name in im_names:
        print(im_name)
        _img = io.imread(im_name)
        # io.imshow(_img)
        # io.show()
        # input("Press Enter to continue...")
        _img = transform.resize(_img, im_shape, mode='constant')
        _img = np.expand_dims(_img, -1)
        _X.append(_img)
    _X = np.array(_X)
    _X -= _X.mean()
    _X /= _X.std()
    return _X


def load_masks(im_names):
    y = []
    for im_name in im_names:
        _img = io.imread(im_name)
        _img = transform.resize(_img, im_shape, mode='constant')
        _img = np.expand_dims(_img, -1)
        y.append(_img)
    y = np.array(y)
    return y


def remove_small_regions(_img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    _img = morphology.remove_small_objects(_img, size)
    _img = morphology.remove_small_holes(_img, size)
    return _img


def get_full_path_for_data(file_names):
    _i = -1
    full_names = ['', '']
    for fname in file_names:
        _i = _i + 1
        full_names[_i] = dataset_bow_legs_dir + '/' + fname
    return full_names


def load_model():
    # load json and create model
    _json_file = open('models/model_bk.json', 'r')
    _loaded_model_json = _json_file.read()
    _json_file.close()
    _loaded_model = model_from_json(_loaded_model_json)
    print("Model loaded from the disc")
    return _loaded_model


def load_weights(file):
    loaded_model.load_weights(file)
    print("Weights loaded to model from the disc")
    return loaded_model


arr = os.listdir(dataset_bow_legs_dir)
img_names = [a for a in arr if a.endswith('png') and not (a.endswith('mask.png'))]
mask_names = [a for a in arr if a.endswith('mask.png')]

full_img_names = get_full_path_for_data(img_names)
full_mask_names = get_full_path_for_data(mask_names)
X = load_imgs(full_img_names)
Y = load_masks(full_mask_names)
print('Data loaded')

number_of_test_data = X.shape[0]
input_shape = X[0].shape
print('X.shape={} Y.shape={}'.format(X.shape, Y.shape))

# Load model and trained weights
loaded_model = load_model()
full_model = load_weights("models/trained_model.hdf5")

# Evaluate loaded model on test data
UNet = loaded_model
model = loaded_model
model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
print("Model compiled")

image_counter = 0


# TODO - implement this method, then in app import inference_new, upload image, render mask and then call calculate_angle_for_mask
def render_mask_for_image(image):
    pred = UNet.predict(image)[..., 0].reshape(input_shape[:2])
    cv2.imshow("Predicted new", pred)
    cv2.waitKey(0)
    print("Done")
    return 0


for image in img_names:
    xx_ = X[image_counter, :, :, :]
    yy_ = Y[image_counter, :, :, :]
    xx = xx_[None, ...]
    yy = yy_[None, ...]

    # TODO - this should be used later to calculate angle in separate method
    mask_for_image = render_mask_for_image(xx) # pack preprocessing into separate method
    # img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
    # cv2.imshow("Input", img)
    # cv2.waitKey(0)
    pred = UNet.predict(xx)[..., 0].reshape(input_shape[:2])
    """
    As it turned out, pred is our b/w image of bones (mask). So, there's no point in uploading both image and 
    its mask - just upload one and calculate the angle.
    """
    mask = yy[..., 0].reshape(input_shape[:2])
    # cv2.imshow("prediction", pred)
    # cv2.waitKey(0)

    gt = mask > 0.5
    pr = pred > 0.5

    pr_bin = img_as_ubyte(pr)
    pr_opened = morphology.opening(pr_bin)

    # Remove regions smaller than 0.5% of the image
    pr = remove_small_regions(pr, 0.005 * np.prod(im_shape))
    pr_out = img_as_ubyte(pr)

    # TODO Analyze the angle of bone
    _, contours, _ = cv2.findContours(pr_out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    backtorgb = cv2.cvtColor(pr_out, cv2.COLOR_GRAY2RGB)

    # Finding extreme points
    cnts = cv2.findContours(pr_out.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object and extreme points
    cv2.drawContours(backtorgb, [c], -1, (0, 255, 255), 2)
    cv2.circle(backtorgb, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(backtorgb, extRight, 8, (0, 255, 0), -1)
    cv2.circle(backtorgb, extTop, 8, (255, 0, 0), -1)
    res = cv2.circle(backtorgb, extBot, 8, (255, 255, 0), -1)

    # show the output image
    cv2.imshow("Extreme points", res)
    cv2.waitKey(0)

    # TODO Refactor
    # draw contours for first bone
    x, y, w, h = cv2.boundingRect(contours[0])
    test = cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw contours for the second bone
    x, y, w, h = cv2.boundingRect(contours[1])
    test = cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Leg bones with rectangle', test)
    print("Press any key to continue...")
    cv2.waitKey(0)

cv2.destroyAllWindows()
