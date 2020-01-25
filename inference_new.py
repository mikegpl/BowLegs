import math
import os.path

import cv2
import imutils as imutils
import numpy as np
from keras.models import model_from_json
from keras.optimizers import RMSprop
from skimage import img_as_ubyte
from skimage import morphology
from skimage import transform, io

from model.losses import bce_dice_loss, dice_coeff

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

dir_img_mask = 'results/new/'
dataset_bow_legs_dir = 'uploads'
im_shape = (512, 256)
VISUAL = False


# TODO refactor these funcs
def load_imgs(im_names):
    _X = []
    for im_name in im_names:
        print(im_name)
        _img = io.imread(im_name)
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
    print("Model loaded from the disk")
    return _loaded_model


def load_weights(_loaded_model, file):
    _loaded_model.load_weights(file)
    print("Weights loaded to model from the disc")
    return _loaded_model


# TODO - in app import inference_new, upload image, render mask and call calculate_angle_for_mask
def render_mask_for_image(im_name, trained_model):
    print('Start render mask for image {} '.format(im_name))
    loaded_img = io.imread(im_name)
    loaded_img = transform.resize(loaded_img, im_shape, mode='constant')
    loaded_img = np.expand_dims(loaded_img, -1)

    loaded_img_array = np.array([loaded_img])
    loaded_img_array -= loaded_img_array.mean()
    loaded_img_array /= loaded_img_array.std()

    xx_ = loaded_img_array[0, :, :, :]
    xx = xx_[None, ...]

    predicted_mask = trained_model.predict(xx)[..., 0].reshape(loaded_img_array[0].shape[:2])

    # Binary masks
    pr = predicted_mask > 0.5

    # Remove regions smaller than 0.5% of the image
    pr = remove_small_regions(pr, 0.005 * np.prod(im_shape))
    pr_out = img_as_ubyte(pr)

    if VISUAL:
        cv2.imshow("Mask for image", pr_out)
        cv2.waitKey(0)
        print("Render mask for image - done")
    return pr_out


def angle_from_mask(_mask):
    # TODO Analyze the angle of bone
    _, contours, _ = cv2.findContours(_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    backtorgb = cv2.cvtColor(_mask, cv2.COLOR_GRAY2RGB)

    # Finding extreme points
    cnts = cv2.findContours(_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # determine the most extreme points along the contour
    extreme_left = tuple(c[c[:, :, 0].argmin()][0])
    extreme_right = tuple(c[c[:, :, 0].argmax()][0])
    extreme_top = tuple(c[c[:, :, 1].argmin()][0])
    extreme_bottom = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object and extreme points
    cv2.drawContours(backtorgb, [c], -1, (0, 255, 255), 2)
    cv2.circle(backtorgb, extreme_left, 8, (0, 0, 255), -1)
    cv2.circle(backtorgb, extreme_right, 8, (0, 255, 0), -1)
    cv2.circle(backtorgb, extreme_top, 8, (255, 0, 0), -1)
    res = cv2.circle(backtorgb, extreme_bottom, 8, (255, 255, 0), -1)

    if VISUAL:
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

    if VISUAL:
        cv2.imshow('Leg bones with rectangle', test)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    [vx1, vy1, _, _] = cv2.fitLine(contours[0], cv2.cv2.DIST_L2, 0, 0.01, 0.01)
    [vx2, vy2, _, _] = cv2.fitLine(contours[1], cv2.cv2.DIST_L2, 0, 0.01, 0.01)
    vec1 = np.squeeze(np.asarray([vx1, vy1]))
    vec2 = np.squeeze(np.asarray([vx2, vy2]))
    return angle_between(vec1, vec2)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    in_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return abs(180 - math.degrees(in_radians))


def build_model():
    m = load_model()
    loaded_m = load_weights(m, "models/trained_model.hdf5")
    loaded_m.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    return loaded_m


def angle_for_image(path, predicting_model):
    mask = render_mask_for_image(path, predicting_model)
    return angle_from_mask(mask)


if __name__ == '__main__':
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
    model = load_model()
    full_model = load_weights(model, "models/trained_model.hdf5")

    # Evaluate loaded model on test data
    UNet = model

    # U = model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    U2 = full_model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    print("Model compiled")

    mask = render_mask_for_image('uploads/photo_002115_.png', full_model)
    print("Angle {0:.2f}".format(angle_from_mask(mask)))

    # To just test calculating angle.
    # model = build_model()
    # print("Angle {0:.2f}".format(angle_for_image('uploads/photo_002115_.png',model)))
