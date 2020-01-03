# https://pydicom.github.io/pydicom/stable/getting_started.html
import os
import sys
import os.path			# from os.path import exists
import shutil			# shutil.copy2
import glob
import argparse
import numpy as np
import cv2
from model.losses import bce_dice_loss, dice_loss, dice_coeff
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.models import model_from_json
import keras.backend as K
from skimage import morphology, color, io, exposure
from skimage import img_as_ubyte
from skimage import transform, io, img_as_float, exposure

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["CUDA_VISIBLE_DEVICES"]='-1'

dir_img_mask = 'results/new/'
dataset_bow_legs_dir = 'uploads'
img_names = []
mask_names = []
#img_names = ['!002115_.png', '!002308_.png']
#mask_names = ['!002115__mask.png', '!002308__mask.png']
# dataset_bow_legs_dir = 'dataset_bow-legs'
# img_names = ['mask_050/!002115_.png', 'mask_051/!002308_.png']
# mask_names = ['mask_050/!002115__mask.png', 'mask_051/!002308__mask.png']
im_shape = (512, 256)


# TODO refactor these funcs
def load_imgs(im_names):
    _X = []
    for im_name in im_names:
        print(im_name)
        _img = io.imread(im_name)
        _img = transform.resize(_img, im_shape,  mode='constant')
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
        _img = transform.resize(_img, im_shape,  mode='constant')
        _img = np.expand_dims(_img, -1)
        y.append(_img)
    y = np.array(y)
    return y		


# -------------------------------------------------------------------
def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, 
    predicted lung field filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))

    # min_val = gt.min()
    # max_val = gt.max()
    # print('min_val = {} max_val = {}\n'.format(min_val, max_val))

    # boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt	
    # boundary = morphology.dilation(gt, morphology.disk(1)) ^ gt

    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(_img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    _img = morphology.remove_small_objects(_img, size)
    _img = morphology.remove_small_holes(_img, size)
    return _img
# -------------------------------------------------------------------


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')	


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", type=str, required=False, default='models/trained_model.hdf5',
                        help="full path to model")
    # parser.add_argument( "-i", "--integer", type=int, default=50)
    parser.add_argument('-d', '--disp_test_images', type=str2bool, default=False,
                        help="whether to display test image names. default: False")
    parser.add_argument('-s', '--save_out_images', type=str2bool, default=True,
                        help="whether to save out images. default: True")

    return parser.parse_args()


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


def load_weights():
    # load weights into a new model
    if ".hdf5" not in args.fname:
        _list_of_files = glob.glob(args.fname + '/' + '*.hdf5')
        _model_weights = _list_of_files[0]
    else:
        _model_weights = args.fname
    loaded_model.load_weights(_model_weights)
    print("Weights loaded to model from the disc")


def save_mean_results(_ious, _dices):
    print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(_ious.mean(), _dices.mean()))
    with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(_ious.mean(), _dices.mean()), file=f)
        print('\n', file=f)

    with open("results/bow-legs_IoU_Dice.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(_ious.mean(), _dices.mean()), file=f)


def copy_image_to_results(_image, _pr_opened):
    if args.save_out_images:
        if not os.path.exists(dir_img_mask):
            os.makedirs(dir_img_mask)
        cv2.imwrite(_image, _pr_opened)


def copy_mask_to_results(_image_name_no_extension):
    if args.save_out_images:
        _file_name_in = dataset_bow_legs_dir + '/' + _image_name_no_extension + '_mask' + '.png'
        _file_name_out = 'results/new/' + _image_name_no_extension + '_mask_manual' + '.png'
        _img_exists = os.path.isfile(_file_name_in)
        if not _img_exists:
            print('{} does not exists\n'.format(_file_name_in))
            sys.exit("Exit ...")
        shutil.copy2(_file_name_in, _file_name_out)


def save_results_4x(_image, _im_masked_4x):
    if args.save_out_images:
        dir_im_masked_4x = 'results/new/masked_4x'
        if not os.path.exists(dir_im_masked_4x):
            os.makedirs(dir_im_masked_4x)
        im_name_masked_4x = '{}/{}'.format(dir_im_masked_4x, image)
        _im_masked_4x = img_as_ubyte(_im_masked_4x)
        io.imsave(im_name_masked_4x, _im_masked_4x)


if __name__ == '__main__':

    arr = os.listdir(dataset_bow_legs_dir)
    img_names = [a for a in arr if a.endswith('png') and not (a.endswith('mask.png'))]
    mask_names = [a for a in arr if a.endswith('mask.png')]

    args = parse_args()

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
    load_weights()
 
    # Evaluate loaded model on test data
    UNet = loaded_model	
    model = loaded_model	
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])	
    print("Model compiled")

    ious = np.zeros(number_of_test_data)
    dices = np.zeros(number_of_test_data)
    mean_IoU = np.zeros(number_of_test_data)
    image_counter = 0

    for image in img_names:
        print(image)
        image_name_no_extension = os.path.splitext(image)[0]

        xx_ = X[image_counter, :, :, :]
        yy_ = Y[image_counter, :, :, :]
        xx = xx_[None, ...]
        yy = yy_[None, ...]		
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))

        pred = UNet.predict(xx)[..., 0].reshape(input_shape[:2])
        mask = yy[..., 0].reshape(input_shape[:2])

        # Binary masks
        gt = mask > 0.5
        pr = pred > 0.5

        pr_bin = img_as_ubyte(pr)
        pr_opened = morphology.opening(pr_bin)

        # Remove regions smaller than 0.5% of the image
        pr = remove_small_regions(pr, 0.005 * np.prod(im_shape))
        pr_out = img_as_ubyte(pr)

        copy_image_to_results(image, pr_opened)
        copy_mask_to_results(image_name_no_extension)

        # ---------------------------------------------------------------
        # Conversion to a new size
        im_name_x_ray_original_size_test = dataset_bow_legs_dir + '/' + 'x-ray_test/' + image
        im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)
        height, width = im_x_ray_original_size.shape[:2]							# height, width  -- original image size
        ratio = float(height) / width
        new_shape = (4*256, int(4*256*ratio))
        im_x_ray_4x = cv2.resize(im_x_ray_original_size, new_shape)
        dir_img_x_ray_4x = 'results/new/4x/'
        if not os.path.exists(dir_img_x_ray_4x):
            os.makedirs(dir_img_x_ray_4x)
        im_name_x_ray_4x = '{}/{}'.format(dir_img_x_ray_4x, image)
        cv2.imwrite(im_name_x_ray_4x, im_x_ray_4x)

        # mask
        im_mask_original_size = cv2.imread( image, cv2.IMREAD_GRAYSCALE)
        im_mask_4x = cv2.resize(im_mask_original_size, new_shape)
        im_name_mask_4x = '{}/{}'.format(dir_img_x_ray_4x, '/' + image_name_no_extension + '_mask_manual' + '.png')
        cv2.imwrite(im_name_mask_4x, im_mask_4x)

        # Unet output
        pr_openned_4x = cv2.resize( pr_opened, new_shape)
        im_name_pr_openned_4x = '{}/{}'.format(dir_img_x_ray_4x, image_name_no_extension + '_mask_Unet' + '.png')
        cv2.imwrite(im_name_pr_openned_4x, pr_openned_4x)

        gt_4x = cv2.resize(img_as_ubyte(gt), new_shape)

        gt_4x = gt_4x > 0.5
        pr_openned_4x = pr_openned_4x > 0.5
        im_x_ray_4x_ = im_x_ray_4x/255
        im_masked_4x = masked(im_x_ray_4x, gt_4x, pr_openned_4x, 0.5)			# img.max()=1.0 gt.max()=True pr.max()=True

        save_results_4x(image, im_masked_4x)

        ious[image_counter] = IoU(gt, pr)
        dices[image_counter] = Dice(gt, pr)
        print('{}  {:.4f} {:.4f}'.format(img_names[image_counter], ious[image_counter], dices[image_counter]))

        with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
            print('{}  {:.4f} {:.4f}'.format(img_names[image_counter], ious[image_counter], dices[image_counter]), file=f)
        image_counter += 1

    save_mean_results(ious, dices)

cv2.destroyAllWindows()
