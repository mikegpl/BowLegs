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


#dataset_bow_legs_dir = 'uploads'
#img_names = ['!002115_.png', '!002308_.png']
#mask_names = ['!002115__mask.png', '!002308__mask.png']
dataset_bow_legs_dir = 'dataset_bow-legs'
img_names = ['mask_050/!002115_.png', 'mask_051/!002308_.png']
mask_names = ['mask_050/!002115__mask.png', 'mask_051/!002308__mask.png']


# TODO refactor these fun


def load_imgs(im_names, _im_shape):
    _X = []
    for im_name in im_names:
        print(im_name)
        _img = io.imread(im_name)
        _img = transform.resize(_img, _im_shape,  mode='constant')
        _img = np.expand_dims(_img, -1)
        _X.append(_img)
    _X = np.array(_X)
    _X -= _X.mean()
    _X /= _X.std()
    return _X


def load_masks(im_names, _im_shape):
    y = []
    for im_name in im_names:
        _img = io.imread(im_name)
        _img = transform.resize(_img, _im_shape,  mode='constant')
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

    min_val = gt.min()	
    max_val = gt.max()
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


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img
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


if __name__ == '__main__':

    args = parse_args()
    im_shape = (512, 256)

    full_img_names = get_full_path_for_data(img_names)
    full_mask_names = get_full_path_for_data(mask_names)
    X = load_imgs(full_img_names,  im_shape)
    Y = load_masks(full_mask_names, im_shape)
    print('Data loaded')

    n_test = X.shape[0]
    inp_shape = X[0].shape
    print('X.shape={} Y.shape={}'.format(X.shape, Y.shape))

    # Load model and trained weights
    loaded_model = load_model()
    load_weights()
 
    # Evaluate loaded model on test data
    UNet = loaded_model	
    model = loaded_model	
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])	
    print("Model compiled")

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    mean_IoU = np.zeros(n_test)

    i = 0
    num_imgs = X.shape[0]
    for ii in range(num_imgs):
        xx_ = X[ii, :, :, :]
        yy_ = Y[ii, :, :, :]
        xx = xx_[None, ...]
        yy = yy_[None, ...]		
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))

        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        pr_bin = img_as_ubyte(pr)
        pr_openned = morphology.opening(pr_bin)

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.005 * np.prod(im_shape))
        # pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        pr_out = img_as_ubyte( pr)	

        file_name = '!002308_.png'
        print('file_name')
        print(file_name)

        if args.save_out_images:
            #dir_img_mask = 'results/bow-legs_test/{}'.format(sub_dir_name)
            dir_img_mask = 'results/bow-legs_test/'
            if not os.path.exists(dir_img_mask):
                os.makedirs(dir_img_mask)
            img_name = file_name

            cv2.imwrite(img_name, pr_openned)

        file_name_no_ext = os.path.splitext(file_name)[0]
        # ('file', '.ext')  --> os.path.splitext(file_name_no_ext)[0] ('file')
        file_name_in = dataset_bow_legs_dir + '/' + file_name_no_ext + '_mask' + '.png'
        # dataset_bow-legs/mask_001/img_0001_mask.png

        if args.save_out_images:

            file_name_out = 'results/bow-legs_test' + '/' + file_name_no_ext + '_mask_manual' + '.png'
            # results/bow-legs_test/img_0006_mask_manual.png

            img_exists = os.path.isfile(file_name_in)		
            if not img_exists:
                print('{} does not exists\n'.format(file_name_in))
                sys.exit("Exit ...")

            shutil.copy2(file_name_in, file_name_out)

        # ---------------------------------------------------------------
        # Conversion to a new size
        # im_name_x_ray_original_size_test = dataset_bow_legs_dir + '/' + 'x-ray_test/' + file_name			# data_bow-legs/x-ray/img_0001.png
        # im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)
        #
        # height, width = im_x_ray_original_size.shape[:2]							# height, width  -- original image size
        #
        # ratio = float(height) / width
        #
        # new_shape = (4*256, int(4*256*ratio))
        # im_x_ray_4x = cv2.resize(im_x_ray_original_size, new_shape)
        #
        # dir_img_x_ray_4x = 'results/bow-legs_test_4x/'
        # if not os.path.exists(dir_img_x_ray_4x):
        #     os.makedirs(dir_img_x_ray_4x)
        # im_name_x_ray_4x = '{}/{}'.format(dir_img_x_ray_4x, file_name)
        # cv2.imwrite(im_name_x_ray_4x, im_x_ray_4x)
        #
        # # mask
        # im_mask_original_size = cv2.imread( file_name_in, cv2.IMREAD_GRAYSCALE)
        # im_mask_4x = cv2.resize(im_mask_original_size, new_shape)
        # im_name_mask_4x = '{}/{}'.format(dir_img_x_ray_4x, '/' + file_name_no_ext + '_mask_manual' + '.png')
        # cv2.imwrite( im_name_mask_4x, im_mask_4x)
        #
        # # Unet output
        # pr_openned_4x = cv2.resize( pr_openned, new_shape)
        # im_name_pr_openned_4x = '{}/{}'.format(dir_img_x_ray_4x, file_name_no_ext + '_mask_Unet' + '.png')
        # cv2.imwrite(im_name_pr_openned_4x, pr_openned_4x)
        #
        # gt_4x = cv2.resize(img_as_ubyte(gt), new_shape)
        #
        # gt_4x = gt_4x > 0.5
        # pr_openned_4x = pr_openned_4x > 0.5
        # im_x_ray_4x_ = im_x_ray_4x/255
        # im_masked_4x = masked( im_x_ray_4x, gt_4x, pr_openned_4x, 0.5)			# img.max()=1.0 gt.max()=True pr.max()=True

        # if args.save_out_images:
        #     dir_im_masked_4x = 'results/bow-legs_masked_4x'
        #     if not os.path.exists(dir_im_masked_4x):
        #         os.makedirs(dir_im_masked_4x)
        #     im_name_masked_4x = '{}/{}'.format(dir_im_masked_4x, file_name)
        #
        #     im_masked_4x = img_as_ubyte( im_masked_4x)
        #     io.imsave( im_name_masked_4x, im_masked_4x)

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print('{}  {:.4f} {:.4f}'.format(img_names[ii], ious[i], dices[i]))

        with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
            print('{}  {:.4f} {:.4f}'.format(img_names[ii], ious[i], dices[i]), file=f)

        i += 1
        if i == n_test:
            break

    print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format( ious.mean(), dices.mean() ))
    with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean() ), file=f)
        print('\n', file=f)

    with open("results/bow-legs_IoU_Dice.txt", "a", newline="\r\n") as f:
        print( 'Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean() ), file=f)

# print('\n')
# print('Press any key to exit ')
# cv2.waitKey(0)
cv2.destroyAllWindows()
