from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from io import BytesIO
from flask import send_file
from matplotlib import cm

def load_img(file):
    img = Image.open(file)
    out_np = np.asarray(img)
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np, img.format


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):

    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return tens_orig_l, tens_rs_l


def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


def serve_pil_image(np_img, format):
    pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
    img_io = BytesIO()
    pil_img.save(img_io, format, quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/' + format)
