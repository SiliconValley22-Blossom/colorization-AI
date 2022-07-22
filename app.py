from flask import Flask, send_file
from flask import request
from service import generate
from util import preprocess_img, load_img, postprocess_tens, serve_pil_image
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

gen = generate(pretrained=True).eval()


@app.route("/", methods=['get'])
def test():
    return "hello"

@app.route('/image', methods=['post'])
def post_image():  # put application's code here
    file = request.files['file']
    img, fm = load_img(file)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    result = postprocess_tens(tens_l_orig, gen(tens_l_rs).cpu())

    return serve_pil_image(result, fm)


if __name__ == '__main__':
    app.run()
