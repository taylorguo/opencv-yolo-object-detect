# coding:utf-8
from utils import is_downloadable, get_name, download_image, IMG_DIR
from yolo import yolov3_detector

import time, os

import requests
# from PIL import Image
from flask  import Flask, render_template, request, jsonify
app = Flask(__name__)

detector = yolov3_detector()
net = detector.load_net_weights()

@app.route("/", methods=['GET'])
def homepage():
	# return render_template("index.html")
	return "Yolov3 Object Detection API"

@app.route("/predict", methods=['POST'])
def predict():

	data = {"success": False}
	s = time.time()

	if request.method == "POST":
		req_data = request.get_json()
		image_url = req_data["image"]

		if is_downloadable(image_url):
			image_name = get_name(image_url)
			if (image_name.endswith("jpg") or image_name.endswith("png") or image_name.endswith("jpeg")):
			# for image_name.rsplit(".", 1)[1] in IMG_EXT:
				res = requests.get(image_url)
			# 	image_array = Image.open(io.BytesIO(res.content))

				download_image(image_url,image_name)
				image_path = os.path.join(IMG_DIR, image_name)
				try:
					result = detector.object_detector(net, image_path)
				except Exception:
					error_msg = "Not supported image format"
					data = {"image": image_url, "success": False, "error_msg": error_msg}
				else:
					data = {"image":image_url, "success": True, "object_names":result["names"], "rect":result["objects"]}
			else:
				error_msg = "Not supported image format"
				data = {"image":image_url, "success": False, "error_msg": error_msg}
		else:
			data = {"image": image_url, "success": False, "error_msg": "Invalid image hyperlink"}

	print(" -  API takes {:.3f} seconds!\n".format(time.time()-s))

	return jsonify(str(data))

if __name__ == "__main__":
	print(" - Starting api service ...")
	app.run(host="127.0.0.1")
