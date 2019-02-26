
#### images utils
import requests
import os

IMG_EXT = ["jpg", "jpeg", "png"]

IMG_DIR = "images"

def is_downloadable(url):
	# does url contain a downloadable resource
	h = requests.head(url, allow_redirects=True)
	header = h.headers
	content_type = header.get("content-type")
	if "text" in content_type.lower():
		return False
	if "html" in content_type.lower():
		return False

	# target file should smaller thant 5MB
	content_length = header.get("contect-length", None)
	if content_length and content_length > 5e7:
		return False

	return True

def get_name(url):
	if url.find("/"):
		return url.rsplit("/", 1)[1]


def download_image(url, image_name):
	r = requests.get(url)

	if not os.path.exists(IMG_DIR):
		os.mkdir(IMG_DIR)

	with open(os.path.join(IMG_DIR,image_name), "wb") as f:
		f.write(r.content)

	return True
