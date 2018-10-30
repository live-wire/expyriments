import torch 
import numpy as np
# import matplotlib as mpl
# mpl.use("TkAgg")
# import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import argparse
import urllib
from io import BytesIO

cnn = models.vgg19(pretrained=True).eval()

classes = json.load(open("imagenet_class_index.json"))
id2label = [classes[str(k)][1] for k in range(len(classes))]

loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def getPrediction(filepath="images/llama.jpg", url=None):
	if(url):
		file = BytesIO(urllib.request.urlopen(url).read())
		img = Image.open(file)
	else:
		img = Image.open(filepath)
	# plt.imshow(img)
	# plt.show()
	out = cnn(loader(img).unsqueeze(0))
	# print(out[0].sort(), out.sort()[1])
	sortedout = out[0].sort()
	finalclasses = []
	for idx in sortedout[1][-5:]:
	    finalclasses.append(id2label[idx])
	finalclasses.reverse()
	return finalclasses

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", help="Image file to be predicted")
	parser.add_argument("--url", help="Image url to be predicted")
	args = parser.parse_args()
	filepath = "images/llama.jpg"
	if (args.url):
		fileurl = args.url
		getPrediction(url=fileurl)
	elif (args.file):
		filepath = args.file
		getPrediction(filepath=filepath)
