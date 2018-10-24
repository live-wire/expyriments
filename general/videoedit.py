# Video reverser

import imageio
import argparse
import sys
import numpy as np
import cv2
import os

args = None
# Bresenham's Line equqtion
# Sample m elements from a range of n
sampler = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]

# Silence warnings
import imageio.core.util

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", help="file path of video to be reversed")
	parser.add_argument("--fps", help="sample file to this fps")
	parser.add_argument("--resize", help="resize the samples to this")
	parser.add_argument("-flip", action="store_true")
	args = parser.parse_args()
	if (not args.video):
		print("Usage $ python %s --video <filename>"%sys.argv[0])
		return
	filename = args.video
	filename = process(filename, args)
	if (args.flip):
		delfilename = filename
		filename = flip(filename)
		os.remove(delfilename)
	
	print("[ DONE ]\nGenerated file : ", filename)
	print("----- "*20)

def flip(filename):
	vid = imageio.get_reader(filename, 'ffmpeg')
	fps = vid.get_meta_data()['fps']
	edited_file = filename[:filename.find(".")]+"_flip"+filename[filename.find("."):]
	writer = imageio.get_writer(edited_file, fps=fps)
	for i,im in reversed(list(enumerate(vid))):
		writer.append_data(im)
	writer.close()
	return edited_file

def process(filename, args = None, fpsArg = None, resizeArg = None):
	vid = imageio.get_reader(filename, 'ffmpeg')
	fps = vid.get_meta_data()['fps']
	newfps = fps
	if (fpsArg):
		newfps = fpsArg
	elif (args.fps):
		newfps = int(args.fps)
	edited_file = filename[:filename.find(".")]+"_processed"+filename[filename.find("."):]
	print("FILE:", filename, "\nFPS:", fps, "| New FPS:", newfps)
	print("Video length (frames):", len(vid))
	print("Video length (seconds):", len(vid)/fps)
	totalsamples = newfps * (len(vid)/fps)
	samples = sampler(int(totalsamples), len(vid))
	print("Samples:", len(samples))
	writer = imageio.get_writer(edited_file, fps=newfps)
	print("---","Writing now","---")
	ptillnow = 0
	for i,im in enumerate(vid):
		if (i in samples):
			progress = int((i/len(vid))*100)
			if progress!=ptillnow:
				print("|====[ "+str(progress)+" % ]", end='\r')
				ptillnow = progress
			resized_image = im
			if (resizeArg):
				resize = resizeArg
				resized_image = cv2.resize(im, resize)
			elif (args.resize):
				resize = (int(args.resize), int(args.resize))
				resized_image = cv2.resize(im, resize)
			black_white = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
			writer.append_data(black_white)
	print("")
	writer.close()
	return edited_file

if __name__ == "__main__":
	main()

