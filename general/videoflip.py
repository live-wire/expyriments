# Video reverser

import imageio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="file path of video to be reversed")
args = parser.parse_args()

def main():
	if(not args.video):
		print("Usage $ python videoflip.py --video <filename>")
		return
	filename = args.video
	vid = imageio.get_reader(filename, 'ffmpeg')
	fps = vid.get_meta_data()['fps']
	reversed_file = filename[:filename.find(".")]+"_reversed"+filename[filename.find("."):]
	writer = imageio.get_writer(reversed_file, fps=fps)
	for i,im in reversed(list(enumerate(vid))):
	    writer.append_data(im)
	writer.close()
	print("DONE ====>\nGenerated file : ", reversed_file)


if __name__ == "__main__":
	main()