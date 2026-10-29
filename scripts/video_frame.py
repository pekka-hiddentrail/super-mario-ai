"""Save one frame of a rendered video as a PNG, to check the layout. Usage: video_frame.py <mp4> <seconds> <out.png>"""
import sys

import imageio.v2 as imageio

r = imageio.get_reader(sys.argv[1])
fps = r.get_meta_data()["fps"]
imageio.imwrite(sys.argv[3], r.get_data(int(float(sys.argv[2]) * fps)))
