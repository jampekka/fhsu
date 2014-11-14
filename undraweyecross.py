import os
import sys
import numpy as np
from sanegst import gst
from scipy import ndimage
from scipy.ndimage import correlate

from PIL import Image
from StringIO import StringIO

DEFAULT_TEMPLATE_FILE = os.path.join(
				os.path.dirname(__file__),
				'cross_template.png')
def log_messages(*args):
	#print args
	return True

def frame_to_numpy(frame):
	if not hasattr(frame, 'read'):
		frame = StringIO(frame)
	img = Image.open(frame)
	
	img = np.array(img, dtype=float)
	img /= 255.0
	return img


def norm_corr(image, template):
	corr = correlate(image, template)
	return corr/float(np.prod(template.shape))

def megagreen(frame):
	green = frame[:,:,1].copy()
	green -= frame[:,:,0] + frame[:,:,2]
	green[green < 0.1] = -1.0
	green[green > 0] = 1.0
	return green

magic_limit = 0.7

def get_frame_cross(frame, template):
	def plotit():
		import pylab
		#pylab.ion()
		pylab.cla()
		pylab.imshow(frame)
		if pointcorr > magic_limit:
			pylab.scatter(*point[::-1])
			
		pylab.show()

	data = megagreen(frame)
	corr = norm_corr(data, template)
	point = np.unravel_index(np.argmax(corr), corr.shape)
	pointcorr = corr[point]
	#plotit()
	return point[::-1], pointcorr


def get_image_cross(imagefile, template):
	frame = frame_to_numpy(open(imagefile))
	return get_frame_cross(frame, template)

def get_template(templatefile=DEFAULT_TEMPLATE_FILE):
	template = frame_to_numpy(open(templatefile))
	template = megagreen(template)
	return template


def get_file_image_cross(imagefile, templatefile=DEFAULT_TEMPLATE_FILE):
	template = get_template(templatefile)
	return get_image_cross(template, imagefile)

def frame_crosser(templatefile=DEFAULT_TEMPLATE_FILE):
	template = get_template(templatefile)
	def get_cross(frame):
		return get_frame_cross(frame, template)
	return get_cross
	
if __name__ == '__main__':
	print get_file_image_cross(sys.argv[1], sys.argv[2])
