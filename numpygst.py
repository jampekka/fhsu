import numpy as np
import gst
import gobject
import logging

logger = logging.getLogger("numpygst")
gobject.threads_init()

def log_messages(*args):
	print >>sys.stderr, args
	pass

class NumpyGstError(Exception):
	pass

GST_FRAME_FORMAT="video/x-raw-rgb, bpp=(int)24, depth=(int)24, endianness=(int)4321, red_mask=(int)16711680, green_mask=(int)65280, blue_mask=(int)255"

class NumpyFrame(np.ndarray):
	def __new__(cls, frame, timestamp):
		obj = np.asarray(frame).view(cls)
		obj.timestamp = timestamp
		return obj
	
	def __array_finalize__(self, obj):
		if obj is None: return
		self.timestamp = getattr(obj, 'timestamp', None)

def gst_to_npy(frame):
	npy = np.frombuffer(frame.data, dtype=np.uint8)
	npy = np.copy(npy)
	caps = frame.get_caps()[0]
	shape = (caps['height'], caps['width'], 3)
	npy = npy.reshape(shape)
	# TODO: This isn't the frame's actual timestamp
	#	but the exact(ish) time we seeked into
	npy = NumpyFrame(npy, frame.timestamp/float(gst.SECOND))
	return npy

class NumpyGst(object):
	def __init__(self, path, message_listener=log_messages):
		self._error = None
		#gobject.threads_init()
		self.path = path
		self.pipeline = gst.parse_launch("""
		filesrc name=src ! decodebin name=dec sync=true !
			ffmpegcolorspace !
			%s !
			appsink max-buffers=1 drop=false sync=true name=sink
		"""%(GST_FRAME_FORMAT))
		
		self.pipeline.get_bus().connect("sync-message::error", self._sync_handler)
		self.pipeline.get_bus().enable_sync_message_emission()

		self.src = self.pipeline.get_by_name("src")
		self.sink = self.pipeline.get_by_name("sink")
		self.src.set_property("location", self.path)
	
		self._check_error()
		self.pipeline.get_state()
		self.pipeline.set_state(gst.STATE_PAUSED)
		self.pipeline.get_state()
		self._check_error()
		self.pipeline.set_state(gst.STATE_PLAYING)

		if self.sink.emit("pull-preroll") is None:
			raise NumpyGstError("Unable to pull preroll")
		self.frame = None

	def _sync_handler(self, bus, msg):
		self._error = msg.parse_error()[1]

	def _check_error(self):
		if self._error:
			raise NumpyGstError(self._error)



	def current(self):
		return self.frame

	def next(self):
		self._check_error()
		frame = self.sink.emit("pull-buffer")
		if not frame:
			raise StopIteration
		self.frame = gst_to_npy(frame)
		return self.frame
	def __iter__(self):
		return self
	
	@property
	def time(self):
		# TODO: We shouldn't use this.
		# read the stream time instead
		if self.frame is not None:
			return self.frame.timestamp
		else:
			return 0.0
	
	@time.setter
	def time(self, time):
		time = time*gst.SECOND
		self.pipeline.seek_simple(gst.FORMAT_TIME,
					gst.SEEK_FLAG_FLUSH,
					int(time))
	
	@property
	def duration(self):
		return self.pipeline.query_duration(gst.FORMAT_TIME)[0]/float(gst.SECOND)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import sys
	import cv
	
	video = NumpyGst(sys.argv[1])
	video.seek(100)

	for frame in video:
		# OpenCV seems to do BGR instead of RGB
		# NOTE: There seems to be a bug in cv.fromarray
		#	that causes memory corruption without the
		#	copy
		cvframe = frame[:,:,::-1].copy()
		cvframe = cv.fromarray(cvframe)
		cv.ShowImage("image", cvframe)
		cv.WaitKey(1)
		
