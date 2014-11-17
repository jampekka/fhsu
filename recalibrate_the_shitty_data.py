import sys
import tables
import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy.interpolate
from se_scenecam import screencapture_to_scenecamera
from undraweyecross import magic_limit, frame_crosser
from numpygst import NumpyGst
from sync_smarteye_video import se_timestamp_fit, get_se_frame_number
import argh


# There's probably some nice dimension-agnostic
# matrix trick to this, but couldn't find it in
# 2 minutes :(
def invert_simple_linear_model((a, b)):
	return np.poly1d((1/a, -b/a))

def pep(videopath, gazepath, start=0.0, end=-1.0, decimate=10.0, nsamples=100,
		retry_coeff=10, latency_in_se=2,
		corr_limit=magic_limit, q_limit=0.5):
	# I already have one filesystem, no need
	# for another crappy one :(
	gazefile, crapstuff = gazepath.rsplit(':', 1)
	#gazedata = tables.openFile(gazefile)
	#table = gazedata.root
	#for part in crapstuff.split('/'):
	#	table = getattr(table, part)
	#
	#gazedata = table[:]
	#print gazedata.dtype.names
	
	#table['video_ts'] = 
	#se_to_vid_ts = np.poly1d(se_timestamp_fit(videopath))
	#vid_ts_to_se = invert_simple_linear_model(se_to_vid_ts)
	gaze = pd.HDFStore(gazefile)[crapstuff]
	gdir = gaze[['g_direction_x', 'g_direction_y', 'g_direction_z']].values
	#gts = se_to_vid_ts(gaze['se_frame_number'])
	
	#gq = gaze[['g_direction_q']].values
	gdirinterp = scipy.interpolate.interp1d(gaze['se_frame_number'], gdir, axis=0)
	gqinterp = scipy.interpolate.interp1d(gaze['se_frame_number'], gaze['g_direction_q'].values, axis=0)
	#gaze = gaze.reindex(se_to_vid_ts(gaze['se_frame_number']))
	#gazeint = gaze.interpolate()


	#points = get_video_crosses(videopath,
	#		start=start, decimate=decimate)
	crosser = frame_crosser()
	fitdata = []
	frames = NumpyGst(videopath)
	
	def loopinject(iterator, callback):
		for value in iterator:
			yield value
			callback()
	
	if end < start:
		end = frames.duration
	def seek_to_random():
		frames.time = start + np.random.rand()*(end-start)
	
	seek_to_random()
	for i, frame in enumerate(loopinject(frames, seek_to_random)):
		if i >= nsamples*retry_coeff:
			raise Exception("Didn't find enough (%i) good frames in (%i) random frames!"%(nsamples, i))
		# TODO: Can't trust the gstreamer timestamps!
		import matplotlib.pyplot as plt

		frameno = get_se_frame_number(frame)
		if frameno is None:
			continue
		frameno -= latency_in_se
		try:
			gazepos = gdirinterp(frameno)
			q = gqinterp(frameno)
		except ValueError:
			continue
		
		frame = frame/255.0
		#plt.imshow(frame)
		#plt.show()
		point, pointcorr = crosser(frame)
		

		if pointcorr < corr_limit:
			continue
		if q <= q_limit: # The q-limit could actually probably be zero
			continue
		se_pix = screencapture_to_scenecamera(*point)
		fitdata.append((gazepos, se_pix))
		if len(fitdata) >= nsamples:
			break
	else:
		raise Exception("Not enough (>=%i) high quality samples"%(nsamples))
	
	gazes, pixels = map(np.array, zip(*fitdata))
	gazes = np.hstack((gazes, np.ones(gazes.shape[0]).reshape(-1, 1)))
	#import matplotlib.pyplot as plt
	#plt.plot(gazes[:,0], pixels[:,0])
	#plt.show()
	
	# Fitting these separatedly. I have a hunch that it's OK.
	# and anyway statsmodel's RLM doesn't support nd response-variables
	#fitter = sm.GLS(pixels, gazes)
	#fit = fitter.fit()
	#print fit.params.T
	yparams = sm.RLM(pixels[:,1], gazes).fit().params
	xparams = sm.RLM(pixels[:,0], gazes).fit().params
	fit = np.vstack((xparams, yparams))

	fitted = np.dot(gazes, fit.T)
	#print fitted.shape
	error = fitted - pixels
	abserr = np.sqrt(np.sum(error**2, axis=1))
	print >>sys.stderr, "Median error in pixels ", np.median(abserr)
	print " ".join(map(str, yparams))
	print " ".join(map(str, xparams))
	
	#for ts, point, pointcorr in points:
	#	print ts, vid_ts_to_se(ts), point, pointcorr
	#gaze['video_ts'] = np.polyval(table['se_frame_number']
	
	
	#
	#print se_to_vidstamp
	

if __name__ == '__main__':
	argh.dispatch_command(pep)
