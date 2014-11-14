"""
Functions for projections to/from the scenecamera/screencapture geometry.

If you aren't doing something stupid, use only these functions:

Get calibration:
calibration = load_scenecamera_calibration(calibration_file_path)

Get "physical" angles (in degrees) from smarteye data
heading, pitch = np.degrees(smarteye_to_angles(smarteye_rec, calibration))

Get "physical" angles (in degrees) from a screencapture video (eg. naksu):
heading, pitch = np.degrees(screencapture_to_angles(x, y)

To get screencapture pixels from "physical" angles (in degrees)
(for eg plotting on images):
x, y = angles_to_scenecamera(np.radians(heading), np.radians(pitch))

Although a bit wrong (as the mapping isn't really linear),
if you want a degree scale for the scenecamera images (do a crop first from
a screencapture!), you can use for plt.imshow
extent=np.degrees(FAKE_SCENECAMERA_ANGLE_EXTENT)

There's a bit of a problem that the calibration is done on the distorted image
and the camera matrix on the undistorted. However, the effect doesn't seem
to be very big before high eccentricities (run the test-function for a graph).



Weird stuff follows (not needed if you use only the functions above):

To understand the weirdness of the functions, one should note that there's
actually (at least) seven different coordinate systems:

1. Distorted screencapture. These are the raw pixel coordinates of the
   raw debutin screencapture video.
2. Distorted scenecamera. The previous coordinates, but corrected with
   the scenecamera window offset. This is the coordinate system that the
   .cal calibration uses, although it shouldn't.
4. Undistorted scenecamera. Scenecamera part of the distorted screencapture
   undistorted with the estimated parameters.
5. Normalized undistorted scenecamera. Coordinates "normalized" with the
   camera intrinsic matrix from undistorted screencapture. Shouldn't be
   used by outside code, mainly used by the distortion
   coefficients (yep, a bit of circularity...) and implicitly by the
   angle conversions.
6. Undistorted view angles. Angular (physical) eccentricity of "objects" in
   the scenecamera, in relation to the scenecamera center (more specifically
   in relation to the "principal point" determined by the intrinsic matrix,
   but should be very close).

7. Subject-specific smarteye directions. The x, y, z -directions used by
   smarteye_to_*-functions. Shouldn't be used as is in any calculations,
   as the coordinate system is mostly arbitrary without the .cal calibration!

8..n. Various "legacy" systems from different fits.

"""
import sys
import itertools

import numpy as np

# Offset of the smarteye scenecam video window in the
# screen capture video
SCENECAMERA_SCREENCAPTURE_OFFSET = (2, 24) # Pixels in x, y
SCENECAMERA_DIMENSIONS = (634-2, 509-24) # This is some weird-ass resolution

SCREENCAPTURE_DIMENSIONS = (684, 569) # Pixels in width, height (x, y)

def screencapture_to_scenecamera(x, y, (dx, dy)=SCENECAMERA_SCREENCAPTURE_OFFSET):
	return x - dx, y - dy

def scenecamera_to_screencapture(x, y, (dx, dy)=SCENECAMERA_SCREENCAPTURE_OFFSET):
	return x + dx, y + dy

def screencapture_to_scenecamera_crop(image):
	ox, oy = SCENECAMERA_SCREENCAPTURE_OFFSET
	w, h = SCENECAMERA_DIMENSIONS
	return image[oy:oy+h,ox:ox+w]

# From Valo12 calibrations, coefficients k1, k2, p1, p2 for
# lens distortion, see
# http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
#SCENECAMERA_LENS_DISTORTION = [-2.6838779690956371e-01,
#				6.4794020897190052e-03,
#   				8.3341611938131735e-04,
#				3.9678304449706790e-02]

#SCENECAMERA_CAMERA_MATRIX = [
#	[6.2907667905219034e+02, 0., 3.2541055473063017e+02],
#	[0., 6.5989057042079526e+02, 2.0382536382905391e+02],
#	[0., 0., 1.]
#	]

# Calibration reprojection error 0.137165 with nine images
#SCENECAMERA_CAMERA_MATRIX = np.array([
#	[ 478.0478033 ,    0.        ,  323.90108559],
#	[   0.        ,  493.84456285,  233.84814632],
#	[   0.        ,    0.        ,    1.        ]
#	])
#SCENECAMERA_LENS_DISTORTION = np.array(
#	[-0.45179574,  0.41459022,  0.        ,  0.        , -0.32955369]
#	)

# Reprojection error 0.148066 with nine images and sixth order
# distortion (k3) fixed to zero. There's some remaining distortion in
# the periphery, but enabling k3 or tangential distortion leads to
# very bad results in the corners
SCENECAMERA_CAMERA_MATRIX =\
np.array([[ 479.8012712 ,    0.        ,  324.02028172],
       [   0.        ,  495.66585046,  233.8912918 ],
       [   0.        ,    0.        ,    1.        ]])
SCENECAMERA_LENS_DISTORTION =\
np.array([-0.41917683,  0.20830564,  0.        ,  0.        ,  0.        ])

# Params (fx, cx), (fy, cy)
SCENECAMERA_CAMERA_PARAMS=((SCENECAMERA_CAMERA_MATRIX[0][0], SCENECAMERA_CAMERA_MATRIX[0][2]),
		(SCENECAMERA_CAMERA_MATRIX[1][1], SCENECAMERA_CAMERA_MATRIX[1][2]))

# TODO: Handles only radial stuff for now.
#There's something weird going on in the tangential stuff.

def normalized_undistort(x0, y0, threshold,
		iter_callback=None, max_iters=100):
	x0 = np.array(x0, dtype=float)
	y0 = np.array(y0, dtype=float)
	x = x0.copy()
	y = y0.copy()
	if x.size == 0: return x, y
	
	for i in range(max_iters):
		# We know that the result of the undistortion
		# should be so that distort(x, y) == x0, y0, so compare to that
		# and "descend" towards the optimum
		# TODO: This overshoots at times?
	
		dx, dy = normalized_distort(x, y)

		ex = dx - x0
		ey = dy - y0

		x -= ex
		y -= ey
		
		if iter_callback is not None:
			iter_callback()
		
		err2 = ex**2 + ey**2
		if np.max(err2) < threshold**2:
			break
	else:
		invalid = (np.sqrt(err2) >= threshold**2) | ~np.isfinite(err2)
		x[invalid] = np.nan
		y[invalid] = np.nan
		#raise ValueError("Undistortion didn't converge in %i steps (error %f)"%(
		#		max_iters, np.max(np.sqrt(err2))))
	return x, y

def scenecamera_undistort(x, y, distortion=SCENECAMERA_LENS_DISTORTION,
		((fx, cx), (fy, cy))=SCENECAMERA_CAMERA_PARAMS,
		dimensions=SCENECAMERA_DIMENSIONS,
		threshold=0.01, strict=True):
	
	x = np.array(x, dtype=float)
	y = np.array(y, dtype=float)
	
	# There are multiple solutions for the undistortion equation
	# and it seems to give wrong answers for (very) out of picture pixels,
	# so check for them here
	
	out_of_image = (x < 0) | (x >= dimensions[0]) | (y < 0) | (y >= dimensions[1])

	x = (x - cx)/fx
	y = (y - cy)/fy

	x0 = x.copy()
	y0 = y.copy()
	
	x, y = normalized_undistort(x, y, threshold/max(fx, fy))

	if strict and np.any(out_of_image):
		if x.shape == ():
			x = np.nan
			y = np.nan
		else:
			x[out_of_image] = np.nan
			y[out_of_image] = np.nan
	
	x *= fx
	x += cx

	y *= fy
	y += cy

	return x, y

def normalized_distort(x, y, (k1, k2, p1, p2, k3)=SCENECAMERA_LENS_DISTORTION):
	# Tangential stuff disabled. Something
	# weird going on there
	if p1 != 0 or p2 != 0:
		raise NotImplemented("Tangential distortion not implemented")
	if k3 != 0:
		raise NotImplemented("Sixth degree radial distortion (k3) not implemented")

	x2 = x**2
	y2 = y**2
	r2 = x2 + y2
	xy = x*y
	radial = (1 + k1*r2 + k2*r2**2) #+ k3*r2**3)
	x_new = x*radial #+ 2*p2*xy + p1*(r2 + 2*x2)
	y_new = y*radial #+ 2*p1*xy + p2*(r2 + 2*y2)
	return x_new, y_new


def scenecamera_distort(x, y, distortion=SCENECAMERA_LENS_DISTORTION,
		((fx, cx), (fy, cy))=SCENECAMERA_CAMERA_PARAMS):
	x = (x - cx)/fx
	y = (y - cy)/fy

	x_new, y_new = normalized_distort(x, y, distortion)
	
	x_new *= fx
	x_new += cx

	y_new *= fy
	y_new += cy

	return x_new, y_new


def undistorted_scenecamera_heading(x, (fx, cx)=SCENECAMERA_CAMERA_PARAMS[0]):
	return np.arctan((x - cx)/fx)

def undistorted_scenecamera_pitch(y, (fy, cy)=SCENECAMERA_CAMERA_PARAMS[1]):
	# Change the coordinates so that positive pitch is up (as opposed
	# to the pixels where upper left is zero)
	return -np.arctan((y - cy)/fy)
	
def undistorted_scenecamera_to_angles(x, y):
	return undistorted_scenecamera_heading(x), undistorted_scenecamera_pitch(y)

def distorted_scenecamera_to_angles(x, y, strict=True):
	x, y = scenecamera_undistort(x, y, strict=strict)
	return undistorted_scenecamera_heading(x), undistorted_scenecamera_pitch(y)

def distorted_screencapture_to_angles(x, y, strict=True):
	return distorted_scenecamera_to_angles(*screencapture_to_scenecamera(x, y),
		strict=strict)

screencapture_to_angles = distorted_screencapture_to_angles

def heading_to_undistorted_scenecamera(heading, (fx, cx)=SCENECAMERA_CAMERA_PARAMS[0]):
	return np.tan(heading)*fx + cx

def pitch_to_undistorted_scenecamera(pitch, (fy, cy)=SCENECAMERA_CAMERA_PARAMS[1]):
	return np.tan(-pitch)*fy + cy

def angles_to_undistorted_scenecamera(heading, pitch):
	return heading_to_undistorted_scenecamera(heading), pitch_to_undistorted_scenecamera(pitch)

def angles_to_distorted_scenecamera(heading, pitch):
	return scenecamera_distort(*angles_to_undistorted_scenecamera(heading, pitch))

def angles_to_screencapture(heading, pitch):
	return screencapture_to_scenecamera(angles_to_distorted_scenecamera(heading, pitch))

def load_scenecamera_calibration(filepath):
	return np.genfromtxt(filepath)

def smarteye_to_distorted_scenecamera(data, calibration):
	data = np.vstack((data['g_direction_x'],
		data['g_direction_y'],
		data['g_direction_z'],
		np.ones(len(data))))
	
	x, y = np.dot(calibration, data)
	return x, y

def smarteye_to_undistorted_angles(data, calibration):
    x, y = smarteye_to_distorted_scenecamera(data, calibration)
    return distorted_scenecamera_to_angles(x, y)

smarteye_to_angles = smarteye_to_undistorted_angles

FAKE_SCENECAMERA_ANGLE_EXTENT=[
	distorted_scenecamera_to_angles(0, SCENECAMERA_CAMERA_MATRIX[1][2]-1)[0],
	distorted_scenecamera_to_angles(SCENECAMERA_DIMENSIONS[0]-1,
		SCENECAMERA_CAMERA_MATRIX[1][2]-1)[0],
	distorted_scenecamera_to_angles(SCENECAMERA_CAMERA_MATRIX[1][2], 0)[1],
	distorted_scenecamera_to_angles(SCENECAMERA_CAMERA_MATRIX[1][2],
		SCENECAMERA_DIMENSIONS[1]-1)[1]
	]

def distorted_scenecamera_image_to_angle_image(image):
	# TODO: The mapping should be cached (and optimized)
	import cv2
	w, h = SCENECAMERA_DIMENSIONS
	
	cx, cy = zip(*itertools.product(*zip([0, 0], [w-1, h-1])))

	ch, cp = distorted_scenecamera_to_angles(cx, cy)
	hs, he = np.min(ch), np.max(ch)
	ps, pe = np.min(cp), np.max(cp)

	P, H = np.mgrid[0:h:1.0, 0:w:1.0]

	H /= w/(he - hs)
	H += hs
	
	P /= h/(pe - ps)
	P += ps
	
	
	x, y = angles_to_distorted_scenecamera(H.flatten(), P.flatten())
	
	X = x.reshape((h, w)).astype(np.float32)
	Y = y.reshape((h, w)).astype(np.float32)
	dst = cv2.remap(image, X, Y, cv2.INTER_CUBIC)
	
	return dst[::-1], [hs, he, ps, pe]

def screencapture_image_to_angle_image(img):
	return distorted_scenecamera_image_to_angle_image(
		screencapture_to_scenecamera_crop(img))

def estimate_scenecamera_parameters(images):
	from estimate_camera_parameters import estimate_camera_parameters, plot_result
	crop = (SCENECAMERA_SCREENCAPTURE_OFFSET, SCENECAMERA_DIMENSIONS)
	camera_matrix, distortion, error = estimate_camera_parameters(crop,
		images, result_callback=plot_result)
	print("# Reprojection error %f"%error)
	print("SCENECAMERA_CAMERA_MATRIX =\\\n%s"%repr(camera_matrix))
	print("SCENECAMERA_LENS_DISTORTION =\\\n%s"%repr(distortion))

	
	


def test_undistortion_convergence():
	import matplotlib.pyplot as plt
	#ux, uy = (0, SCENECAMERA_CAMERA_PARAMS[1][1])

	#ux, uy = (-300, -300) # Somewhere between -200 and -300 there's
				# a "fake minima"
	ux, uy = (0, SCENECAMERA_DIMENSIONS[1]-1)
	#ux, uy = (0, 480)
	((fx, cx), (fy, cy))=SCENECAMERA_CAMERA_PARAMS
	ux = (ux - cx)/fx
	uy = (uy - cy)/fy

	x0, y0 = normalized_distort(ux, uy)
	
	def error_func(x, y, x0, y0):
		x, y = normalized_distort(x, y)
		return np.sqrt((x - x0)**2 + (y - y0)**2)
	
	w = 1000/fx
	h = 1000/fy
	X, Y = np.mgrid[(x0-w):(x0+w):1/fx,(y0-h):(y0+h):1/fy]
	
	#rng = np.arange(x0-100/fx, x0+100/fx, 0.1)

	err = error_func(X.flatten(), Y.flatten(), x0, y0)
	err = err.reshape(X.shape)

	plt.contour(X*fx + cx, Y*fy + cy, np.log(err))

	plt.plot(ux*fx + cx, uy*fy + cy, 'go')
	plt.plot(x0*fx + cx, y0*fy + cy, 'ro')
	
	import inspect
	guesses = []
	def cb():
		# OMG!
		parent = inspect.getouterframes(inspect.currentframe())[1][0]
		p = parent.f_locals
		guesses.append(map(float, (p['x'], p['y'])))

	try:
		x, y = normalized_undistort(x0, y0, 0.01/max(fx, fy), iter_callback=cb)
	except ValueError:
		print >>sys.stderr, "No convergence!"
	
	gx, gy = map(np.array, zip(*guesses))
	gx = gx*fx + cx
	gy = gy*fy + cy
	
	plt.plot(gx, gy, 'b.-')

	#plt.plot(rng*fx + cx, error_func(rng, x0)*fx)
	#plt.axvline(x0*fx + cx)
	#plt.axhline(0)
	plt.show()

	
def test():
	import matplotlib.pyplot as plt
	
	pp_x = SCENECAMERA_CAMERA_PARAMS[0][1]
	pp_y = SCENECAMERA_CAMERA_PARAMS[1][1]
	ox, oy = SCENECAMERA_SCREENCAPTURE_OFFSET
	sc_w, sc_h = SCENECAMERA_DIMENSIONS


	sx = np.linspace(ox, sc_w-1, 100)
	sy = np.linspace(oy, sc_h-1, 100)

	x, y = screencapture_to_scenecamera(sx, sy)
	#x = np.arange(0, SCREENCAPTURE_DIMENSIONS[0])
	#y = np.arange(0, SCREENCAPTURE_DIMENSIONS[0])
	#y = np.array([SCENECAMERA_CAMERA_PARAMS[1][1]]*len(x))
	
	# The old linear stuff
	px2heading = np.poly1d((0.11786904561521629, -37.482356505638776))
	heading2px = np.poly1d((1.0/px2heading[1], -px2heading[0]/px2heading[1]))
	py2pitch = np.poly1d(( -0.11786904561521629, 29.46726140380407))
	pitch2py = np.poly1d((1.0/py2pitch[1], -py2pitch[0]/py2pitch[1]))
	
	heading, pitch = distorted_screencapture_to_angles(sx, sy)
	
	dheading, dpitch = undistorted_scenecamera_to_angles(x, y)

	plt.subplot(2,1,1)
	h = np.degrees(dheading)
	plt.plot(y, np.degrees(heading), label="Real")
	plt.plot(y, np.degrees(dheading), label="Non-undistorted")
	plt.plot(y, px2heading(x), label="Old linear approx")
	plt.axhline(0, color='black')
	plt.ylabel("Heading (degrees)")
	plt.xlabel("x (pixels)")

	plt.subplot(2,1,2)
	p = np.degrees(dpitch)
	plt.plot(y, np.degrees(pitch), label="Real")
	plt.plot(y, np.degrees(dpitch), label="Non-undistorted")
	plt.plot(y, py2pitch(y), label="Old linear approx")
	plt.axhline(0, color='black')
	plt.ylabel("Pitch (degrees)")
	plt.xlabel("y (pixels)")
	plt.legend()
	
	plt.figure()


	plt.scatter(pp_x, pp_y)
	
	#x_rng = np.linspace(pp_x-200, pp_x+200, 10)

	#y_rng = np.linspace(pp_y-200, pp_y+200, 10)

	#x = np.hstack( ([x_rng[0]]*len(y_rng), x_rng, [x_rng[-1]]*len(y_rng), x_rng[::-1]) )
	#y = np.hstack( (y_rng[::-1], [y_rng[0]]*len(x_rng), y_rng, [y_rng[-1]]*len(x_rng)) )
	dX, dY = np.mgrid[0:SCENECAMERA_DIMENSIONS[0]:10,
		0:SCENECAMERA_DIMENSIONS[1]:10]
	dx = dX.flatten()
	dy = dY.flatten()

	
	#plt.plot(x, y, '.', label="Original")

	#dx, dy = scenecamera_distort(x, y)

	#plt.plot(dx, dy, '.', label="Distorted")
	ux, uy = scenecamera_undistort(dx, dy)

	invalid = np.isnan(ux) | np.isnan(uy)
	#print "INVALID", dx[invalid], dy[invalid]
	#assert not (np.any(~np.isfinite(ux)) or np.any(~np.isfinite(ux)))
	edges = (dx == np.min(dx)) | (dy == np.min(dy)) | (dx == np.max(dx)) | (dy == np.max(dy))
	plt.plot(ux.reshape(dX.shape), uy.reshape(dY.shape), color='black')
	plt.plot(ux.reshape(dX.shape).T, uy.reshape(dY.shape).T, color='black')
	
	#plt.plot(dx[edges], dy[edges], '.', label="Distorted")
	
	#plt.gca().add_patch(plt.Rectangle((0, 0), sc_w, sc_h))
	
	#plt.legend()
	plt.gca().invert_yaxis()
	
	plt.figure()
	X, Y = np.mgrid[0:SCENECAMERA_DIMENSIONS[0]:10,
		0:SCENECAMERA_DIMENSIONS[1]:10]
	#x[0] = np.nan
	#x[-1] = np.nan
	#y[0] = np.nan
	#y[-1] = np.nan
	x = X.flatten()
	y = Y.flatten()
	
	dx, dy = scenecamera_distort(x, y)
	plt.plot(dx.reshape(X.shape), dy.reshape(Y.shape), color='black')
	plt.plot(dx.reshape(X.shape).T, dy.reshape(Y.shape).T, color='black')
	#plt.legend()

	plt.gca().invert_yaxis()

	plt.show()

if __name__ == '__main__':
	test()
	#test_undistortion_convergence()
	#estimate_scenecamera_parameters(sys.argv[1:])
