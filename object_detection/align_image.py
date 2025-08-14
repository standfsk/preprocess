import os, argparse
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def ECC(im1, im2):
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	warp_matrix = np.eye(2, 3, dtype=np.float32)
	criteria = (
		cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
		5000,
		1e-6
	)

	(cc, warp_matrix) = cv2.findTransformECC(
		templateImage=im2,
		inputImage=im1,
		warpMatrix=warp_matrix,
		motionType=cv2.MOTION_AFFINE,
		criteria=criteria
	)

	aligned = cv2.warpAffine(im1, warp_matrix, (im2.shape[1], im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	return aligned, warp_matrix

# (ORB) feature based alignment
def featureAlign(im1, im2):
	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(1000)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	matches = sorted(matches, key=lambda x: x.distance)
	numGoodMatches = int(len(matches) * 0.15)
	matches = matches[:numGoodMatches]

	# Draw top matches
	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
	cv2.imwrite("matches.jpg", imMatches)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))

	return im1Reg, h


def align_using_calibration(thermal_img, K_thermal, K_rgb, R, T):
	h, w = thermal_img.shape[:2]

	# Generate pixel grid
	u, v = np.meshgrid(np.arange(w), np.arange(h))
	ones = np.ones_like(u)
	pixels_hom = np.stack((u, v, ones), axis=-1).reshape(-1, 3).T  # shape (3, N)

	# Backproject to 3D (z=1 since depth is unknown, scale ignored)
	rays = np.linalg.inv(K_thermal) @ pixels_hom  # shape (3, N)

	# Transform to RGB camera frame
	rays_rgb = R @ rays + T  # shape (3, N)

	# Project into RGB image
	proj = K_rgb @ rays_rgb
	proj /= proj[2, :]  # normalize by z

	# Reshape to image
	map_x = proj[0, :].reshape(h, w).astype(np.float32)
	map_y = proj[1, :].reshape(h, w).astype(np.float32)

	# Warp thermal into RGB space
	thermal_warped = cv2.remap(thermal_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

	return thermal_warped


def translation(im0, im1):
	# Convert images to grayscale
	im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

	shape = im0.shape
	f0 = fft2(im0)
	f1 = fft2(im1)
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = np.unravel_index(np.argmax(ir), shape)
	if t0 > shape[0] // 2:
		t0 -= shape[0]
	if t1 > shape[1] // 2:
		t1 -= shape[1]
	return [t0, t1]

if __name__ == '__main__':
	# Read the images to be aligned
	im1 = cv2.imread("dataset/a/1-1.jpg")
	im2 = cv2.imread("dataset/a/1-2.jpg")

	im1 = cv2.resize(im1, (640, 480))
	im2 = cv2.resize(im2, (640, 480))

	aligned, warp_matrix = featureAlign(im1, im2)
	cv2.imwrite("reg_image.jpg",
				aligned,
				[cv2.IMWRITE_JPEG_QUALITY, 90])
	print(warp_matrix)
