import cv2
import numpy as np
from bound import BoundingBox
from note import Note

import glob
import imutils
from matplotlib import pyplot as plt


def get_staffs(img, verbose=False):
	"""
	It takes image as input and find staffs
	:param img:
	:param verbose:
	:return: staff positions
	"""

	img_copy = np.copy(img)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bw = cv2.bitwise_not(img_gray)
	thresh = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 2))   
	dilation = cv2.dilate(thresh, kernel, iterations=10)
	# get contours
	_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# for each contour create bounding rectangle
	bounding_boxes = []
	for contour in contours:
		(x, y, w, h) = cv2.boundingRect(contour)
		box = BoundingBox(x, y, w, h)
		bounding_boxes.append(box)

	boxes_sorted = sorted(bounding_boxes, key=lambda x: x.w, reverse=True)
	# threshold width
	tw = boxes_sorted[0].w - 100
	# threshold height
	th = boxes_sorted[0].h - 10

	staffs = []
	for box in bounding_boxes:
		# discard rectangles that are smaller than threshold
		if box.w < tw or box.h < th:
			continue

		staffs.append(box)
		cv2.rectangle(img_copy, box.pt1, box.pt2, (255, 0, 0), 1)

	if verbose:
		# display image with contours
		cv2.imshow('Staffs detection', img_copy)
	
	return list(reversed(staffs))


def remove_staves(img, staffs, verbose=False):
	"""
	It takes image as input and returns image without staves and image only with staves
	:param img:
	:param verbose:
	:return:
	"""

	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply adaptiveThreshold at the bitwise_not of gray
	img_gray = cv2.bitwise_not(img_gray)
	bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

	staves = []
	i = 1
	for staff in staffs:
		# Get cropped image
		crop_img = bw[staff.pt1[1]:staff.pt2[1], staff.pt1[0]:staff.pt2[0]]

		# Create the images that will use to extract the horizontal and vertical lines
		horizontal = np.copy(crop_img)
		vertical = np.copy(crop_img)

		# Specify size on horizontal axis
		cols = horizontal.shape[1]
		horizontal_size = cols // 30

		# Create structure element for extracting horizontal lines through morphology operations
		horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
		horizontal = cv2.erode(horizontal, horizontal_structure)
		horizontal = cv2.dilate(horizontal, horizontal_structure)
		staves.append(horizontal)

		# Specify size on vertical axis
		rows = vertical.shape[0]
		vertical_size = rows // 30

		# Create structure element for extracting vertical lines through morphology operations
		vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
		vertical = cv2.erode(vertical, vertical_structure)
		vertical = cv2.dilate(vertical, vertical_structure)

		vertical = cv2.bitwise_not(vertical)

		edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

		kernel = np.ones((2, 2), np.uint8)
		edges = cv2.dilate(edges, kernel)
		smooth = np.copy(vertical)

		smooth = cv2.blur(smooth, (2, 2))

		(rows, cols) = np.where(edges != 0)
		vertical[rows, cols] = smooth[rows, cols]

		if verbose:
			cv2.imshow('Stave' + str(i), horizontal)
			i += 1

	if verbose:
		cv2.imshow('original', img)
		# cv2.imshow('gray', gray)
		# cv2.imshow('edges', edges)
		# cv2.imshow('vertical', vertical)
		# cv2.imshow('horizontal', horizontal)

	# cv2.imwrite('smooth.png', vertical)
	return staves


def detect_lines(img, staves, staff_positions, verbose=False):
	"""
	It takes original image, image only with staves, staff positions as input and list of tuples:
	first element is index of line on original image and second element is sum per line
	:param img:
	:param lines_img:
	:param staff_positions:
	:param verbose:
	:return:
	"""

	img_copy = np.copy(img)
	lines_pos = []
	for stave, staff in zip(staves, staff_positions):
		lines = list((ind + staff.y, val) for ind, val in enumerate(stave.sum(axis=1)))
		lines = sorted(lines, key=lambda x: x[1], reverse=True)[:5]
		lines_pos.append([l[0] for l in lines])

	if (verbose):
		points = []
		for lines in lines_pos:
			for line in lines:
				points.append([0, line])
				points.append([img_copy.shape[1], line])
		points = np.array(points).reshape((-1, 2, 2))
		cv2.polylines(img_copy, points, False, (0, 0, 255), 1)
		cv2.imshow('Detected lines', img_copy)

	return lines_pos


def fit_matching(img, templates, threshold, verbose=False):
	"""

	:param img:
	:param templates:
	:param threshold:
	:return:
	"""

	best_positions_count = -1
	best_positions = []
	best_scale = 1
	w = h = 0

	for scale in [i / 100.0 for i in range(30, 300, 10)]:
		positions_count = 0
		positions = []
		for template in templates:
			template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
			matching = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
			matching = np.where(matching >= threshold)
			positions_count += len(matching[0])
			positions += map(list, zip(*matching[::-1]))

			if verbose:
				print("Scale: {}, detected objects: {}".format(scale, positions_count))

			if positions_count > best_positions_count:
				best_positions_count = positions_count
				best_positions = positions
				best_scale = scale
				w, h = template.shape[::-1]
			elif positions_count < best_positions_count:
				pass
	if verbose:
		for pt in best_positions:
			cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
		cv2.imshow('Matching', img)
	return best_positions, best_scale


def detect(img, staffs, templates, threshold, verbose=False, color=(0, 255, 0)):
	"""
	Detect objects using templates and return objects positions.
	:param img:
	:param staffs:
	:param templates:
	:param threshold:
	:param verbose:
	:return:
	"""

	img_copy = np.copy(img)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	matching_results = []
	for staff in staffs:
		crop_img = img_gray[staff.pt1[1]:staff.pt2[1], staff.pt1[0]:staff.pt2[0]]
		positions, scale = fit_matching(crop_img, templates, threshold)

		tw, th = templates[0].shape[::-1]
		tw *= scale
		th *= scale

		bounding_boxes = []
		for i in range(0, len(positions)):
			x = positions[i][0] + staff.x
			y = positions[i][1] + staff.y
			box = BoundingBox(x, y, tw, th)
			bounding_boxes.append(box)

		matching_results.append(np.copy(bounding_boxes))

	if verbose:
		for boxes in matching_results:
			for box in boxes:
				box.draw(img_copy, color, 1)
		cv2.imshow("Detection", img_copy)
		cv2.waitKey(0)

	return matching_results


def merge_boxes(bounding_boxes, threshold):
	"""
	Merge bounding boxes that overlap each over
	"""

	filtered_boxes = []
	bounding_boxes = bounding_boxes.tolist()
	while len(bounding_boxes) > 0:
		box = bounding_boxes.pop(0)
		bounding_boxes.sort(key=lambda b: b.get_distance(box))
		is_merging = True
		while is_merging:
			is_merging = False
			i = 0
			for _ in range(len(bounding_boxes)):
				if box.get_overlap_ratio(bounding_boxes[i]) > threshold or bounding_boxes[i].get_overlap_ratio(box):
					box = box.merge(bounding_boxes.pop(i))
					is_merging = True
				elif bounding_boxes[i].get_distance(box) > box.w/2 + bounding_boxes[i].w/2:
					break
				else:
					i += 1
		filtered_boxes.append(box)

	return filtered_boxes


def get_notes_pitches(lines_pos, notes_boxes):
	"""

	:param img:
	:param staffs:
	:param lines_pos:
	:param notes_pos:
	:return:
	"""

	notes = []
	gap_height = (lines_pos[1] - lines_pos[0]) / 2
	middle = lines_pos[1] + gap_height
	for note_box in notes_boxes:
		note_ind = int((note_box.middle[1] - middle) / gap_height)
		label = Note.NOTES[note_ind][0]
		pitch = Note.NOTES[note_ind][1]
		notes.append(Note(label, pitch, note_box))

	return notes