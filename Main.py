import numpy as np
import argparse
import imutils
import glob
import cv2
def find_Obj():
	image = cv2.imread('./image/objs.png')
	cv2.imshow('',image)
	cv2.waitKey(0)
	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	# image = cv2.imread('./image/objs.png')
	# cv2.imshow('0 - Original Image', image)
	# cv2.waitKey(0)

	# Create a black image with same dimensions as our loaded image(0->Black)
	blank_image = np.zeros((image.shape[0], image.shape[1], 3))

	# Create a copy of our original image
	orginal_image = image
	copy_orginal=image

	# Grayscale our image
	# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# Find Canny edges
	# edged = cv2.Canny(gray, 50, 200)
	cv2.imshow('1 - Canny Edges', edged)
	cv2.waitKey(0)

	# Find contours and print how many were found
	cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	def x_cord_contour(cnts):
		# peri = cv2.arcLength(cnts, True)
		# approx = cv2.approxPolyDP(cnts, 0.015 * peri, True)
		#Returns the X cordinate for the contour centroid
		if cv2.contourArea(cnts) > 10:
			M = cv2.moments(cnts)
			return (int(M['m10']/M['m00']))
		else:
			pass


	def label_contour_center(image, c):
		# Places a red circle on the centers of contours
		M = cv2.moments(c)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])

		# Draw the countour number on the image
		cv2.circle(image,(cx,cy), 10, (0,0,255), -1)
		return image


	# Load our image
	image = cv2.imread('./image/objs.png')
	orginal_image = image.copy()


	# Computer Center of Mass or centroids and draw them on our image
	# for (i, c) in enumerate(cnts):
		# orig = label_contour_center(image, c)

	cv2.imshow("Contour Centers ", image)
	cv2.waitKey(0)

	# Sort by left to right using our x_cord_contour function
	# contours_left_to_right = sorted(cnts, key = x_cord_contour, reverse = False)


	# Labeling Contours left to right
	for (i,c)  in enumerate(cnts):
	#     copy_orignal=orignal_image.copy()
		cv2.drawContours(orginal_image, [c], -1, (0,0,255), 3)  
		# M = cv2.moments(c)
		# cx = int(M['m10'] / M['m00'])
		# cy = int(M['m01'] / M['m00'])
	#     cv2.putText(orginal_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
		# cv2.imshow('Left to Right Contour', orginal_image)
		# cv2.waitKey(0)
		(x, y, w, h) = cv2.boundingRect(c)  
	#     return bounded rectangle

		# Let's now crop each contour and save these images
		cropped_contour = copy_orginal[y:y + h, x:x + w]
		image_name = "obj_number_" + str(i+1) + ".png"
		print (image_name)
		cv2.imwrite("./Objects./"+image_name, cropped_contour)
		cv2.namedWindow('Object')
		cv2.imshow('Object',cropped_contour)
		cv2.waitKey(1000)
		cv2.destroyWindow('Object')

	cv2.destroyAllWindows()

def Main():


	image = cv2.imread('./image/target_image.png')
	# cv2.imshow("Source Image", image)

	# cv2.waitKey(0)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# loop over the images to find the template in
	for imagePath in glob.glob('./objects' + "/*.png"):
	# load the image, convert it to grayscale, and initialize the
		# bookkeeping variable to keep track of the matched region

		template = cv2.imread(imagePath)
		# print(template)
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template = cv2.Canny(template, 50, 200)
		(tH, tW) = template.shape[:2]
		# cv2.imshow("Template", template)
		# cv2.waitKey(0)

		found = None
		# loop over the scales of the image
		for scale in np.linspace(0.2, 1.0, 20)[::-1]:
			# resize the image according to the scale, and keep track
			# of the ratio of the resizing
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])
			# if the resized image is smaller than the template, then break
			# from the loop
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break
					# detect edges in the resized, grayscale image and apply template
			# matching to find the template in the image
			edged = cv2.Canny(resized, 50, 200)
			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
			# check to see if the iteration should be visualized
			if (True):
				# draw a bounding box around the detected region
				clone = np.dstack([edged, edged, edged])
				cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
					(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
				# cv2.imshow("Visualize", clone)
				# cv2.waitKey(0)
			# if we have found a new maximum correlation value, then update
			# the bookkeeping variable
			if found is None or maxVal > found[0]:
				found = (maxVal, maxLoc, r)
		# unpack the bookkeeping variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
			(_, maxLoc, r) = found
			(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
			(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
			# draw a bounding box around the detected result and display the image
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			
	cv2.imshow("Image", image)
	cv2.imwrite("./Result.png", image)
	Objects= cv2.imread('./image/objs.png')
	cv2.imshow("Objects",Objects)
	cv2.waitKey(0)	

find_Obj()
Main()