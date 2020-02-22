import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw
import PIL
import math
from math import sqrt

img=Image.open('resimOrjinal.png')


def bilinearResize(img, x, y):
	im = img.resize((x,y), Image.BILINEAR)
	im.save('bilinear.png')
bilinearResize(img,1500,1500)

def neighbourResize(img,x,y):
	im = img.resize((x,y), Image.NEAREST)
	im.save('nearest.png')
neighbourResize(img,1500,1500)





def boxFilter(input_image):
	input_pixels = input_image.load()
	box_kernel = [[1 / 9, 1 / 9, 1 / 9],
		      [1 / 9, 1 / 9, 1 / 9],
		      [1 / 9, 1 / 9, 1 / 9]]

	kernel = box_kernel

	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution between intensity and kernels
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('boxFilter.png')
boxFilter(img)


def convolve(img):
	input_image = img
	input_pixels = input_image.load()
	box_kernel = [[1 / 9, 1 / 9, 1 / 9],
		      [1 / 9, 1 / 9, 1 / 9],
		      [1 / 9, 1 / 9, 1 / 9]]

	# Select kernel here:
	kernel = box_kernel

	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution between intensity and kernels
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('convolve.png')	

convolve(img)


def highPassFilter(img):
	input_image = img
	input_pixels = input_image.load()

	# High-pass kernel
	kernel = [[  0  , -.5 ,    0 ],
		  [-.5 ,   3  , -.5 ],
		  [  0  , -.5 ,    0 ]]

	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution with kernel
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('highPassFilter.png')	
highPassFilter(img)


def sharpenFilter(img):
	input_image = img
	input_pixels = input_image.load()

	# High-pass kernel
	kernel = [[  0  , -.1 ,    0 ],
		  [-1 ,   5  , -1 ],
		  [  0  , -1 ,    0 ]]

	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution with kernel
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('sharpenFilter.png')
sharpenFilter(img)


def embossFilter(img):
	input_image = img
	input_pixels = input_image.load()

	# High-pass kernel
	kernel = [[-2, -1, 0],
		  [-1, 1, 1],
		  [0, 1, 2]]

	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution with kernel
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('embossFilter.png')
embossFilter(img)


def gaussianFilter(img):
	input_image = img
	input_pixels = input_image.load()

	# Gaussian kernel
	kernel = [[1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256],
		  [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
		  [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
		  [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
		  [1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256]]


	# Middle of the kernel
	offset = len(kernel) // 2

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	# Compute convolution between intensity and kernels
	for x in range(offset, input_image.width - offset):
		for y in range(offset, input_image.height - offset):
			acc = [0, 0, 0]
			for a in range(len(kernel)):
				for b in range(len(kernel)):
					xn = x + a - offset
					yn = y + b - offset
					pixel = input_pixels[xn, yn]
					acc[0] += pixel[0] * kernel[a][b]
					acc[1] += pixel[1] * kernel[a][b]
					acc[2] += pixel[2] * kernel[a][b]

			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
	    
	output_image.save('gaussianFilter.png')
gaussianFilter(img)



def makeSobel(img):
	input_image = img
	input_pixels = input_image.load()

	# Calculate pixel intensity as the average of red, green and blue colors.
	intensity = [[sum(input_pixels[x, y]) / 3 for y in range(input_image.height)] for x in range(input_image.width)]

	# Sobel kernels
	kernelx = [[-1, 0, 1],
		   [-2, 0, 2],
		   [-1, 0, 1]]
	kernely = [[-1, -2, -1],
		   [0, 0, 0],
		   [1, 2, 1]]

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

# Compute convolution between intensity and kernels
	for x in range(1, input_image.width - 1):
		for y in range(1, input_image.height - 1):
			magx, magy = 0, 0
			for a in range(3):
				for b in range(3):
					xn = x + a - 1
					yn = y + b - 1
					magx += intensity[xn][yn] * kernelx[a][b]
					magy += intensity[xn][yn] * kernely[a][b]

        # Draw in black and white the magnitude
			color = int(sqrt(magx**2 + magy**2))
			draw.point((x, y), (color, color, color))
    
	output_image.save('Sobel.png')

makeSobel(img)


