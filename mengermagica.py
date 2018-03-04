#!/usr/local/bin/python3

# pyvox module to export .vox file
from pyvox.models import Vox as vx
from pyvox.writer import VoxWriter as vw

# deepcopy is used when subdividing by nested concatenation.
from copy import deepcopy as dp

# pprint isn't called, but it's useful for debugging.
from pprint import pprint

import numpy as np

# ______________________________________________________
# ______________________________________________________
# This method is loosely based on
# http://www.fundza.com/algorithmic/menger/
#
# Tested on https://chrmoritz.github.io/Troxel/
# And on https://ephtracy.github.io
#
# Created by Adi Mathew on 12/31/2017
# Goodbye 2017~
# ______________________________________________________
# ______________________________________________________


class MengerMagica:
	"""
	A class that handles creation of a 3 dimensional voxel array to be
	displayed in MagicaVoxel by @ephtracy.
	Use the output or inverseOutput method to create files/start processing.
	"""

	def __init__(self, isColor = False):
		# ---------
		# Boolean indicating whether to assign colors to each voxel
		# If mono, each voxel is indicated by a 1 or 0.
		# Note that the magicavoxel file format only recognizes a given
		# voxels palette index. We use the included (slightly modified pyvox)
		# to assign rgb to each voxel, which is then added to a palette/index.
		self.__monoroot = 1
		self.__colorroot = [81, 168, 221]
		self.__iscolor = isColor

		# The starting voxel. Needs to be 3D for pyvox
		self.__base = [[[ self.__colorroot if isColor else self.__monoroot ]]]
		# The (x,y,z) indices to be removed based on the lut base indices below
		# pimaryLUT is populated after the first subdiv in generateLUT
		self.__primaryLUT = []
		# The following prevent recalculation. (haven't tested...)
		self.__lastRunDivisions = None
		self.__lastRunOutput = None
		# Use this to visualize all deleted voxels. Returned by inverseOutput()
		self.__outputInverse = None
		# ---------

		# The indices of a first level Menger cube which are to be removed
		# Try changing these for some interesting models.
		self.lut = [4, 10, 12, 13, 14, 16, 22]

	# ______________________________________________________
	def __subdiv(self, element):
		"""
		This function takes an input multidimensional list.
		It then duplicates the inputs component rows/cols/depths twice along
		each axis to essentially triple the volume, resulting in a voxel cube.
		This output cube is built of 27 input elements.
		"""
		cube = tmp = dp(element)

		colorizer = None
		if self.__iscolor:
			# Blanket adds a number depending on level to r,g,b
			level = int(np.cbrt(tmp.shape[0]))
			colorizer = np.vectorize(lambda t: 0 if t == 0 else t + (9 * level))
		else:
			# Uses the palette index for color assignment
			# Disable this if you want an all-white (first palette index) cube
			# colorizer = np.vectorize(lambda t: 0 if t == 0 else t + 1)
			pass

		tmp = colorizer(tmp) if colorizer else tmp

		for i in range(3):
			for j in range(2):
				cube = np.concatenate((cube,tmp), axis= 2 - i)

			tmp=dp(cube)
		return cube

	# ______________________________________________________
	def __generateLUT(self, ndcube):
		"""
		This function takes a n-dimensional numpy array of voxels.
		If the input cube is of level 1 subdivision, i.e 3 voxels/side, then
		we save the indices to the global primary var.
		Otherwise, this function gets the relative level above 1 and uses the level
		to create and return a list of (x,y,z) tuples whose voxels will be
		removed.

		Changes in global lut > changes primary > changes completeLUT.
		"""
		completeLUT = []

		# A constant that is one power less than the lenght of each side of cube
		# It is used for generating indices of holes for the LUT.
		# (Since holes are technically coming from one iteration down)
		factor = int(ndcube.shape[0]/3)

		if factor == 1:
			ctr = 0
			for idx, e in np.ndenumerate(ndcube):
				if (ctr/3 if self.__iscolor else ctr) in self.lut:
					if self.__iscolor:
						(x, y, z, _) = idx
						completeLUT.append((x,y,z))
					else:
						completeLUT.append(idx)
				ctr += 1
			self.__primaryLUT = completeLUT
		else:
			# This branch creates ranges of tuples from low to hi.
			# i.e "Generate indices from (0,0,0) to (3,3,3) non inclusive."
			lows = []
			his = []
			for t in self.__primaryLUT:
				l = [factor * i for i in t]
				l = tuple(l)
				h = [factor * (i + 1) for i in t]
				h = tuple(h)
				lows.append(l)
				his.append(h)

			assert (len(lows) == len(his))
			# Based on the above lows/his, fill in the intermediate voxel indices.
			for l in range(len(lows)):
				low = lows[l]
				hi = his[l]
				for x in range(low[0],hi[0]):
					for y in range(low[1],hi[1]):
						for z in range(low[2],hi[2]):
							completeLUT.append((x,y,z))

		return completeLUT

	# ______________________________________________________
	def __carve(self, ndcube, table):
		"""
		This function takes an input n-dimensional numpy array and a table of
		indices which will be set to 0.
		MUST be called ONLY after generateLUT() returns.
		"""
		# ONlY if this is >= level 1.
		for idx, e in np.ndenumerate(ndcube):
			if self.__iscolor:
				(x, y, z, _) = idx
				if (x,y,z) in table:
					ndcube[x][y][z] = [0, 0, 0]
			else:
				(x, y, z) = idx
				if idx in table:
					ndcube[x][y][z] = 0

		return ndcube

	# ______________________________________________________
	def __menger(self, start, divisions):
		"""
		The primary recursive method to repeatedly call the above methods.
		It takes a starting 3D array as input and the depth to which to traverse.
		Unsurprisingly, it returns a menger cube as an nd numpy array.
		"""

		# Ensure input is a numpy array
		start = np.array(start)

		# Base case
		if divisions == 0:
			return start

		# Subdivide > based on the bigger cube decide zero indices > remove them
		cube = self.__subdiv(start)
		removals = set(self.__generateLUT(cube))
		cube = self.__carve(cube,removals)

		return self.__menger(cube, divisions - 1)

	# ______________________________________________________
	def sliced(self, cube = None, depth = 1, filename = None):
		"""
		This method takes a menger cube and slices off a face along each axis till
		the depth specified.
		"""
		output = dp(cube) if cube else dp(self.__lastRunOutput)

		for i in range(depth):
			for j in range(3): # Use this if you want to mirror all 3 axes
				output = np.delete(output, 0, axis=j)

		if filename:
			vox = vx.from_dense(output)
			vw(filename, vox).write()

		return output

	# ______________________________________________________
	def output(self, divisions, filename = None):
		"""
		Outputs a filename.vox file and returns the n-dimensional voxel array.
		"""

		if not self.__lastRunOutput or self.__lastRunDivisions != divisions:
			ans = self.__menger(self.__base, divisions)
			ans = ans.astype('uint8')

			# TODO : Long-Diagonal slice
			# diag = np.diag_indices(3, ndim=3)
			# ans[diag] = 9

			# Store if needed for re-rerun.
			self.__lastRunDivisions = divisions
			self.__lastRunOutput = dp(ans)

		if filename:
			vox = vx.from_dense(ans)
			vw(filename, vox).write()

		return self.__lastRunOutput

	# ______________________________________________________
	def inverseOutput(self, divisions, filename = None):
		"""
		Returns all voxels that were deleted by calling np.vectorize on
		the array returned by output().
		"""

		if not self.__outputInverse:
			ans = self.__menger(self.__base, divisions)

			# Invert the 1's and 0's.
			# NOTE: Does NOT work with color
			self.__outputInverse = dp(ans)
			inverter = np.vectorize(lambda t: 0 if t == 1 else 1)
			self.__outputInverse = inverter(self.__outputInverse)

		if filename:
			vox = vx.from_dense(self.__outputInverse)
			vw(filename, vox).write()

		return self.__outputInverse
# ______________________________________________________
# ______________________________________________________


if __name__ == "__main__":
	num = 4
	menger = MengerMagica(isColor=True)
	# menger.lut.extend([0,2,6,8,18,20,24,26]) # Menger star
	# [0,1,3,9,2,6,18]) # Menger diagonal fail
	# [0,1,3,9,10,12]
	op = menger.output(num, "menger_pattern.vox")

	"""
	for i in range(pow(3, num)):
		slc = menger.sliced(depth=i, filename="menger_sliced_{}.vox".format(i+1))
	"""

