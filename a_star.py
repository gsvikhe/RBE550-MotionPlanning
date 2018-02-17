import sys
from PIL import Image
import copy
from Queue import PriorityQueue
import time
import numpy
import matplotlib.pyplot as plt 
'''
These variables are determined at runtime and should not be changed or mutated by you
'''
start = (0, 0)  # a single (x,y) tuple, representing the start position of the search algorithm
end = (0, 0)    # a single (x,y) tuple, representing the end position of the search algorithm
difficulty = "" # a string reference to the original import file

'''
These variables determine display color, and can be changed by you, I guess
'''

PURPLE = (85, 26, 139)
LIGHT_GRAY = (50, 50, 50)
DARK_GRAY = (100, 100, 100)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

'''
These variables are determined and filled algorithmically, and are expected (and required) be mutated by you
'''
path = []       # an ordered list of (x,y) tuples, representing the path to traverse from start-->goal
expanded = {}   # a dictionary of (x,y) tuples, representing nodes that have been expanded
frontier = {}   # a dictionary of (x,y) tuples, representing nodes to expand to in the future


def heuristic(a, b):
	(x1, y1) = a
	(x2, y2) = b
	#a=numpy.asarray(a)
	#b=numpy.asarray(b)
	h = abs(x2 - x1) + abs(y2 - y1) # Manhattan Distance
	#h=numpy.linalg.norm(a-b)       # Euclidean Distance
	return h

def expand_node(map, size, node):
	x = node[0]
	y = node[1]
	neighbors = []

	x_bound = size[0]
	y_bound = size[1]
	
        #This is the 4-distance
	if (x+1 < x_bound) and (map[x+1,y] != 0): #1
		neighbors.append((x+1,y))

	if (y+1 < y_bound) and (map[x,y+1] != 0): #1
		neighbors.append((x,y+1))

	if (y>=1) and map[x,y-1] != 0: #1
		neighbors.append((x,y-1))

	if (x>=1) and map[x-1,y] != 0: #1
		neighbors.append((x-1,y))
	#(x - 1, y-1),(x+1, y + 1),(x + 1, y-1),(x-1, y + 1)] these would be the remaining neighbours for the 8-distance

	return neighbors
	
def a_star_search(map, size, start, goal):
	front = PriorityQueue(0)
	front.put((0, start))
	came_from = {}
	explored = {}

	came_from[start] = None
	explored[start] = 0
	G = 0
	while not front.empty():
		current_node = front.get()[1]

		if current_node == goal:
			G = front.get()[0]
			break

		for next in expand_node(map, size, current_node):
			new_cost = explored[current_node] + 1
			if next not in explored or new_cost < explored[next] :
				explored[next] = new_cost
				total_cost = new_cost + heuristic(goal, next)
				front.put((total_cost, next))
				came_from[next] = current_node

	path = []
	while current_node != start:
		path.append(current_node)
		current_node = came_from[current_node]
	path.append(start)
	path.reverse()
   
	frontier = {}
	for f in front.queue:
		frontier[f[1]] = f[0]

	return path, explored, frontier, G

def search(map, size):

	global path, start, end, path, expanded, frontier
	
	"""
	This function is meant to use the global variables [start, end, path, expanded, frontier] to search through the
	provided map.
	:param map: A '1-concept' PIL PixelAccess object to be searched. (basically a 2d boolean array)
	"""
	print ("")
	print "Start Point: " + str(start)
	print "End   Point: " + str(end)

	# O is unoccupied (white); 1 is occupied (black)
	print ("")
	print "pixel value at start point ", map[start[0], start[1]]
	print "pixel value at end point ", map[end[0], end[1]]

	start_time = time.time()
	path, expanded, frontier, G = a_star_search(map, size, start, end)
	end_time = time.time()
	time_diff = end_time - start_time

	print("")
	print("Cost G:     ") + str(G) + ' units'
	print("Time Taken: ") + str(time_diff) + ' sec'

	visualize_search("out.png") # see what your search has wrought (and maybe save your results)

def visualize_search(save_file="lol.png"):
	"""
	:param save_file: (optional) filename to save image to (no filename given means no save file)
	"""
	im = Image.open(difficulty).convert("RGB")
	pixel_access = im.load()

	# draw start and end pixels
	pixel_access[start[0], start[1]] = GREEN
	pixel_access[end[0], end[1]] = GREEN

	# draw expanded pixels
	for pixel in expanded.keys():
		pixel_access[pixel[0], pixel[1]] = LIGHT_GRAY

	# draw path pixels
	for pixel in path:
		pixel_access[pixel[0], pixel[1]] = GREEN

	 # draw frontier pixels
	for pixel in frontier.keys():
		pixel_access[pixel[0], pixel[1]] = RED
	
	# display and (maybe) save results
	im.show()
	if(save_file != "do_not_save.png"):
		im.save(save_file)
	im.close()

if __name__ == "__main__":
	# Throw Errors && Such
	assert sys.version_info[0] == 2                                 # require python 2 (instead of python 3)
	assert len(sys.argv) == 2, "Incorrect Number of arguments"      # require difficulty input

	# Parse input arguments
	function_name = str(sys.argv[0])
	difficulty = str(sys.argv[1])
	print ("")
	print ("A* Implementation")
	print "running " + function_name + " with " + difficulty + " difficulty."

	# Hard code start and end positions of search for each difficulty level
	if difficulty == "crazy.jpg":
		start = (80, 273)
		end = (1086, 356)
	elif difficulty == "ultra.jpg":
		start = (38, 22)
		end = (274, 274)
        elif difficulty == "weird.png":
		start = (169, 33)
		end = (382, 626)
	elif difficulty == "medium.gif":
		start = (8, 201)
		end = (110, 1)
	elif difficulty == "hard.gif":
		start = (10, 1)
		end = (401, 220)
	elif difficulty == "very_hard.gif":
		start = (1, 324)
		end = (580, 1)
	elif difficulty == "nishant_item.png":
		start = (112, 476)
		end = (989, 114)
	else:
		assert False, "Incorrect difficulty level provided"

	# Perform search on given image
	im = Image.open(difficulty)
	im=im.convert('1')
	search(im.load(), im.size)
