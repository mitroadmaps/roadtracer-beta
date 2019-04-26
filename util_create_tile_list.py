from discoverlib import graph
import json
import math
import sys

regions = sys.argv[1].split(',')
fnames = sys.argv[2].split(',')
out_fname = sys.argv[3]
tile_size = 4096.0

json_tiles = []

for i in xrange(len(regions)):
	region = regions[i]
	fname = fnames[i]

	g = graph.read_graph(fname)
	tiles = set()
	for vertex in g.vertices:
		x = int(math.floor(vertex.point.x / tile_size))
		y = int(math.floor(vertex.point.y / tile_size))
		tiles.add((x, y))

	for x, y in tiles:
		json_tiles.append({
			'x': x,
			'y': y,
			'region': region,
		})

with open(out_fname, 'w') as f:
	json.dump(json_tiles, f)
