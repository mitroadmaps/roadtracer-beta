from discoverlib import geom, graph
import fiona
import sys

fname = sys.argv[1]
out_name = sys.argv[2]
f = fiona.open(fname)
g = graph.Graph()
vertex_map = {}

def add(coords):
	points = [geom.FPoint(c[0], c[1]) for c in coords]
	vertices = []
	for point in points:
		if point not in vertex_map:
			vertex_map[point] = g.add_vertex(point)
		vertices.append(vertex_map[point])
	for i in xrange(len(points) - 1):
		src = vertices[i]
		dst = vertices[i + 1]
		if src == dst:
			continue
		g.add_bidirectional_edge(src, dst)

for shape in f:
	t = shape['geometry']['type']
	if t == 'MultiLineString':
		for coords in shape['geometry']['coordinates']:
			add(coords)
	else:
		add(shape['geometry']['coordinates'])
g.save(out_name)
