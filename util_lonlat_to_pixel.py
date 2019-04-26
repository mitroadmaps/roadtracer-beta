from discoverlib import coords, graph
import sys

zoom = 18
origin_tile = [int(sys.argv[1]), int(sys.argv[2])]
in_fname = sys.argv[3]
out_fname = sys.argv[4]

g = graph.read_graph(in_fname, fpoint=True)
for vertex in g.vertices:
	vertex.point = coords.lonLatToMapbox(vertex.point, zoom, origin_tile)
g.save(out_fname)
