from discoverlib import geom, graph
import fiona
from shapely import geometry
import sys

fname = sys.argv[1]
out_name = sys.argv[2]
g = graph.read_graph(fname, merge_duplicates=True, fpoint=True)
road_segments, _ = graph.get_graph_road_segments(g)

lines = []
seen_node_pairs = set()
for rs in road_segments:
	if (rs.src().id, rs.dst().id) in seen_node_pairs or (rs.dst().id, rs.src().id) in seen_node_pairs:
		continue
	points = []
	for edge in rs.edges:
		if len(points) > 0 and edge.src.point == points[-1]:
			print 'adding edge {} -> {}; points: {}; edges: {}'.format(edge.src.point, edge.dst.point, points, [(e.src.point, e.dst.point) for e in rs.edges])
			print 'skip'
			continue
		points.append(edge.src.point)
	points.append(rs.dst().point)
	points = [(p.x, p.y) for p in points]
	lines.append(geometry.LineString(points))
	seen_node_pairs.add((rs.src().id, rs.dst().id))

schema = {'geometry': 'LineString','properties': {}}
with fiona.open(out_name, 'w', 'ESRI Shapefile', schema) as layer:
	for line in lines:
		layer.write({
			'geometry': geometry.mapping(line),
			'properties': {},
		})
