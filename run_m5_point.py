from discoverlib import geom, graph
import model_m5d as model
import model_utils
import tileloader_sea2 as tileloader

from collections import deque
import numpy
import math
import os
import os.path
from PIL import Image
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

MODEL_BASE = sys.argv[1]
tileloader.tile_dir = sys.argv[2]
tileloader.graph_dir = sys.argv[3]
tileloader.pytiles_path = sys.argv[4]

ROAD_WIDTH = 40
SEGMENT_LENGTH = 20
WINDOW_SIZE = 256
NUM_TRAIN_TILES = 1024
TILE_SIZE = 4096
ANGLE_ONEHOT = 64
PROB_FROM_ROAD = 1.0
NO_DETECT = False
ENABLE_ROTATION = True

tiles = tileloader.Tiles(SEGMENT_LENGTH)
tiles.prepare_training()
num_val = max(5, len(tiles.train_tiles) / 10 + 1)
print 'using {} of {} tiles as validation set'.format(num_val, len(tiles.train_tiles))
val_tiles = tiles.train_tiles[:num_val]
train_tiles = tiles.train_tiles[num_val:]

# initialize model and session
print 'initializing model'
m = model.Model(bn=True, angle_weight=40)
session = tf.Session()
model_path = MODEL_BASE + '/model_latest/model'
best_path = MODEL_BASE + '/model_best/model'
if os.path.isfile(model_path + '.meta'):
	print '... loading existing model'
	m.saver.restore(session, model_path)
else:
	print '... initializing a new model'
	session.run(m.init_op)

if ENABLE_ROTATION:
	FETCH_FACTOR = 2
else:
	FETCH_FACTOR = 1

def get_tile_rect(tile):
	p = geom.Point(tile.x, tile.y)
	return geom.Rectangle(
		p.scale(TILE_SIZE),
		p.add(geom.Point(1, 1)).scale(TILE_SIZE)
	)

tile_edgeprobs = {}
def get_tile_edgeprobs(tile):
	k = '{}_{}_{}'.format(tile.region, tile.x, tile.y)
	if k not in tile_edgeprobs:
		gc = tiles.get_gc(tile.region)
		rect = get_tile_rect(tile).add_tol(-WINDOW_SIZE*FETCH_FACTOR/2)
		edge_ids = []
		edge_lengths = []
		for edge in gc.graph.edges:
			if rect.contains(edge.src.point) and rect.contains(edge.dst.point):
				edge_ids.append(edge.id)
				edge_lengths.append(edge.segment().length())
		edge_lengths = numpy.array(edge_lengths, dtype='float32')
		edge_probs = edge_lengths / edge_lengths.sum()
		tile_edgeprobs[k] = (edge_ids, edge_probs)
	return tile_edgeprobs[k]

def compute_targets(gc, point, edge_pos):
	angle_targets = numpy.zeros((64,), 'float32')

	def best_angle_to_pos(pos):
		angle_points = [model_utils.get_next_point(point, angle_bucket, SEGMENT_LENGTH) for angle_bucket in xrange(64)]
		distances = [angle_point.distance(pos.point()) for angle_point in angle_points]
		point_angle = numpy.argmin(distances) * math.pi * 2 / 64.0 - math.pi
		edge_angle = geom.Point(1, 0).signed_angle(pos.edge.segment().vector())
		avg_vector = geom.vector_from_angle(point_angle, 50).add(geom.vector_from_angle(edge_angle, 50))
		avg_angle = geom.Point(1, 0).signed_angle(avg_vector)
		return int((avg_angle + math.pi) * 64.0 / math.pi / 2)

	def set_angle_bucket_soft(target_bucket):
		for offset in xrange(31):
			clockwise_bucket = (target_bucket + offset) % 64
			counterclockwise_bucket = (target_bucket + 64 - offset) % 64
			for bucket in [clockwise_bucket, counterclockwise_bucket]:
				angle_targets[bucket] = max(angle_targets[bucket], pow(0.75, offset))

	def set_by_positions(positions):
		for pos in positions:
			best_angle_bucket = best_angle_to_pos(pos)
			set_angle_bucket_soft(best_angle_bucket)

	cur_edge = edge_pos.edge
	cur_rs = gc.edge_to_rs[cur_edge.id]

	potential_rs = []
	if cur_rs.edge_distances[cur_edge.id] + edge_pos.distance + SEGMENT_LENGTH < cur_rs.length():
		potential_rs.append(cur_rs)
	else:
		for rs in cur_rs.out_rs(gc.edge_to_rs):
			if rs == cur_rs or rs.is_opposite(cur_rs):
				continue
			potential_rs.append(rs)
	opposite_rs = gc.edge_to_rs[cur_rs.edges[-1].get_opposite_edge().id]
	if cur_rs.edge_distances[cur_edge.id] + edge_pos.distance - SEGMENT_LENGTH > 0:
		potential_rs.append(opposite_rs)
	else:
		for rs in opposite_rs.out_rs(gc.edge_to_rs):
			if rs == opposite_rs or rs.is_opposite(opposite_rs):
				continue
			potential_rs.append(rs)

	expected_positions = []
	for rs in potential_rs:
		pos = rs.closest_pos(point)
		rs_follow_positions = graph.follow_graph(pos, SEGMENT_LENGTH)
		expected_positions.extend(rs_follow_positions)
	set_by_positions(expected_positions)

	return angle_targets

def get_example(traintest='train'):
	while True:
		if traintest == 'train':
			tile = random.choice(train_tiles)
		elif traintest == 'test':
			tile = random.choice(val_tiles)

		edge_ids, edge_probs = get_tile_edgeprobs(tile)
		if len(edge_ids) > 80 or len(edge_ids) > 0:
			break

	# determine rotation factor
	rotation = None
	if ENABLE_ROTATION:
		rotation = random.random() * 2 * math.pi

	rect = get_tile_rect(tile)
	small_rect = rect.add_tol(-WINDOW_SIZE*FETCH_FACTOR/2)

	# get random edge position
	edge_id = numpy.random.choice(edge_ids, p=edge_probs)
	gc = tiles.get_gc(tile.region)
	edge = gc.graph.edges[edge_id]
	distance = random.random() * edge.segment().length()

	# convert to point and add noise
	point = graph.EdgePos(edge, distance).point()
	if random.random() < PROB_FROM_ROAD:
		if random.random() < 0.2:
			noise_amount = 10 * SEGMENT_LENGTH
		else:
			noise_amount = ROAD_WIDTH / 1.5
		noise = geom.Point(random.random() * 2*noise_amount - noise_amount, random.random() * 2*noise_amount - noise_amount)
		point = point.add(noise)
		point = small_rect.clip(point)
	else:
		point = geom.Point(random.randint(0, small_rect.lengths().x - 1), random.randint(0, small_rect.lengths().y - 1))
		point = point.add(small_rect.start)
		point = small_rect.clip(point)

	# match point to edge if possible
	threshold = ROAD_WIDTH
	closest_edge = None
	closest_distance = None
	for edge in gc.edge_index.search(point.bounds().add_tol(threshold)):
		d = edge.segment().distance(point)
		if d < threshold and (closest_edge is None or d < closest_distance):
			closest_edge = edge
			closest_distance = d
	closest_pos = None
	if closest_edge is not None:
		closest_pos = closest_edge.closest_pos(point)

	# generate input
	origin = point.sub(geom.Point(WINDOW_SIZE/2, WINDOW_SIZE/2))
	tile_origin = origin.sub(rect.start)
	fetch_rect = geom.Rectangle(tile_origin, tile_origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE))).add_tol(WINDOW_SIZE*(FETCH_FACTOR-1)/2)
	big_ims = tiles.cache.get_window(tile.region, rect, fetch_rect)
	input = big_ims['input'].astype('float32') / 255.0
	if rotation:
		input = scipy.ndimage.interpolation.rotate(input, rotation * 180 / math.pi, reshape=False, order=0)
		input = input[WINDOW_SIZE/2:3*WINDOW_SIZE/2, WINDOW_SIZE/2:3*WINDOW_SIZE/2, :]

	# compute targets
	if closest_edge is not None:
		angle_targets = compute_targets(gc, point, closest_pos)
		if rotation:
			shift = int(rotation * 32 / math.pi)
			new_targets = numpy.zeros((64,), 'float32')
			for i in xrange(64):
				new_targets[(i + shift) % 64] = angle_targets[i]
			angle_targets = new_targets
	else:
		angle_targets = numpy.zeros((64,), 'float32')

	detect_targets = numpy.zeros((64*FETCH_FACTOR, 64*FETCH_FACTOR, 1), dtype='float32')
	if not NO_DETECT:
		fetch_rect = geom.Rectangle(origin, origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE))).add_tol(WINDOW_SIZE*(FETCH_FACTOR-1)/2)
		for edge in gc.edge_index.search(fetch_rect.add_tol(32)):
			start = edge.src.point.sub(fetch_rect.start).scale(float(64)/WINDOW_SIZE)
			end = edge.dst.point.sub(fetch_rect.start).scale(float(64)/WINDOW_SIZE)
			for p in geom.draw_line(start, end, geom.Point(64*FETCH_FACTOR, 64*FETCH_FACTOR)):
				detect_targets[p.x, p.y, 0] = 1
		if rotation:
			detect_targets = scipy.ndimage.interpolation.rotate(detect_targets, rotation * 180 / math.pi, reshape=False, order=0)
			detect_targets = detect_targets[32:96, 32:96, :]

	info = {
		'region': tile.region,
		'point': point,
		'origin': origin,
		'closest_pos': closest_pos,
		'rotation': rotation,
	}

	return info, input, angle_targets, detect_targets

val_examples = [get_example('test') for _ in xrange(2048)]

def vis_example(example, outputs=None):
	info, input, angle_targets, detect_targets = example
	x = numpy.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype='uint8')
	x[:, :, :] = input * 255
	x[WINDOW_SIZE/2-2:WINDOW_SIZE/2+2, WINDOW_SIZE/2-2:WINDOW_SIZE/2+2, :] = 255

	gc = tiles.get_gc(info['region'])
	rect = geom.Rectangle(info['origin'], info['origin'].add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
	for edge in gc.edge_index.search(rect):
		start = edge.src.point
		end = edge.dst.point
		for p in geom.draw_line(start.sub(info['origin']), end.sub(info['origin']), geom.Point(WINDOW_SIZE, WINDOW_SIZE)):
			x[p.x, p.y, 0:2] = 0
			x[p.x, p.y, 2] = 255

	if info['closest_pos'] is not None:
		p = info['closest_pos'].point().sub(info['origin'])
		x[p.x-2:p.x+2, p.y-2:p.y+2, 0] = 255
		x[p.x-2:p.x+2, p.y-2:p.y+2, 1:3] = 0

	for i in xrange(WINDOW_SIZE):
		for j in xrange(WINDOW_SIZE):
			di = i - WINDOW_SIZE/2
			dj = j - WINDOW_SIZE/2
			d = math.sqrt(di * di + dj * dj)
			a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * 64 / 2 / math.pi)
			if a >= 64:
				a = 63
			elif a < 0:
				a = 0
			elif d > 100 and d <= 120 and angle_targets is not None:
				x[i, j, 0] = angle_targets[a] * 255
				x[i, j, 1] = angle_targets[a] * 255
				x[i, j, 2] = 0
			elif d > 70 and d <= 90 and outputs is not None:
				x[i, j, 0] = outputs[a] * 255
				x[i, j, 1] = outputs[a] * 255
				x[i, j, 2] = 0
	return x

best_loss = None

for epoch in xrange(9999):
	start_time = time.time()
	train_losses = []
	for _ in xrange(1024):
		examples = [get_example('train') for _ in xrange(model.BATCH_SIZE)]
		feed_dict = {
			m.is_training: True,
			m.inputs: [example[1] for example in examples],
			m.angle_targets: [example[2] for example in examples],
			m.detect_targets: [example[3] for example in examples],
			m.learning_rate: 1e-5,
		}
		if ANGLE_ONEHOT:
			feed_dict[m.angle_onehot] = model_utils.get_angle_onehot(ANGLE_ONEHOT)
		_, angle_loss, detect_loss, loss = session.run([m.optimizer, m.angle_loss, m.detect_loss, m.loss], feed_dict=feed_dict)
		train_losses.append((angle_loss, detect_loss, loss))

	train_loss = numpy.mean([l[0] for l in train_losses]), numpy.mean([l[1] for l in train_losses]), numpy.mean([l[2] for l in train_losses])
	train_time = time.time()

	val_losses = []
	for i in xrange(0, len(val_examples), model.BATCH_SIZE):
		examples = val_examples[i:i+model.BATCH_SIZE]
		feed_dict = {
			m.is_training: False,
			m.inputs: [example[1] for example in examples],
			m.angle_targets: [example[2] for example in examples],
			m.detect_targets: [example[3] for example in examples],
		}
		if ANGLE_ONEHOT:
			feed_dict[m.angle_onehot] = model_utils.get_angle_onehot(ANGLE_ONEHOT)
		angle_loss, detect_loss, loss = session.run([m.angle_loss, m.detect_loss, m.loss], feed_dict=feed_dict)
		val_losses.append((angle_loss, detect_loss, loss))

	val_loss = numpy.mean([l[0] for l in val_losses]), numpy.mean([l[1] for l in val_losses]), numpy.mean([l[2] for l in val_losses])
	val_time = time.time()

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss)

	m.saver.save(session, model_path)
	if best_loss is None or val_loss[0] < best_loss:
		best_loss = val_loss[0]
		m.saver.save(session, best_path)
