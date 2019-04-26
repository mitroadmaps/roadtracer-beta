from discoverlib import geom
from discoverlib import graph
import model_utils

import json
import numpy
import os
import random
import rtree
import scipy.ndimage
import time

tile_dir = None
graph_dir = None
pytiles_path = None

angles_dir = None
tile_size = 4096
window_size = 256

def get_tile_dirs():
	if isinstance(tile_dir, str):
		return [tile_dir]
	else:
		return tile_dir

def get_tile_keys():
	keys = []
	for i in xrange(len(get_tile_dirs())):
		if i == 0:
			keys.append('input')
		else:
			keys.append('input{}'.format(i))
	return keys

def load_tile(region, i, j):
	d = {}
	for pathIdx, path in enumerate(get_tile_dirs()):
		prefix = '{}/{}_{}_{}_'.format(path, region, i, j)
		sat_im = scipy.ndimage.imread(prefix + 'sat.png')
		if sat_im.shape == (tile_size, tile_size, 4):
			sat_im = sat_im[:, :, 0:3]
		sat_im = sat_im.swapaxes(0, 1)
		if pathIdx == 0:
			d['input'] = sat_im
		else:
			d['input{}'.format(pathIdx)] = sat_im
	if angles_dir:
		angle_im = numpy.fromfile('{}/{}_{}_{}.bin'.format(angles_dir, region, i, j), dtype='uint8')
		angle_im = angle_im.reshape(tile_size/4, tile_size/4, 64)
		d['angles'] = angle_im

	return d

def load_rect(region, rect, load_func=load_tile):
	# special case for fast load: rect is single tile
	if rect.start.x % tile_size == 0 and rect.start.y % tile_size == 0 and rect.end.x % tile_size == 0 and rect.end.y % tile_size == 0 and rect.end.x - rect.start.x == tile_size and rect.end.y - rect.start.y == tile_size:
		return load_func(region, rect.start.x / tile_size, rect.start.y / tile_size)

	tile_rect = geom.Rectangle(
		geom.Point(rect.start.x / tile_size, rect.start.y / tile_size),
		geom.Point((rect.end.x - 1) / tile_size + 1, (rect.end.y - 1) / tile_size + 1)
	)
	full_rect = geom.Rectangle(
		tile_rect.start.scale(tile_size),
		tile_rect.end.scale(tile_size)
	)
	full_ims = {}

	for i in xrange(tile_rect.start.x, tile_rect.end.x):
		for j in xrange(tile_rect.start.y, tile_rect.end.y):
			p = geom.Point(i - tile_rect.start.x, j - tile_rect.start.y).scale(tile_size)
			tile_ims = load_func(region, i, j)
			for k, im in tile_ims.iteritems():
				scale = tile_size / im.shape[0]
				if k not in full_ims:
					full_ims[k] = numpy.zeros((full_rect.lengths().x / scale, full_rect.lengths().y / scale, im.shape[2]), dtype='uint8')
				full_ims[k][p.x/scale:(p.x+tile_size)/scale, p.y/scale:(p.y+tile_size)/scale, :] = im

	crop_rect = geom.Rectangle(
		rect.start.sub(full_rect.start),
		rect.end.sub(full_rect.start)
	)
	for k in full_ims:
		scale = (full_rect.end.x - full_rect.start.x) / full_ims[k].shape[0]
		full_ims[k] = full_ims[k][crop_rect.start.x/scale:crop_rect.end.x/scale, crop_rect.start.y/scale:crop_rect.end.y/scale, :]
	return full_ims

class TileCache(object):
	def __init__(self):
		self.cache = {}
		self.last_used = {}

	def get(self, region, rect):
		k = '{}.{}.{}.{}.{}'.format(region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
		if k not in self.cache:
			self.cache[k] = load_rect(region, rect)
		self.last_used[k] = time.time()
		return self.cache[k]

	def get_window(self, region, big_rect, small_rect):
		big_dict = self.get(region, big_rect)
		small_dict = {}
		for k, v in big_dict.items():
			scale = tile_size / v.shape[0]
			small_dict[k] = v[small_rect.start.x/scale:small_rect.end.x/scale, small_rect.start.y/scale:small_rect.end.y/scale, :]
		return small_dict

def get_tile_list():
	tiles = []
	with open(pytiles_path, 'r') as f:
		for json_tile in json.load(f):
			tile = geom.Point(int(json_tile['x']), int(json_tile['y']))
			tile.region = json_tile['region']
			tiles.append(tile)
	downloaded = set([fname.split('_sat.png')[0] for fname in os.listdir(get_tile_dirs()[0]) if '_sat.png' in fname])
	dl_tiles = [tile for tile in tiles if '{}_{}_{}'.format(tile.region, tile.x, tile.y) in downloaded]
	print 'found {} total tiles, using {} downloaded tiles'.format(len(tiles), len(dl_tiles))
	return dl_tiles

class Tiles(object):
	def __init__(self, segment_length):
		self.segment_length = segment_length

		# load tile list
		# this is a list of point dicts (a point dict has keys 'x', 'y')
		# don't include test tiles
		print 'reading tiles'
		self.all_tiles = get_tile_list()
		self.regions = set([tile.region for tile in self.all_tiles])
		self.cache = TileCache()

		self.gcs = {}

	def get_gc(self, region):
		if region in self.gcs:
			return self.gcs[region]
		print 'loading gc for {}'.format(region)
		fname = os.path.join(graph_dir, region + '.graph')
		g = graph.read_graph(fname)
		gc = graph.GraphContainer(g)
		self.gcs[region] = gc
		return gc

	def cache_gcs(self, regions):
		for region in regions:
			self.get_gc(region)

	def prepare_training(self):
		self.cache_gcs(self.regions)
		self.train_tiles = list(self.all_tiles)
		random.shuffle(self.train_tiles)

	def num_tiles(self):
		return len(self.train_tiles)
