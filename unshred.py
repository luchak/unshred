#!/usr/bin/env python
"""
Unshred some pictures.

Usage:
  ./unshred.py path/to/imagefile.png

This will produce imagefile-unshredded.jpg in your working directory.

Quick implementation notes:
  - The actual unshredding part of this consists of solving an asymmetric
    traveling salesman's problem, using some sort of column similarity measure
    (negated, since we want to minimize) as edge weights.
  - Shred similarity computed by normalized cross-correlation of edges
    (http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation)
  - Shred width found by autocorrelation on similarities of adjacent image
    columns
  - Shred ordering TSP solved by greedily aggregating most similar shreds
    without forming cycles

Things that that did not work as well:
  - solving the TSP exactly using dynamic programming (very slow)
  - solving the TSP using NN instead of greedily (worked worse with buggy
    similarities)
  - using normalized column differences instead of correlation (correlation is
    more robust)
  - computing shred width by computing average correlation for interval size
    (filtered autocorrelation was more robust by far when I had buggy
    correlation calculation; this may no longer be true for fixed correlation
    calculation)
  - 3-opt refinement for TSP solutions. If your TSP heuristic is returning bad
    results, you're better off fixing it by finding better edge similarity
    measures than by tweaking the TSP.

Known issues:
  - images with solid-colored borders tend to get these (low-cost) borders
    relocated to the center of the frame
  - non-photographic images have all kinds of assorted problems

Possible improvements:
  - mutual information for column similarity
  - reward low-frequency regions less
  - special case solid border detection

contact: matt@mattstanton.com
"""

import os
import sys

import Image
import numpy

def Autocorrelate(x):
  """Returns the autocorrelation of x.
  
  See http://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation . 
  """
  frf = numpy.fft.rfft(x - x.mean(), 2*len(x))
  ac = numpy.fft.irfft(frf*frf.conjugate())[:len(x)]
  return ac / ac[0]

class UnionFind(object):
  """Keeps track of which objects are in disjoint sets.

  Clients should use self.union_index directly for queries.

  Note that we're not doing the clever tree-based thing here, because we don't need to.
  """
  def __init__(self, num_members):
    self.unions = {i: set([i]) for i in range(num_members)}
    self.union_index = range(num_members)

  def AddEdge(self, a, b):
    """Join the sets containing a and b."""
    a_union = self.union_index[a]
    b_union = self.union_index[b]
    if a_union == b_union:
      return
    self.unions[a_union] = self.unions[a_union].union(self.unions[b_union])
    for i in self.unions[b_union]:
      self.union_index[i] = a_union
      assert i in self.unions[a_union]

    del self.unions[b_union]

def AdjacencyToOrder(right_adj):
  """Transform an index of right-hand adjacencies into an ordering.

  For example,  [3, None, 0, 1] becomes [2, 0, 3, 1].
  """

  order = []
  num_items = len(right_adj)
  # The leftmost shred has the unique distinction of being nobody's right-hand
  # neighbor. We first search for the leftmost shred, then trace the rest of
  # the order through right_adj.
  leftmost = set(range(num_items))
  for i in right_adj:
    if i is not None:
      leftmost.remove(i)
  leftmost = list(leftmost)[0]
  order = [leftmost]
  for i in range(num_items - 1):
    next_item = right_adj[order[-1]]
    order.append(next_item)
  return order

def Unshred(image):
  """Reassembles an image in which regularly-sized columnar blocks have been
  scrambled.
  
  Assumes that the blocks are of regular size and that the image is
  qualitatively similar to a photograph.

  We reassemble the image using a heuristic approach, repeatedly joining the
  two most-similar blocks that can be sensibly joined until the whole image has
  been reassembled (this is just the greedy heuristic for the TSP). Column edge
  similarity is measured by normalized cross-correlations.

  Shred size detection is computed by examining the highpass-filtered
  autocorrelation of similarities of adjacent columns, and taking the first big
  peak (corresponding to the smallest shreds). Taking a larger shred size
  exposes us to potentially picking up higher harmonics.
  """

  SHRED_DETECT_THRESHOLD = 0.6
  width, height = image.size

  cols = numpy.array(image, numpy.float64)
  # Perceptual scaling for image color components.
  cols = cols[:,:,:3] * numpy.array((0.3, 0.59, 0.11))
  cols = cols.transpose((1, 0, 2)).reshape((width, height*3))
  # Center columns and normalize variances
  cols -= numpy.mean(cols, axis=1)[:,numpy.newaxis]
  col_norms = numpy.sqrt(numpy.sum(cols*cols, axis=1)) + 1e-10
  cols /= col_norms[:,numpy.newaxis]

  # Compute similarities of the edges of adjacent columns.
  adjacent_col_correlations = numpy.array(
      [numpy.dot(cols[i], cols[i+1]) for i in range(len(cols) - 1)])
  # This line is (I believe after a lot of testing) correct, and it's
  # definitely faster than the line above, but it triggers a weird numpy bug on
  # my machine for larger images:
  # adjacent_col_correlations = numpy.sum(cols[:-1,:] * cols[1:,:], axis=1)

  # Detect shred size. We produce an autocorrelation for the energies of
  # adjacent columns, highpass filter it, and use the shred width corresponding
  # to the first high peak (where "high" is defined by SHRED_DETECT_THRESHOLD).
  autocorrelation = numpy.convolve(Autocorrelate(adjacent_col_correlations),
                                   numpy.array((1.0, -1.0)))
  # No abs() in normalization because we don't care about negative values and
  # don't want them to potentially throw us off.
  autocorrelation /= numpy.max(autocorrelation[2:])
  candidate_shred_widths = range(2, width/2 + 1)
  threshold = min(SHRED_DETECT_THRESHOLD, max(autocorrelation))
  correlation_above_threshold = [w for w in candidate_shred_widths
                                 if autocorrelation[w] >= threshold]
  shred_width = correlation_above_threshold[0]
  num_shreds = width / shred_width
  print >>sys.stderr, 'Using %d shreds of width %d' % (num_shreds, shred_width)


  # Generate the cost matrix. cost[i, j] is the cost of placing column i to the
  # left of column j.
  left_boundaries = numpy.array([shred_width*i for i in range(num_shreds)])
  left_boundary_features = numpy.array(cols[left_boundaries])
  right_boundaries = numpy.array([shred_width*(i+1)-1 for i in range(num_shreds)])
  right_boundary_features = numpy.array(cols[right_boundaries])
  cost_matrix = -numpy.dot(right_boundary_features, left_boundary_features.T)

  # Generate, for each shred i, a list of all other shreds j in order of how
  # much i wants to have j as its right-hand neighbor (where "wants" =
  # minimizes cost). Most-preferred shreds are at the tail of the list and
  # can be removed with pop().
  right_neighbor_prefs = [list(reversed(cost_matrix[i].argsort())) for i in range(num_shreds)]

  # Build a list of right-hand shred adjacencies by repeatedly attaching the
  # lowest-cost two shreds to each other while ensuring that we ultimately end
  # up with one big cycle.
  remaining_left = set(range(num_shreds))
  remaining_right = set(range(num_shreds))
  right_adj = [None] * num_shreds
  connected_components = UnionFind(num_shreds)
  while len(remaining_left) > 1:
    min_cost = 1e100
    best_l = None
    # Examine all shreds that are not yet someone's left-hand neighbor to see
    # which has the lowest-cost untaken right-hand neighbor.
    for l in remaining_left:
      r = right_neighbor_prefs[l][-1]
      # Remove candidates that are already someone's right neighbor or that
      # would create a cycle
      while ((r not in remaining_right) or
             (connected_components.union_index[r] == connected_components.union_index[l])):
        right_neighbor_prefs[l].pop()
        r = right_neighbor_prefs[l][-1]
      if cost_matrix[l, r] < min_cost:
        min_cost = cost_matrix[l, r]
        best_l = l
    best_r = right_neighbor_prefs[best_l].pop()
    remaining_left.remove(best_l)
    remaining_right.remove(best_r)
    right_adj[best_l] = best_r
    connected_components.AddEdge(best_l, best_r)

  order = AdjacencyToOrder(right_adj)

  # Assemble a new image using the computed order.
  result = Image.new('RGBA', image.size)
  for dst, src in enumerate(order):
    source_region = image.crop((shred_width*src, 0, shred_width*(src + 1), height))
    destination_point = (shred_width*dst, 0)
    result.paste(source_region, destination_point)

  return result


if __name__ == '__main__':
  usage = 'Usage: %s imagefile.png' % sys.argv[0]
  try:
    input_filename = sys.argv[1]
  except IndexError:
    print >>sys.stderr, usage
    sys.exit(1)
  try:
    image = Image.open(input_filename)
  except IOError:
    print >>sys.stderr, usage
    print >>sys.stderr, 'Could not open file "%s"' % input_filename
    sys.exit(1)

  output_filename = '%s-unshredded.jpg' % os.path.splitext(os.path.basename(input_filename))[0]
  Unshred(image).save(output_filename, 'JPEG')
  print >>sys.stderr, 'Output written to %s' % output_filename

