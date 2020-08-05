"""
ONLY EDIT IN MODELFACTORY
What:    Simulation module in python
Author:  Arne Tobias Elve
Contact: arne.t.elve(at)ntnu.no
Date:    2018-09-24
Update:  2018-10-22
"""

import numpy as np


class IndexSet(object):
  """
  Index sets class
  Used as a placeholder for now.
  """

  def __init__(self, name, mapping, blocking, super = None):
    self.blocking = blocking
    self.mapping = mapping
    self.super = super
    self.name = name

###############################################################################
#   KHATRI RAO PRODUCT RELATED FUNCTIONS                                      #
###############################################################################


def __mkblocks(mat, size):
  """
  Inputs:
    mat: Matrix
    blkRows: Blocks in rows
    blkCols: Blocks in columns

  Returns:
    matrices: List of blocks  in matrix,  sequence  based  the  representation:
              (block-row-number, block-column-number).
  """
  matrices = []
  currow = 0
  mat = np.array(mat)
  if len(size) < 2:             # HACK: Adding more dimension to column vectors
    size.append([1])

  for nspecies in size[0]:
    if nspecies == 0:                                        # HACK: Superhack!
      nspecies = 1
    curcol = 0
    for aspecies in size[1]:
      if aspecies == 0:                                      # HACK: Superhack!
        aspecies = 1
      matrices.append(mat[currow:(currow+nspecies), curcol:(curcol+aspecies)])
      curcol += aspecies
    currow += nspecies
  return matrices

###############################################################################


def khatriRao(mata, inda, matb, indb):
  """
  mata and matb are the matrices that are multiplied.
  Inda and indb are the respective index sets
  """
  # Always doing mapping to the super set, which means  that blocking cannot be
  # zero. It need to be represented due to the smaller index sets.
  sizea = [ind.blocking for ind in inda]
  sizeb = [ind.blocking for ind in indb]

  matsA, matsB = [__mkblocks(mat, size) for (mat, size)
                  in zip([mata, matb], [sizea, sizeb])]
  matrices = np.array([np.kron(A, B) for (A, B) in zip(matsA, matsB)])
  size = (len(sizea[0]), len(sizea[1]))
  mat = np.concatenate(
        [np.concatenate(matrices[node*size[1]:node*size[1]+size[1]], axis = 1)
            for node in range(size[0])], axis = 0)
  return np.array(mat)
###############################################################################


def blockReduce(var1, reduction_set, block_set, var2):
  """
  Inputs:
    var1 and var2: variables
    reduction_set: reduction set
    block_set: set that contains the blocking
  Returns:
    Reduced matrix
  """
  # TODO: Reduction set not needed now. Remove or needed later?
  counter = 0
  value = []
  for l in block_set.blocking:
    valll1 = var1[counter:counter+l]
    valll2 = var2[counter:counter+l]
    value.append(np.einsum('ij,ij -> j', valll1, valll2))
    counter += l
  return np.array(value)


def blockProduct(var, index, reuction, block_set):
  if len(index) > 1:                                                   # Matrix
    size = [ind.blocking for ind in index]
    mats = __mkblocks(var, size)
    matrices = []
    for mat in mats:
      if block_set == index[0]:
        matrices.append(np.prod(mat, axis = 0))
      elif block_set == index[1]:
        matrices.append(np.transpose([np.prod(mat, axis = 1)]))

    if block_set == index[1]:
      matrices = np.array(matrices)
      sizes = (len(size[0]), len(size[1]))
      tmpMats = []
      for node in range(sizes[0]):
        tmpMats.append(np.concatenate(
                       matrices[node*sizes[1]:node*sizes[1]+sizes[1]],
                       axis = 1))
      mat = np.concatenate(tmpMats, axis = 0)
      return np.array(mat)
    else:
      matrices = np.array(matrices)
      sizes = (len(size[0]), len(size[1]))
      tmpMats = []
      for node in range(sizes[0]):
        tmpMats.append([np.concatenate(matrices[node*sizes[1]:node*sizes[1]+sizes[1]])])

      mat = np.concatenate(tmpMats, axis = 0)
      return np.array(mat)
  else:
    counter = 0
    value = []
    for l in block_set.blocking:
      valll1 = var[counter:counter+l]
      value.append([np.prod(valll1)])
      counter += l
    return np.array(value)

###############################################################################


if __name__ == '__main__':
  # Index sets:
  A = IndexSet('A', [0, 1], [1, 1])
  AS = IndexSet('AS', [0, 1], [2, 2], A)
  N = IndexSet('N', [0, 1, 2, 3], [1, 1, 1, 1])
  NS = IndexSet('NS', [0, 1, 2], [2, 2, 2], N)
  S = IndexSet('S', [0, 1], [1, 1])

  # Matrices
  Fm = np.array([[-1.0, 0.0],
                 [1.0, -1.0],
                 [0.0,  1.0],
                 [0.0,  0.0]])
  Pn = np.array([[ 1, 1, 1, 1],
                 [10, 1, 1, 1],
                 [ 1, 1, 1, 1],
                 [ 2, 1, 1, 1],
                 [ 1, 1, 1, 1],
                 [ 3, 1, 1, 1]])

  T = np.array([[1.0], [2.0], [3.0]])
  ps = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
  # print(khatriRao( Pn, [NS, AS], Fm, [N, A]))
  # print(khatriRao(ps, [NS], T, [N]))
  print(blockProduct(Pn, [NS, AS], S, AS))
