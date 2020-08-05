"""
Author:  Arne Tobias Elve
What:    Internal model representation
Started: 2019-05-21
Reason:  To handle dynamic changes to the model
Status:  Production
Contact: arne.t.elve(at)ntnu.no
"""

from collections import OrderedDict


class Model(object):
  """
  Internal model representation handling changes to the model based on networks
  Inputs:
    - model_dict
    - typed_token_dict
  Returns:
    - model object handlig structure
  """

  def __init__(self, model_dict, typed_token_dict, ontology):
    self.model_dict = model_dict
    self.typed_token_dict = typed_token_dict
    self.ontology = ontology

    self.nodes = OrderedDict()
    self.arcs = OrderedDict()
    self.typed_tokens = OrderedDict()
    self.networks = OrderedDict()
    self.indices = OrderedDict()

    self.index_dict = self.ontology.readIndices()

    # print(self.index_dict)

    self.loadModel()
    self.generate_networks()
    self.populate_index_sets()
    # print('HELLO FROM THE OTHER OTHER SIDE')

  def loadModel(self):
    """
    Collect and initialize the internal model reprentation
    """
    for label, node in self.model_dict["nodes"].items():
      # print(node)
      if node["type"] == 'node_composite':
        continue
      self.nodes[label] = Node(label, node)

    for label, arc in self.model_dict["arcs"].items():
      self.arcs[label] = Arc(label, arc)

    for label, tt_dict in self.typed_token_dict.items():
      self.typed_tokens[label] = TypedToken(label, tt_dict)

  def generate_networks(self):
    for network in self.ontology.networks:
      nodelist = []
      arclist = []
      union_subset = self.ontology.ontology_in_hiearchy[network].split('_')

      for label, node in self.nodes.items():
        if node.network in union_subset:
          nodelist.append(label)

      for label, arc in self.arcs.items():
        if arc.network in union_subset:
          arclist.append(label)

      self.networks[network] = Network(nodelist, arclist)

  def populate_index_sets(self):
    for label, dict in self.index_dict.items():
      self.indices[label] = IndexSet(label, dict, self)


class Node(object):
  """
  Internal node representation
  """
  ___refs___ = []

  def __init__(self, label, node_dict):
    self.___refs___.append(self)                            # Internal labeling

    # Explicit labeling of attributes used
    self.label = label
    self.network = node_dict["network"]
    self.named_network = node_dict["named_network"]
    self.tokens = node_dict["tokens"]
    self.type = node_dict["type"]
    if self.type == 'dynamic':
      self.injected_conversions = node_dict["injected_conversions"]
    else:
      self.injected_conversions = {}

class Arc(object):
  """
  Internal arc representation
  """
  ___refs___ = []

  def __init__(self, label, arc_dict):
    self.___refs___.append(self)                            # Internal labeling
    self.label = label

    self.network = arc_dict["network"]
    self.token = arc_dict["token"]
    self.mechanism = arc_dict["mechanism"]
    self.typed_tokens = arc_dict["typed_tokens"]
    self.source = arc_dict["source"]
    self.sink = arc_dict["sink"]


class TypedToken(object):
  """
  Internal representation of typed token
  """

  def __init__(self, label, tt_dict):
    self.label = label
    self.instances = tt_dict['instances']
    self.conversions = tt_dict['conversions']
    # print(tt_dict)


class Network(object):
  """
  Internal network represenation
  Inputs:
    nodes: list of nodes in network
    arcs: list of arcs in network
  """

  def __init__(self, nodes, arcs):
    self.nodes = nodes
    self.arcs = arcs


class IndexSet(object):
  """
  Internal index set representation

  Added functionality for labeling and representation
  """

  def __init__(self, label, index_dict, model):
    self.label = label
    self.ind_dict = index_dict
    self.type = index_dict["type"]
    self.model = model
    self.networks = index_dict['network']
    self.mapping, self.blocking = self.make_mapping()
    # self.

  def make_mapping(self):
    # mapping = []
    # blocking = []

    if self.type == "index":
      mapping, blocking = self.make_mapping_index()
    elif self.type == "block_index":
      mapping, blocking = self.make_mapping_block()
    return mapping, blocking

  def make_mapping_index(self):
    mapping = []
    blocking = []

    if "_" in self.label:
      ttok, conv = self.label.split("_")
    # if ttok in self.model.typed_tokens.keys() and conv == "conversion":
    if self.label == "node":
      # keys =
      for label in sorted(self.model.nodes.keys(), key = float):
        node = self.model.nodes[label]
        # for label, node in self.model.nodes.items():
        mapping.append(int(label))
        blocking.append(1)
    elif self.label == "arc":
      # for label, arc in self.model.arcs.items():
      for label in sorted(self.model.arcs.keys(), key = float):

        mapping.append(int(label))
        blocking.append(1)
    elif self.label in self.model.typed_tokens.keys():         # typed tokens
      for i, label in enumerate(self.model.typed_tokens[self.label].instances):
        mapping.append(i)
        blocking.append(1)
    elif conv == 'conversion' and ttok in self.model.typed_tokens.keys():
      for i, label in enumerate(self.model.typed_tokens[ttok].conversions):
        mapping.append(i)
        blocking.append(1)
    else:
      pass
    return mapping, blocking

  def make_mapping_block(self):
    mapping = []
    blocking = []

    outer = self.ind_dict['outer']                         # Always node or arc
    inner = self.ind_dict['inner']

    self.outer = outer

    nw_tk_ttk = self.model.ontology.token_typedtoken_on_networks
    if outer == 'node':                                   # Node based indexing
      if "_" in inner:
        ttok, conv = inner.split("_")
      else:
        ttok, conv = '', ''
      for label, node in self.model.nodes.items():
        mapping.append(int(label))
        blocksize = 0
        if node.network:
          td = nw_tk_ttk[node.network].items()

          for token, typed_token in [(t, tt) for t, ts in td for tt in ts]:
            if inner == typed_token and token in node.tokens.keys():
              blocksize += len(node.tokens[token])
            elif ttok == typed_token and conv == 'conversion':
              if token in node.injected_conversions.keys():
                blocksize += len(node.injected_conversions[token])
        blocking.append(blocksize)

    elif outer == 'arc':
      for label, arc in self.model.arcs.items():
        mapping.append(int(label))
        td = nw_tk_ttk[arc.network].items()
        blocksize = 0
        for token, typed_token in [(t, tt) for t, ts in td for tt in ts]:
          if inner == typed_token and token == arc.token:
            blocksize += len(arc.typed_tokens)
        blocking.append(blocksize)

    else:
      raise RuntimeError('Neither arc or node set!')

    return mapping, blocking
