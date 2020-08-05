#!/usr/bin/python3

"""
Module for model representation

@summary:      Model framework module
@contact:      arne.t.elve(at)ntnu.no
@requires:     Python 3 or higher
@since:        2018-10-01
@version:      0.1
@change:       2019-04-25
@author:       Arne Tobias Elve
@copyright:    2018 Elve, AT All rights reserved.

"""

import json
from collections import OrderedDict
import numpy as np
from Common.common_resources import getData
# import Common.configuration_file
import os
from time import strftime                                    # Library for time


class Model(object):
  """
  Model class
  """

  def __init__(self, location, model, ontology):
    self.location = location
    self.model = model
    self.ontology = ontology
    self.model_file = '{}/models/{}'.format(location, model)
    # self.typed_token_file = '{}/{}'.format(location, 'typed_tokens.json')
    self.dotfile = '{}/DOT/{}'.format(location, 'model.dot')
    self.pdffile = '{}/DOT/{}'.format(location, 'model.pdf')
    self.nodes = OrderedDict()
    self.arcs = OrderedDict()
    self.typed_tokens = OrderedDict()
    self.groups = OrderedDict()
    self.networks_w_namednw_dict = OrderedDict()
    # self.conversions = OrderedDict()
    self.indices = OrderedDict()
    self.loadTypedTokens()
    self.loadModel()
    self.makeIndex()
    # self.makeDot()
    # self.produceDot()

  def loadTypedTokens(self):
    for token, typed_tokens in self.ontology.token_typedtoken_on_networks.items():
      for typed_token in typed_tokens:
        if typed_token:
          self.typed_tokens[typed_token] = TypedToken(typed_token, token)
        else:
          pass                                                    # Do nothing,

  def loadModel(self):
    """
    - First load the model from json
    - Collect all nodes
    - Collect tokens                               # TODO: figure out if needed
    - Collect typed tokens with reactions
    - Collect all arcs
    - Collect the used mechanisms               # TODO: figure out if necessary
    """
    with open(self.model_file, 'r') as infile:
      self.raw_model_json = json.load(infile, object_pairs_hook=OrderedDict)
    nw_token_to_typed_token = self.ontology.token_typedtoken_on_networks
    token_to_typed_token = {}
    for nw, tokens_tt in nw_token_to_typed_token.items():
      for token, typed_tokens in tokens_tt.items():
        token_to_typed_token[token] = typed_tokens
    for label, node in self.raw_model_json['nodes'].items():
      if node['class'] == 'node_composite':
        continue                                       # TODO: Control this one
      self.nodes[label] = Node(label, node)           # Label and dict unpacked
      if node['class'] == 'node_interface':
        continue
      for token in node['tokens']:
        if token in token_to_typed_token.keys():
          for typed_token in token_to_typed_token[token]:
            self.typed_tokens[typed_token].addInstances(node['tokens'][token])
      if 'injected_conversions' in node.keys():
        for token_c, conversions in node['injected_conversions'].items():
          typed_tokn = token_to_typed_token[token_c]
          self.typed_tokens[typed_tokn].addConversions(conversions)

    for label, arc in self.raw_model_json['arcs'].items():
      self.arcs[label] = Arc(label, arc)

    for label, index_set in self.getIndices().items():
      self.indices[label] = IndexSet(label, index_set, self)

    self.networks_w_namednw_dict = self.raw_model_json['named_networks']

  def getIndices(self):
    indices = self.ontology.readIndices()
    return indices

  def makeIndexNode(self, ind):
    """
    Node index set
    """
    mapping = []
    blocking = []
    con = '_conversion'
    for (label, curnode) in self.nodes.items():
      conversions = []
      if curnode.__dict__['class'] == 'node_interface':
        mapping.append(int(label))
        blocking.append(0)
        continue
      if curnode.type == 'dynamic':                            # HACK: Not sexy
        conversions = curnode.injected_conversions.keys()
      if ind.type == 'block_index':
        mapping.append(int(label))
        size_of_block = 0
        split_block = ind.label.split(' & ')       # Split index set by name
        td = self.ontology.nw_token_typedtoken[
                       curnode.network].items()
        # TYPED TOKENS:
        for tokn, typed_token in [(t, tt) for t, ts in td for tt in ts]:
          if split_block[1] == typed_token and tokn in curnode.tokens.keys():
            size_of_block += len(curnode.tokens[tokn])

          if split_block[1] == typed_token + con and tokn in conversions:
            # TOKEN CONVERSIONS
            if curnode.type == 'dynamic':
              if curnode.injected_conversions[tokn]:
                size_of_block += len(curnode.injected_conversions[tokn])

        blocking.append(size_of_block)

      elif ind.type == "index":            # This case is a regular index set
        mapping.append(int(label))
        blocking.append(1)
      else:
        mapping.append(int(label))
        blocking.append(1)
    return mapping, blocking

  def makeIndexArc(self, ind):
    """
    Arc index set
    """
    mapping = []
    blocking = []
    for (label, curarc) in self.arcs.items():
      # if curarc.network in ind.layer:             # Check if available in layer
      if ind.type == 'block_index':
        split_block = ind.label.split(' & ')       # Split index set by name
        td = self.ontology.nw_token_typedtoken[
                       curarc.network].items()
        mapping.append(int(label))
        size_of_block = 0
        for token, typed_token in [(t, tt) for t, ts in td for tt in ts]:
          if split_block[1] == typed_token and token in [curarc.token]:
            size_of_block += len(curarc.typed_tokens)
        blocking.append(size_of_block)
      elif ind.type == "index":            # This case is a regular index set
        mapping.append(int(label))
        blocking.append(1)
      else:
        print(ind.type)
        print('BUG!!  Not an index type')
    return mapping, blocking

  def makeIndexTypedToken(self, ind):
    """
    Typed token or conversion index set
    """
    mapping = []                                          # mapping to superset
    blocking = []                                            # size of blocking
    typed_t_species = []                                 # instances of species
    conversions = []                                     # conversions in model

    # First loop through nodes to see what is used
    # TODO: This is hard wired, should be prettier. Not look for specific token
    for (label, curnode) in self.nodes.items():        # First search the nodes
      if curnode.network in ind.network:            # Check if available in layer
        if 'mass' in curnode.tokens.keys():
          typed_t_species += curnode.tokens['mass']
          if curnode.type == 'dynamic':
            if 'mass' in curnode.injected_conversions.keys():
              conversions += curnode.injected_conversions['mass']

    if ind.label in self.typed_tokens.keys():     # Then populate index
      for i, typedToken in enumerate(set(typed_t_species)):
        mapping.append(i)
        blocking.append(1)
    elif ind.label.split('_')[0] in self.typed_tokens.keys():
      for i, conversion in enumerate(set(conversions)):
        mapping.append(i)
        blocking.append(1)

    return mapping, blocking

  def makeIndex(self):
    """
    Not sure on how to do this one...
    Idea is to generate in fundamentally as objects.  So list of objects is not
    fundamental enough.

    Should this function be an object itself? Could be fun in that case...

    How does it work as a function?
    List of indexes?

    I do not know...

    After sleeping on it I figured out that the index sets already exist. It is
    "just" the dimensions of the index sets that are missing. The sizes must be
    determined by the already generated graph.

    All the lists below have to be populated in this function. Need to know
    blocking and mapping over to supersets.
    """

    # Have to do a set at the time........
    for ind in IndexSet.___refs___:
      # print('inde: ', ind)
      mapping = []
      blocking = []
      if ind.outer == 'node' or ind.label == 'node':       # It is of node type
        mapping, blocking = self.makeIndexNode(ind)
      elif ind.outer == 'arc' or ind.label == 'arc':       # Arc type index set
        mapping, blocking = self.makeIndexArc(ind)
      else:                                                      # Typed tokens
        mapping, blocking = self.makeIndexTypedToken(ind)
      ind.makeMappingBlocking(mapping, blocking)

  def makeDot(self):
    with open(self.dotfile, 'msg_box') as dotfile:
      # PREAMBLE
      date = '#\t When:    {}'.format(strftime("%Y-%m-%d %H:%M:%S"))
      reason = '#\t Why:     Output to dot language'
      purpose = '#\t Purpose: Dot graph for Model'
      author = '#\t Author:  Arne Tobias Elve'
      dotfile.write('#'*79+'\n')
      dotfile.write('{0:78}#\n'.format(purpose))
      dotfile.write('{0:78}#\n'.format(author))
      dotfile.write('{0:78}#\n'.format(date))
      dotfile.write('{0:78}#\n'.format(reason))
      dotfile.write('#'*79+'\n')
      dotfile.write('digraph G {\n')
      dotfile.write('rankdir = "LR"\n')
      # NODES
      for label, node in self.nodes.items():
        # if node[1]
        style = 'filled'                                        # Default shape
        fillcolor = 'Tomato'                                    # Default color
        if node.type == 'node_composite':
          continue                                          # Check if grouping
        elif node.type == 'constant':                               # Reservoir
          fillcolor = 'Gold1'
        elif node.type == 'event':                       # Event-dynamic system
          fillcolor = 'Snow4'
        node_string = '{} [style = {}, label = "{}" fillcolor = {}];\n'
        dotfile.write(node_string.format(label, style, label, fillcolor))
      # EDGES
      for label, arc in self.arcs.items():
        arcColor = 'Black'
        arrowtype = 'arrowhead = normal'
        if arc.token == 'energy':
          arcColor = 'Firebrick1'
        arc_string = '{} -> {} [label = "{}", {}, color = {}];\n'
        dotfile.write(arc_string.format(arc.source, arc.sink, arc.name,
                                        arrowtype, arcColor))
      # FINISH FILE
      dotfile.write('}')

  def produceDot(self):
    os.system('dot -Tpdf {} > {}'.format(self.dotfile, self.pdffile))


class Node(object):
  """
  Node class
  """
  ___refs___ = []

  def __init__(self, label, dict):
    self.___refs___.append(self)
    self.label = label
    self.genre = 'node'
    self.conversions = {}
    self.__dict__.update(**dict)
    # print(dict)

  def __str__(self):
    return(self.label)


class Arc(object):
  """
  Each arc included in the graph.
  """
  ___refs___ = []

  def __init__(self, label, dict):
    """
    Initiation of an arc object

    Args:
      label: The name of the arc used in the program
      dict:  Dictionary containing all the defined properties.
    """
    self.___refs___.append(self)
    self.label = label
    self.genre = 'arc'
    self.__dict__.update(**dict)
    # print(dict)

  def __str__(self):
    return(self.label)


class TypedToken(object):
  """
  This class contain information about species in the graph
  """
  ___refs___ = []

  def __init__(self, token, label):
    self.___refs___.append(self)
    self.genre = "typed_token"
    self.label = label
    self.token = token
    self.instances = []
    self.conversions = []                                       # List of dicts
    # self.__dict__.update(**dict)

  def addInstances(self, instances):
    """
    Instance just a sigle string
    """
    for instance in instances:
      if instance not in self.instances:
        self.instances.append(instance)
        self.instances.sort()

  def addConversions(self, conversions):
    """
    Instance is a dictionary with reactants and conversions
    """
    for conv in conversions:
      reac, prod = conv.split(' --> ')
      conversion = {}
      conversion['label'] = conv
      conversion['reactants'], conversion['products'] = eval(reac), eval(prod)
      if conversion not in self.conversions:
        self.conversions.append(conversion)


class IndexSet(Model):
  """
  The index set class.

  To be combined  with the index set  already in OntoSim.  Started from scratch
  just because  it can provide some extra inspiration and the structure  of the
  original class is really simple. So that might be easier.
  """
  ___refs___ = []                               # All indexing sets in sequence

  def __init__(self, label, indexSet, model):
    """
    Init set from index dict

    Args:
      indexSet: Index dict

    Generate  the basic  implementation  of the index set.  The index  sets are
    generated  from  the configuration  file representing  the ontology  of the
    model.
    """
    self.___refs___.append(self)                  # Making a list of references
    self.model = model
    self.label = label
    self.outer = ''                        # Need to happen before dict update!
    self.__dict__.update(**indexSet)

    # The bookkeeping of the graph
    self.blocking = []                # Size of the inner blocking in the outer
    self.mapping = []              # Representation into the superset or itself

  def makeMappingBlocking(self, mapping, blocking):
    """
    Combine the methods make blocking and make mapping

    Args:
      mapping: List with the indices into the superset
      blocking: List with the sizes of the inner blocks

    Populate the blocking and the mapping for the current index set.
    """
    self.makeBlocking(blocking)
    self.makeMapping(mapping)

  def makeBlocking(self, blocking = []):
    """
    Define sizes of the inner blocks.

    Args:
      blocking: List with the sizes of the inner blocks

    Populate the blocking variable. This maps over to the mapping
    """
    self.blocking = blocking

  def makeMapping(self, mapping = []):
    """
    Make mapping over to super set.

    Args:
      mapping: List with the indices into the superset

    If no super set the set is defined as its own super set.
    """
    self.mapping = mapping

  def printable(self):
    """
    Printable version explaining to what object in the model it relates. Should
    also consider if it can  be written out  in a matrix form.  Could be better
    than printing with zeros.
    """
    nodes = self.model.nodes
    arcs = self.model.arcs
    typed_tokens = self.model.typed_tokens
    # nw_tk_ttk = self.model.ontology
    nw_token_to_typed_token = self.model.ontology.token_typedtoken_on_networks
    token_to_typed_token = {}
    for nw, tokens_tt in nw_token_to_typed_token.items():
      for token, typedtokens in tokens_tt.items():
        token_to_typed_token[token] = typedtokens
    representation = []
    if self.type == "index" and self.label.startswith('node'):
      for i in self.mapping:
        representation.append(nodes[i].name)   #(nodes[str(i)].name)   #HAP:  str --> int
    elif self.type == 'block_index' and self.label.startswith('node'):

      for i in self.mapping:
        node = nodes[i]  #nodes[str(i)] #HAP: str --> int
        for token, typed_tokens in token_to_typed_token.items():
          for typed_token in typed_tokens:
            if "token" not in dir(node):
              representation.append('{} {}'.format(node.name, 'NONE'))
              continue
            if token in node.tokens.keys() and typed_token == self.inner:
              for inst in node.tokens[token]:
                representation.append('{} {}'.format(node.name, inst))

    elif self.type == "index" and self.label.startswith('arc'):
      for i in self.mapping:
        arc = arcs[str(i)]
        source = nodes[str(arc.source)].name
        sink = nodes[str(arc.sink)].name
        representation.append('{}|{} {}'.format(source, sink, arc.token))
    elif self.type == 'block_index' and self.label.startswith('arc'):
      for i in self.mapping:
        arc = arcs[str(i)]
        source = nodes[str(arc.source)].name
        sink = nodes[str(arc.sink)].name
        for token, typed_tokens in token_to_typed_token.items():
          for typed_token in typed_tokens:
            if "token" not in dir(arc):
              representation.append('{} {}'.format(arc.name, 'NONE'))
              continue
            if token in arc.token and typed_token == self.inner:
              for inst in arc.tokens[token]:
                representation.append('{}|{} {} {}'.format(source, sink, arc.token, inst))
            if len(arc.typed_tokens) == 0:             # TODO: Implement new system
             representation.append('{}|{} {}'.format(source, sink, 'NONE'))
    elif self.type == "index" and self.label in typed_tokens:
      for tt in typed_tokens[self.label].instances:
        representation.append(tt)
    elif self.type == "index" and self.label == 'species_conversion':
      for ttc in typed_tokens['species'].conversions:
        representation.append(ttc['label'])
    return representation

  def __str__(self):                         # The name in the current language
    return self.label

  def __repr__(self):                      # The name in the configuration file
    return self.label
