#!/usr/local/bin/python3
# encoding: utf-8

"""
@summary:      Model factory module
@contact:      arne.t.elve(at)ntnu.no
@requires:     Python 3 or higher
@since:        2018-10-01
@version:      1.0
@change:       2019-04-25
@authors:      Arne Tobias Elve, Heinz A Preisig and Andreas Johannesen
@copyright:    2018 Elve, AT All rights reserved.
"""
import os

from jinja2 import Environment             # sudo apt-get install python-jinja2
from jinja2 import FileSystemLoader
import copy

from Common.common_resources import invertDict, getData, putData

from OntologyBuilder.OntologyEquationEditor.resources import LANGUAGES
from OntologyBuilder.OntologyEquationEditor.resources import FILE_EXTENSIONS
from OntologyBuilder.OntologyEquationEditor.resources import EMPTY_EQ
from OntologyBuilder.OntologyEquationEditor.resources import NEW_EQ
from OntologyBuilder.OntologyEquationEditor.resources import NEW_VAR
from OntologyBuilder.OntologyEquationEditor.resources import CODE

from OntologyBuilder.OntologyEquationEditor.variable_framework import IndexStructureError
from OntologyBuilder.OntologyEquationEditor.variable_framework import Expression
from OntologyBuilder.OntologyEquationEditor.variable_framework import UnitError
from OntologyBuilder.OntologyEquationEditor.variable_framework import Variables
# from OntologyBuilder.OntologyEquationEditor.variable_framework import Equations
from OntologyBuilder.OntologyEquationEditor.variable_framework import VarError
# from OntologyBuilder.OntologyEquationEditor.variable_framework import Indices
from OntologyBuilder.OntologyEquationEditor.variable_framework import Units

from .model_framework import Model

from collections import OrderedDict

from time import strftime                                    # Library for time
import numpy as np
from shutil import copyfile

# GLOBAL CONSTANTS
UTILITIES_FILE_NAME = 'funcUtils'               # File name of exstra functions
OBJ_VAR_CLASSES = ['constant', "network"]


class ModelFactory(object):
  """
  Main class for model factory.
  """

  def __init__(self, ontology, model_name, language, model_loc):
    self.ontology = ontology
    self.mod_name = model_name
    self.language = language
    self.model_loc = model_loc

    self.variable_instantiate_file = os.path.join(self.model_loc, 'cases',
                                                  'instansiate_variables.json')

    self.variables = Variables()
    self.equations = Equations()
    self.equations.linkVariables(self.variables)
    self.variables.linkEquations(self.equations)       # for renaming variables
    self.indices = Indices()
    self.variables.linkIndices(self.indices)           # for renaming variables
    self.aliases_defined = False
    self.system_setup()                                 # SHOULD BE DONE IN EDITOR
    self.setupSimulation()
    self.model = Model(self.ontology_location, self.model_name, self.ontology)
    self.mod_index = self.model.indices                    # Shortening of name
    self.file_ending = FILE_EXTENSIONS[self.language]
    # self.__compile(self.language)
    # self.define_base_network_variables()
    # self.new_dump_network_variables_to_file()
    self.network_variables = getData(os.path.join(self.ontology.onto_path, 'network_matrices.json'))
    # print(self.network_variables)
    self.variables_dict = getData(os.path.join(self.ontology.onto_path, 'variables.json'))
    self.nw_nnw_dict = self.merge_nw_nnw()
    self.object_oriented = False
    self.flow_system = True

    # self.new_dump_network_variables_to_file()
    # self.system_setup()
    # self.__variableSetup()
    # self.calculation_order()

  def system_setup(self):
    self.model_name = self.mod_name + FILE_EXTENSIONS['json']
    print( "SETTING UP SYSTEM:\n{}".format(self.model_name) )
    # location = os.path.join(DIRECTORIES["ontology_repository"], self.ontology_name)
    # print(location)
    # self.ontology = OntologyContainer( self.ontology_name ) # DONE
    self.ontology_location = self.ontology.onto_path
    self.rules = self.ontology.rules
    self.ontology_in_hiearchy = self.ontology.ontology_in_hiearchy
    self.ontology_in_hiearchy_inverse = invertDict(
        self.ontology.ontology_in_hiearchy)
    self.networks = self.ontology.networks
    self.variable_types_on_networks = self.ontology.variable_types_on_networks

    # MANUALLY FILLED IN! :(
    self.loggend_variables = []
    self.root_variables = []
    self.equations_not_included = []
    self.state_variables = ['z', 'v']
    self.state_diffs = ['v', 'a']
    self.variable_frame_start = ['t_start']
    self.variable_frame_end = ['t_end']
    self.state_equations = ['3', '2']
    self.differential_equations = ['1', '1']

  def calculation_order(self):
    # ABOVE IS MANUALLY FILLED IN AND MODEL SPECIFIC :(
    # TODO: make editor to automate this step

    variables = self.variables
    equations = self.equations

    vars_eqs = {}
    for v, variable in variables.items():
      s = copy.copy(variable.label)
      vars_eqs[s] = copy.copy(variable.equation_list)
    equs_incidence = OrderedDict()
    for e in equations:
      id = (copy.copy(equations[e].equation_ID))
      equs_incidence[id] = (copy.copy(equations[id].lhs),
                            copy.copy(equations[id].incidence_list))

    vars = []
    d_equs = []
    # Equation calculation sequence
    for var in self.state_diffs:
      self.traverse_eqs(vars_eqs, variables, equs_incidence, vars, d_equs, var)
    self.calculation_sequence = self.clean_calculation_sequence(d_equs)

    # Variables not included
    self.variables_not_included = []
    for variable in variables:
      if variable not in vars:
        self.variables_not_included.append(variable)

    # Compile variables
    # self.__compileVariables()
    # for enumber, equation in self.equations.items():
    #   equation.included = True
    #   if enumber in self.equations_not_included:
    #     equation.included = False
    #   elif enumber in self.constantEquations:
    #     equation.equation_type = 'constant'
    #   elif enumber in self.differential_equations:
    #     equation.equation_type = 'differential'
    #   elif enumber in self.state_equations:
    #     equation.equation_type = 'integral'

  def clean_calculation_sequence(self, eqs_sequence):
    new_seq = []
    for id in reversed(eqs_sequence):
      if id not in new_seq:
        new_seq.append(id)
    return new_seq

  def traverse_eqs(self, vars_eqs, variables, equs, vars, d_equs, var_symbol):
    var = variables[var_symbol]
    vars.append(var.label)
    for id in vars_eqs[var_symbol]:        # Equation and alternative equations
      if id not in self.equations_not_included and var_symbol not in self.state_variables:
        # print(dir(self.equations[id]))
        d_equs.append(id)              # Added equation to calculation sequence
        for next_var in self.equations[id].incidence_list:
          if next_var == var.label:             # TODO: CHECK! ROOT EXPRESSION I HOPE
            continue
          else:
            self.traverse_eqs(vars_eqs, variables, equs, vars, d_equs, next_var)
      if var_symbol in self.state_variables:  # Also include variables in integ
        for var_frame in self.equations[id].incidence_list:
          d_equs.append(id)            # Added equation to calculation sequence
          vars.append(var_frame)
    return

  def produce_code(self):
    self.__compile(self.language)

  def merge_nw_nnw(self):
    """
    Merge networks from ontology with networks and  named networks from  model.
    """
    nw_nnw = {}
    nw_dict = self.ontology.heirs_network_dictionary
    nnw_dict = self.model.networks_w_namednw_dict
    for label, things in nw_dict.items():
      nw_nnw[label] = things
      if label in nnw_dict.keys():
        nw_nnw[label] = set(nnw_dict[label])
    return nw_nnw

  # def load_network_variables_from_file(self):
  #   """
  #   Read in network variables from json file
  #   """
  #   file = self.ontology.read
  #   # variable_file_name = 'variables/network_variables.json'
  #   # file_spec = '{}/{}'.format(self.model_loc, variable_file_name)
  #   network_dict = getData(file_spec)
  #   return network_dict

  # def new_dump_network_variables_to_file(self):
  #   """
  #   Write network variables with info to file
  #
  #   Used from model factory
  #   """
  #   variable_file_name = 'variables/network_variables.json'
  #   file_spec = '{}/models/{}/{}'.format(self.ontology_location, self.mod_name,
  #                                        variable_file_name)
  #   self.define_base_network_variables()
  #   # print(network_dict)
  #   putData(self.base_network_variables_dict, file_spec)

  # def define_base_network_variables(self):
  #   """
  #   Define the base network variables from which we can pick the networks......
  #   """
  #   graph_base_vars = {}
  #   CONSTITUENT_LIST = ['node', 'arc']
  #   # print(dir(self.ontology))
  #   onto_dict = self.ontology.ontology_tree
  #   for nw in self.ontology.networks:
  #     # START WITH INCIDENCE MATRICES
  #     # F_network_token_mechanism
  #     in_name_str = 'F_{}_{}_{}'
  #     for token in self.ontology.tokens_on_networks[nw]:
  #       for mechanism in self.ontology.arc_info_allnetworks_dict[nw][token]:
  #         label = in_name_str.format(nw, token, mechanism)
  #         graph_base_vars[label] = copy.copy(STRUCTURES_Vars_Equs["network"])
  #         graph_base_vars[label]["label"] = label
  #         graph_base_vars[label]["network"] = nw
  #         graph_base_vars[label]["type"] = 'incidence'
  #         graph_base_vars[label]["token"] = token
  #         graph_base_vars[label]["mechanism"] = mechanism
  #
  #     # SECOND PROJECTIONS
  #     # P_network_element_typedtoken
  #     pr_name_str = 'P_{}_{}_{}'
  #     for thing in CONSTITUENT_LIST:
  #       for token, typed_token in self.ontology.typed_tokens_on_tokens.items():
  #         label = pr_name_str.format(nw, thing, typed_token)
  #         graph_base_vars[label] = copy.copy(STRUCTURES_Vars_Equs["network"])
  #         graph_base_vars[label]["label"] = label
  #         graph_base_vars[label]["network"] = nw
  #         graph_base_vars[label]["type"] = 'projection'
  #         graph_base_vars[label]["token"] = token
  #         graph_base_vars[label]["constituent"] = thing
  #         graph_base_vars[label]["typed_token"] = typed_token
  #
  #     # THIRD STOICHIOMETRIC MATRICES
  #     # N_network_typedtoken
  #     st_name_str = 'N_{}_{}'
  #     for token, typed_token in self.ontology.typed_tokens_on_tokens.items():
  #       label = st_name_str.format(nw, typed_token)
  #       graph_base_vars[label] = copy.copy(STRUCTURES_Vars_Equs["network"])
  #       graph_base_vars[label]["label"] = label
  #       graph_base_vars[label]["network"] = nw
  #       graph_base_vars[label]["type"] = 'stoichiometric'
  #       graph_base_vars[label]["token"] = token
  #       graph_base_vars[label]["typed_token"] = typed_token
  #       graph_base_vars[label]["typed_token_conversion"] = typed_token
  #
  #     # FORTH SELECTIONS
  #     # S_network_consitiuent_type_distribution
  #     s_name_str = 'S_{}_{}_{}'
  #     for thing in CONSTITUENT_LIST:
  #       # if thing == 'node':
  #       for type, dists in onto_dict[nw]['structure'][thing].items():
  #         for dist in dists:
  #           label = s_name_str.format(nw, thing, type, dist)
  #           graph_base_vars[label] = copy.copy(STRUCTURES_Vars_Equs["network"])
  #           graph_base_vars[label]["label"] = label
  #           graph_base_vars[label]["network"] = nw
  #           graph_base_vars[label]["type"] = 'selection'
  #           graph_base_vars[label]["constituent"] = thing
  #           graph_base_vars[label]['constituent_type'] = type
  #           graph_base_vars[label]["distribution"] = dist
  #
  #   self.base_network_variables_dict = graph_base_vars

  def setupSimulation(self):
    # # setup variable indexing
    for nw in self.networks:
      variable_types = self.variable_types_on_networks[nw]
      self.variables.setTypes(variable_types, nw)
      those_who_have_it = self.ontology.heirs_network_dictionary[nw]
      self.variables.setThoseWhoInherit(those_who_have_it, nw)

    # get variables, indices and equations or initialise
    variables = self.ontology.readVariables()                       # variables
    for v in variables:
      if not self.variables.existSymbol(v):        #fixme: namespace issue. Information is in variables
        variables[v]["label"] = v
        variables[v]["units"] = Units(ALL=variables[v]["units"])
        self.variables.addVariable(**variables[v])

    indices = self.ontology.readIndices()                             # indices
    for i in indices:
      self.indices.add(i, **indices[i])

    equations = self.ontology.readEquations()                       # equations
    for e in equations:
      self.equations.addEquation(equation_ID=str(e), **equations[e])

  def __compile(self, language):
    var_list = self.variables.getVariableList()
    expression = Expression(self.variables,
                            self.indices,
                            language=language)
    if self.object_oriented:
      obj_vars = self.object_variables()
      for label, variable in self.variables.items():
        if label in obj_vars:
          var_aliases = self.variables[label].aliases    # Srange due to memory
          ind = [el[0] for el in var_aliases].index(language)
          var_aliases[ind][1] = CODE[language]["obj"].format(
             var_aliases[ind][1]
           )

    for index_symbol, index in self.indices.items():     # Sorting out indexset
      indalias = index['aliases'][language]
      # indalias = [item for item in index["aliases"] if item[0] == language][0]
      if self.object_oriented:
        indali = CODE[language]["obj"].format(indalias)
        self.indices[index_symbol]["aliases"][language] = indali   # Local change
      else:
        indali = indalias
      self.indices[index_symbol]["compiled"] = indali

    for variable_symbol in var_list:
      v = self.variables[variable_symbol]
      if v.label != NEW_VAR:
        v.compiled = v.aliases[language]
        v.compiled_index_list = self.__getCompiledIndexList(variable_symbol,
                                                            language)
    for e in self.equations:
      expr = self.equations[e].rhs

      variable_symbol = self.equations[e].lhs
      if expr not in [EMPTY_EQ, NEW_EQ]:
        try:
          res = expression(expr)
        except (UnitError, IndexStructureError, VarError) as _m:
          print('checked expression failed %s -- %s' % (variable_symbol, _m))
        lhs = expression(variable_symbol)
        self.equations[e].latex = str(res)
        self.equations[e].number = e
        self.equations[e].lhs_compiled = lhs
        self.equations[e].equation_type = res.equation_type

    if language in LANGUAGES['code_generation']:
      if self.object_oriented:
        self.__makeSimulation_new(language)
      else:
        self.__makeSimulation(language)           # Write out simulation file

  def size_of_variable(self, variable):
    """
    Calculate size of variable
    """
    index_structures = variable.index_structures
    if not index_structures:
      return 1
    mapping = [self.mod_index[ind].mapping for ind in index_structures]
    blocking = [self.mod_index[ind].blocking for ind in index_structures]
    size = []
    for i in range(len(mapping)):
      if mapping[i] and blocking[i]:
        length = 0
        for blk in blocking[i]:
          if blk == 0:
            length += 1
          else:
            length += blk
        size.append(length)
      else:
        return None
    return size

  def matrix_string_zeros(self, size, prefix = ''):
    """
    Take in size of matrix and return it as string
    """
    if size == 1:                                                 # Not indexed
      return np.array2string(np.array(0.))
    if len(size) == 1:
      size.append(1)
    mat = np.zeros(size)
    return np.array2string(mat, prefix = prefix, sign = ' ', separator = ',')

  def matrix_string_ones(self, size, prefix = ''):
    """
    Take in size of matrix and return it as string
    """
    if size == 1:                                                 # Not indexed
      return np.array2string(np.array(1.))
    if len(size) == 1:
      size.append(1)
    mat = np.ones(size)
    return np.array2string(mat, prefix = prefix, sign = ' ', separator = ', ')

  def matrix_to_string(self, mat, prefix = ''):       # Fixed sign and seprator
    return np.array2string(mat, prefix = prefix, sign = ' ', separator = ', ')

  def matrix_to_str_w_line_comments(self, mat, row_comments, prefix = ''):
    string_mat = self.matrix_to_string(mat, prefix)
    row_split = string_mat.split('\n')
    if len(row_split) == len(row_comments):
      string = ''
      rest = -2
      first = True
      for row, comment in zip(row_split, row_comments):
        spaces = 79 - len(row) - len(comment) - rest
        if first:               # to many hacks, different length of first line
          rest -= 4
          first = False
        formatted_string = '{v: <{msg_box}} # {r}\n'
        string += formatted_string.format(v = row, w = spaces, r = comment)
      return string
    else:
      return string_mat

  def populateNetworkVariable(self, variable):
    # nwvars = self.network_variables
    # if
    # print(type(nwvars['incidence_matrices'].keys()))
    print(variable, dir(variable))
    mat = np.array([])                                      # empty numpy array

    if variable.label.startswith('N'):
      # var_info = nwvars['conversion_ratio_matrices'][variable.label]
      typed_token = variable.index_structures[0]                   # Quick hack
      mat = self.makeStoichiometricMatrix(variable, typed_token)
    elif variable.label.startswith('F'):
      type, token, mechanism = variable.label.split("_")
      # token = var_info["token"]
      # mechanism = var_info["transfer_mechanism"]
      network = variable.network
      mat = self.makeIncidenceMatrix(variable, token, mechanism, network)
    elif variable.label in nwvars["projection_matrices_arc"].keys():
      var_info = nwvars['projection_matrices_arc'][variable.label]
      token = var_info['index_structures'][0]
      typed_token = var_info['index_structures'][1]
      mat = self.makeProjection(variable, token, typed_token, self.model.arcs)
    elif variable.label in nwvars["projection_matrices_node"].keys():
      var_info = nwvars['projection_matrices_node'][variable.label]
      token = var_info['index_structures'][0]
      typed_token = var_info['index_structures'][1]
      mat = self.makeProjection(variable, token, typed_token, self.model.nodes)
    elif variable.label in nwvars["projection_matrices_conversion"].keys():
      pass # FIX THIS
      #
      # var_info = nwvars['projection_matrices_conversion'][variable.label]
      # token = var_info['index_structures'][0]
      # typed_token = var_info['index_structures'][1]
      # mat = self.makeProjection(variable, token, typed_token, self.model.nodes)

      #   model = self.model
      #   token = 'mass'
      #   typed_token = 'species'
      #   mat = self.makeProjectionConversion(variable, token,
      #                                       typed_token, model.nodes)

    # if variable.type == "network":                            # Already checked

      # pass
    # TODO: Figure out what to do with projection conversions.... New type?
    # if variable.label in self.network_aliases.keys():
    #   long_name = self.network_aliases[variable.label]
    #   var_info = self.base_network_variables_dict[long_name]
    #   if var_info["type"] == 'incidence':
    #   elif var_info["type"] == 'projection':
    #     token = var_info["token"]
    #     typed_token = var_info["typed_token"]
    #     if var_info["constituent"] == 'node':
    #       things = self.model.nodes
    #     elif var_info["constituent"] == 'arc':
    #       things = self.model.arcs
    #   elif var_info["type"] == 'stoichiometric':
    # else:
    #   pass
    # if variable.label == 'Prnr':
    # elif variable.label == 'Snnr':
    #   nodes = self.model.nodes
    #   token = 'mass'
    #   typed_token = 'species'
    #   mat = self.makeSelectionMatrixConversion(nodes, token, typed_token)

    return mat

  def makeStoichiometricMatrix(self, var, typed_token):
    """
    Generate the stoiciometric matrix
    """
    # TODO: Change stoichiometric  matrix to  include the integers of reactions
    model = self.model
    size = self.size_of_variable(var)
    mat = np.zeros(size)
    for i, instance in enumerate(model.typed_tokens[typed_token].instances):
      for j, conversion in enumerate(model.typed_tokens[
                                     typed_token].conversions):
        if instance in conversion['reactants']:
          mat[i, j] = -1.
        elif instance in conversion['products']:
          mat[i, j] = 1.
    if typed_token == var.index_structures[1]:                 # Check sequence
      return np.transpose(mat)
    return mat

  def makeIncidenceMatrix(self, variable, token, mechanism, network):
    """
    Generate incidence matrix based on variable,  token,  network and mechanism
    of transport.

    Args:
      variable: variable object
      token: transported token
      network: which network
      mechanism: mechanism

    Returns:
      mat: Incidence matrix
    """
    model = self.model
    size = self.size_of_variable( variable )
    mat = np.zeros(size)
    for i, (label, node) in enumerate(model.nodes.items()):
      if node.named_network in self.nw_nnw_dict[network]:
        for j, (arc_label, arc) in enumerate(model.arcs.items()):
          if str(node.label) == str(arc.source) and mechanism == arc.mechanism:
            mat[i, j] = -1.
          elif str(node.label) == str(arc.sink) and mechanism == arc.mechanism:
            mat[i, j] = 1.
    return mat

  def makeProjection(self, variable, token, typed_token, constituent_dict):
    """
    Make projection of typed_token to constituent

    Args:
      variable: variable
      typed_token: typed version of the token
      constituent_list: list of pricipal components it sould map for. Either
                        node list or arc list.
    """
    model = self.model
    typed_token_list = model.typed_tokens[typed_token].instances
    thing_list = []
    for label, thing in constituent_dict.items():
      if thing.genre == 'node':
        if thing.__dict__['class'] == 'node_interface':
          thing_list += ['']
        elif token in thing.tokens.keys():
          thing_list += thing.tokens[token]
        else:
          thing_list += ['']
      elif thing.genre == 'arc':
        if token == thing.token:
          thing_list += thing.typed_tokens
        else:
          thing_list += ['']
    mat = np.zeros((len(typed_token_list), len(thing_list)))
    for i, typed_token_instance in enumerate(typed_token_list):
      for j, thing in enumerate(thing_list):
        if typed_token_instance == thing:
          mat[i, j] = 1.
    return mat

  def interface_matrix(self):
    model = self.model
    node_list = model.nodes
    arc_list = model.arcs

    mat_left_in = np.zeros((len(node_list), len(arc_list)))
    mat_left_out = np.zeros((len(node_list), len(arc_list)))
    mat_right_in = np.zeros((len(node_list), len(arc_list)))
    mat_right_out = np.zeros((len(node_list), len(arc_list)))
    nodelist_label = {}
    for i, (nlabel, node) in enumerate(node_list.items()):
      nodelist_label[nlabel] = i
    for i, (nlabel, node) in enumerate(node_list.items()):
      for j, (alabel, arc) in enumerate(arc_list.items()):
        if node.__dict__['class'] == 'node_interface':
          if str(arc.sink) == nlabel:
            mat_left_out[i, j] = 1
            k = nodelist_label[str(arc.source)]
            mat_left_in[k, j] = 1
          elif str(arc.source) == nlabel:
            mat_right_in[i, j] = 1
            k = nodelist_label[str(arc.sink)]
            mat_right_out[k, j] = 1
    return mat_left_in, mat_left_out, mat_right_in, mat_right_out

  def makeSelectionMatrixConversion(self, nodes, token, typed_token):
    # TODO: test with two reactions in nodes
    model = self.model
    node_list = model.nodes
    node_conversion_list = []
    label_list = []
    for label, node in node_list.items():
      if token in node.tokens.keys() and node.type == 'dynamic':
        try:
          node_conversion_list += node.injected_conversions[token]
          label_list += [label for _ in node.injected_conversions[token]]
          # label_list.append()
        except:                         # HACK: I should be shot for doing this
          node_conversion_list += ['']                        # Empty node list
          label_list.append('')
      else:
        node_conversion_list += ['']
        label_list.append('')
    mat = np.zeros((len(node_list), len(label_list)))
    for i, (label, node) in enumerate(node_list.items()):
      for j, node_label in enumerate(label_list):
        if node_label == label:
          mat[i, j] = 1.
    return mat

  def makeProjectionConversion(self, variable, token, typed_token,
                               constituent_dict):
    """
    Make projection of typed_token conversion to constituent

    Args:
      variable: variable
      typed_token: typed version of the token
      constituent_list: list  of pricipal components  it sould map for.  Either
                        node list or arc list.
    """
    model = self.model
    con_list = model.typed_tokens[typed_token].conversions
    thing_list = []
    for label, thing in constituent_dict.items():
      if thing.genre == 'node':                      # Only conversion in nodes
        if token in thing.tokens.keys() and thing.type == 'dynamic':
          try:
            thing_list += thing.injected_conversions[token]
          except:                       # HACK: I should be shot for doing this
            thing_list += ['']                               # Empty thing list
        else:
          thing_list += ['']
    mat = np.zeros((len(con_list), len(thing_list)))
    for i, token_conversion in enumerate(con_list):
      for j, thing in enumerate(thing_list):
        if token_conversion['label'] == thing:
          mat[i, j] = 1.
    return mat

  def object_variables(self):
    # RULE: CONSTANTS ARE SELF
    # RULE: STATES ARE SELF
    # RULE: LOGGEND VARIABLES ARE SELF
    # RULE: ROOT SOLVED VARIABLES ARE SELF
    obj_vars = []
    obj_vars += self.loggend_variables
    obj_vars += self.root_variables
    # obj_vars += self.state_variables
    for label, variable in self.variables.items():
      if variable.type in OBJ_VAR_CLASSES:
        obj_vars.append(label)
    return obj_vars

  def __compileVariables(self):
    """
    Variables
    Returns lists of compiled variables
    """
    state_variables = []
    state_diffs = []
    networks = []
    frames = []
    vars = []

    language = self.language

    instansiate_variables = {}

    self.obj_vars = self.object_variables()
    for key, var in self.variables.items():            # Grouping the variables
      size = self.size_of_variable(var)
      variable_type = var.type
      if not size or key in self.variables_not_included:        # Skip variable
        print('Variable not included:\tvar: {}'.format(var.label))
        continue
      if var.equation_list and variable_type in ['constant', "network"]:
        self.constantEquations.append(*var.equation_list)
        continue
      if variable_type in ['constant', 'frame', 'state']:
        # print(dir(var.units))
        units_pp = var.units.prettyPrint()
        doc_pp = var.doc
        if units_pp:
          units_doc_str = '{}, {}'.format(units_pp, doc_pp)
        else:
          units_doc_str = '{}'.format('Empty', doc_pp)
        instansiate_variables[key] = self.variable_dict(var)
        if var.compiled_index_list:               # Index sets need compilation
          string_version = self.matrix_string_zeros(size, prefix = '   ')
          mat = np.zeros(size)
          index = str(var.index_structures)                 # Convert to string
          rep = self.mod_index[var.index_structures[0]].printable()
          string_w_comments = self.matrix_to_str_w_line_comments(mat, rep,
                                                                 prefix = '  ')
          width = 79 - 16 - len(index) - len(var.compiled)       # 16 is others
          width2 = 79 - 8 - len(units_pp) - len(doc_pp)
          cons_str = '\n{0} = {ar}({v: <{msg_box}}  {com} {ind}\n  {st}  ){v: <{w2}} {com} {udoc}'
          cons_var_str = cons_str.format(var.compiled,
                                         com = CODE[language]["comment"],
                                         ar = CODE[language]["list"],
                                         st = string_w_comments,
                                         udoc = units_doc_str,
                                         ind = index,
                                         w2 = width2,
                                         w = width,
                                         v = '')
          # cons_str = '{0} = {ar}({v: <{msg_box}} {com} {ind}\n  {st})'
          # cons_var_str = cons_str.format(var.compiled,
          #                                ar = CODE[language]['list'],
          #                                st = string_w_comments,
          #                                ind = index,
          #                                msg_box = width,
          #                                v = '')
        else:
          string_version = self.matrix_string_zeros(size, prefix = '   ')
          index = 'none'
          width = 79 - 16 - len(index) - len(var.compiled)       # 16 is others
          width2 = 79 - 9 - len(units_pp) - len(doc_pp)
          cons_str = '\n{0} = {array}({val: <{msg_box}}  # {ind}\n  {st}\n  ) {h: <{w2}} {com} {udoc}'
          cons_var_str = cons_str.format(var.compiled,
                                         array = CODE[language]["list"],
                                         com = CODE[language]["comment"],
                                         udoc = units_doc_str,
                                         st = string_version,
                                         ind = index,
                                         w = width,
                                         w2 = width2,
                                         h = '',
                                         val = '')
        if variable_type in ['frame']:
          frames.append(cons_var_str)
        elif variable_type in ['state']:
          if var.label in self.state_variables:
            state_variables.append(cons_var_str)
          else:
            pass
        else:
          vars.append(cons_var_str)
      elif variable_type in ["network"]:
        mat = self.populateNetworkVariable(var)
        string_mat = self.matrix_to_string(mat, prefix = '      ')
        index = str(var.index_structures)                   # Convert to string
        width = 79 - 20 - len(index) - len(var.compiled)         # 16 is others
        if width < 1:
          width = 1
        nt_str = '{0} = {array}({val: <{msg_box}}  # {ind}\n  {st})'
        netw_var_str = nt_str.format(var.compiled,
                                     array = CODE[language]["list"],
                                     st = string_mat,
                                     ind = index,
                                     w = width,
                                     val = '')
        networks.append(netw_var_str)
    putData(instansiate_variables, self.variable_instantiate_file)
    return [state_variables, state_diffs, networks, frames, vars]

  def __compileEquations(self, language):
    # First classify the equations
    for enumber, equation in self.equations.items():
      equation.included = True
      if enumber in self.equations_not_included:
        equation.included = False
      elif enumber in self.constantEquations:
        equation.equation_type = 'constant'
      elif enumber in self.differential_equations:
        equation.equation_type = 'differential'
      elif enumber in self.state_equations:
        equation.equation_type = 'integral'

    # Then write them out
    eqs = []
    constant_equations = []
    selection_matrices = []
    derivativeEqs, stateEqs = [], []              # Store them as empty strings
    selection_str = '{0} = np.array({val: <{msg_box}}  # {ind}\n  {st})'
    for enumber in self.calculation_sequence:
      equation = self.equations[enumber]
      if equation.included:
        variable = self.variables[equation.lhs]
        size = self.size_of_variable(variable)
        if not size:                                       # Equation not valid
          continue
        index = str(variable.index_structures)
        eq = '{} = {}'.format(equation.lhs_compiled, equation.latex)
        if equation.equation_type == 'integral':
          stateEqs.append(equation.lhs_compiled.__str__())
          if self.variable_frame_start[0] in equation.incidence_list:
            frame_start = self.variable_frame_start[0]
          if self.variable_frame_end[0] in equation.incidence_list:
            frame_end = self.variable_frame_end[0]
          continue                 # Not writing this equation as a regular one
        elif equation.equation_type == 'differential':
          derivativeEqs.append(equation.lhs_compiled.__str__())
          selection_matrix = 'S_{}'.format(equation.lhs_compiled)
          if self.object_oriented:
            selection_matrix = CODE[language]["obj"].format(selection_matrix)
          mat = self.only_dynamic_nodes(variable)
          string_version = self.matrix_to_string(mat, prefix = '  ')
          width = 79 - 22 - len(index) - len(variable.compiled)
          if width < 1:
            width = 1                            # Cannot have negative lengths
          sel_str = selection_str.format(selection_matrix,
                                         val = '',
                                         ind = index,
                                         st = string_version,
                                         w = width)
          equation_with_selection = CODE[language]["."] % (selection_matrix,
                                                           equation.latex)
          eq = '{} = {}'.format(equation.lhs_compiled, equation_with_selection)
          selection_matrices.append(sel_str)
        elif equation.equation_type == 'constant':
          constant_equations.append(eq)
          continue                                   # Not adding any equations
        eqs.append(eq)
      else:
        print('Equation not included:\tvar: {}\tEq.#: {}'.format(
               equation.lhs_compiled, equation.number))
    return [eqs, constant_equations, selection_matrices, derivativeEqs,
            stateEqs, frame_end, frame_start]

  def __compileIndices(self):
    """
    Index sets
    """
    indices = []
    for index_symbol, index in self.indices.items():
      rs = "{} = IndexSet('{}', mapping = {}, blocking = {})"
      string_version = rs.format(index['compiled'],
                                 index_symbol,
                                 self.mod_index[index_symbol].mapping,
                                 self.mod_index[index_symbol].blocking)
      if self.mod_index[index_symbol].mapping:
        indices.append(string_version)
    return indices

  def writeSelections(self, language, selection_matrices):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    s = j2_env.get_template('template_selections.{}'.format(language)).render(
        date = '{}'.format(strftime("%Y-%m-%d %H:%M:%S")),
        equation_selections = selection_matrices,
        model_name = self.model_name,
      )
    sim_name = 'selections_{}{}'.format(self.mod_name, self.file_ending)
    f_name = os.path.join(self.model_loc, language, sim_name)
    f = open(f_name, 'msg_box+')
    f.write(s)
    f.close()

  def writeConstants(self, language, variables):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    s = j2_env.get_template('template_constants.{}'.format(language)).render(
        date = '{}'.format(strftime("%Y-%m-%d %H:%M:%S")),
        model_name = self.model_name,
        variables = variables,
      )
    sim_name = 'constants_{}{}_raw'.format(self.mod_name, self.file_ending)
    f_name = os.path.join(self.model_loc, self.language, sim_name)
    f = open(f_name, 'msg_box+')
    f.write(s)
    f.close()

  def writeInitialStatesAndFrames(self, langu, state_variables, frames):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    s = j2_env.get_template('template_initial_states.{}'.format(langu)).render(
        date = '{}'.format(strftime("%Y-%m-%d %H:%M:%S")),
        state_variables = state_variables,
        model_name = self.model_name,
        frames = frames,
      )
    sim_name = 'initial_states_{}{}_raw'.format(self.mod_name,
                                                self.file_ending)
    f_name = os.path.join(self.model_loc, self.language, sim_name)
    f = open(f_name, 'msg_box+')
    f.write(s)
    f.close()

  def __makeSimulation(self, language):
    self.constantEquations = []
    print('Making {} Simulation'.format(language))
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    self.calculation_order()
    # VARIABLES
    # self.__variableSetup()                                         # Replaced
    variables = self.__compileVariables()
    state_variables, state_diffs, networks, frames, vars = variables

    # EQUATIONS
    # self.__equationSelection()  # Need to happen after compile variables TODO
    equations = self.__compileEquations(language)  # Local compilation language
    [eqs, constant_equations, selection_matrices, derivatives, states,
     frame_end, frame_start] = equations

    # INDEX SETS
    indices = self.__compileIndices()

    states_start, states_end = self.length_of_state_variables()   # Length size
    states_labels = self.states_labels(states)

    self.writeConstants(language, vars)              # Write the constants file
    self.writeSelections(language, selection_matrices)   # Write the selections
    self.writeInitialStatesAndFrames(language, state_variables, frames)

    # Writing out the simulation file
    # print('HELKJ')
    # print(tuple(zip(self.state_variables, self.state_diffs)))
    # print('efsd')
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    # s = j2_env.get_template('template_expl_euler.py.j2'.format(language)).render(
    s = j2_env.get_template('template_main.python'.format(language)).render(
        states_and_lengths = tuple(zip(self.state_variables, states_start, states_end)),
        states_and_derivatives = tuple(zip(self.state_variables, self.state_diffs)),
        states_and_labels = tuple(zip(self.state_variables, states_labels)),
        date = '{}'.format(strftime("%Y-%m-%d %H:%M:%S")),
        constant_equations = constant_equations,
        state_variables = state_variables,
        derivatives = self.state_diffs,
        states = self.state_variables,
        author = os.uname().nodename,
        model_name = self.model_name,
        frame_start = frame_start,
        mod_name = self.mod_name,
        frame_end = frame_end,
        networks = networks,
        indices = indices,
        equations = eqs,
        frames = frames,
        vars = vars,
      )
    sim_name = 'simulation_{}{}'.format(self.mod_name, self.file_ending)
    f_name = os.path.join(self.model_loc, language, sim_name)
    f = open(f_name, 'msg_box+')
    f.write(s)
    f.close()
    print('Wrote out {} simulation to file:\n{}'.format(language, f_name))
    self.copySimulationOperators()

  def __makeSimulation_new(self, language):
    self.constantEquations = []
    print('Making {} Simulation'.format(language))
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    self.calculation_order()
    # INDEX SETS
    indices = self.__compileIndices()

    # VARIABLES
    # self.__variableSetup()                                         # Replaced
    variables = self.__compileVariables()
    state_variables, state_diffs, networks, frames, vars = variables

    # EQUATIONS
    # self.__equationSelection()  # Need to happen after compile variables TODO
    equations = self.__compileEquations(language)  # Local compilation language
    [eqs, constant_equations, selection_matrices, derivatives, states,
     frame_end, frame_start] = equations

    states_start, states_end = self.length_of_state_variables()   # Length size
    states_labels = self.states_labels(self.state_variables)

    self.writeConstants(language, vars)              # Write the constants file
    self.writeSelections(language, selection_matrices)   # Write the selections
    self.writeInitialStatesAndFrames(language, state_variables, frames)

    # Writing out the simulation file
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    s = j2_env.get_template('new_template_main.py.j2'.format(language)).render(
        states_and_lengths = tuple(zip(states, states_start, states_end)),
        states_and_labels = tuple(zip(states, states_labels)),
        date = '{}'.format(strftime("%Y-%m-%d %H:%M:%S")),
        constant_equations = constant_equations,
        state_variables = state_variables,
        author = os.uname().nodename,
        model_name = self.model_name,
        frame_start = frame_start,
        derivatives = derivatives,
        mod_name = self.mod_name,
        frame_end = frame_end,
        networks = networks,
        indices = indices,
        equations = eqs,
        frames = frames,
        states = states,
        vars = vars,
      )
    sim_name = 'simulation_{}{}'.format(self.mod_name, self.file_ending)
    f_name = os.path.join(self.model_loc, language, sim_name)
    f = open(f_name, 'msg_box+')
    f.write(s)
    f.close()
    print('Wrote out {} simulation to file:\n{}'.format(language, f_name))
    self.copySimulationOperators()

  def file_loc(self):
    sim_name = 'simulation_{}{}'.format(self.mod_name, self.file_ending)
    f_name = os.path.join(self.model_loc, self.language, sim_name)
    return 'Wrote out {} simulation to file:\n{}'.format(self.language, f_name)

  def __getCompiledIndexList(self, variable_symbol, language):
    v = self.variables[variable_symbol]
    ind = []
    if v.index_structures:        # index structure is part of the variable def
      for i in v.index_structures:
        index = self.indices.compileIndex(i, language)
        if self.object_oriented:
          index = CODE[language]["obj"].format(index)
        ind.append(index)                             # alias_i_dict[language])
    else:
      ind = ''
    return ind

  def length_of_state_variables(self):
    states_start = []
    states_end = []
    start = 0
    end = 0
    for state_length in [self.size_of_variable(self.variables[var])[0]
                         for var in self.state_variables]:
      end += state_length
      states_start.append(start)
      states_end.append(end)
      start += state_length
    return states_start, states_end

  def copySimulationOperators(self):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    file_name = UTILITIES_FILE_NAME + self.file_ending
    source = os.path.join(THIS_DIR, file_name)
    destination = os.path.join(self.model_loc, self.language, file_name)
    copyfile(source, destination)

  def only_dynamic_nodes(self, variable):
    """
    Calculate size of variable
    """
    nodes = self.model.nodes
    size = self.size_of_variable(variable)
    mat = np.zeros((size))
    index_structures = variable.index_structures[0]
    if not index_structures:
      return 1
    curIndex = 0
    for m, b in zip(self.mod_index[index_structures].mapping,
                    self.mod_index[index_structures].blocking):
      if b == 0:                                          # HACK: Hacke-ti-hack
        b = 1
      if nodes[str(m)].type == 'dynamic':
        mat[curIndex:curIndex+b] = 1.
      curIndex += b
    return mat[:, np.newaxis]

  def states_labels(self, states):
    reps = []
    for state in states:
      var = self.variables[state]
      reps.append(self.mod_index[var.index_structures[0]].printable())
    return reps

  def variable_dict(self, var):
    dict = {}
    dict["label"] = var.label
    dict["variable_type"] = var.type
    dict["layer"] = var.network
    dict["doc"] = var.doc
    dict["index_structures"] = var.index_structures
    dict["units"] = var.units.asList()
    return dict

  def flow_system(self):
    pass
