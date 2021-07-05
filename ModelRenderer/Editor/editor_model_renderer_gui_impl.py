# from collections import OrderedDict
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from Common.common_resources import getOntologyName, getData
from Common.common_resources import askForModelFileGivenOntologyLocation
from Common.common_resources import askForCasefileGivenLocation
from Common.resource_initialisation import DIRECTORIES, FILES
from Common.ontology_container import OntologyContainer
from TaskBuilder.ModelRenderer.Editor.editor_model_factory_gui import Ui_MainWindow
from OntologyBuilder.OntologyEquationEditor.resources import LANGUAGES
from ModelBuilder.ModelComposer.modeller_model_data import ModelContainer

from TaskBuilder.ModelRenderer.main import ModelRenderer
# FOLDERS = ['variables', 'model_json', 'equations']


class Ui_ModelFactory(QtWidgets.QMainWindow):
  def __init__(self):
    QtWidgets.QMainWindow.__init__(self)
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    self.ontology_name = getOntologyName()
    self.ontology = OntologyContainer(self.ontology_name)
    self.ontology_location = self.ontology.onto_path

    models_file = DIRECTORIES["model_library_location"] % self.ontology_name
    self.model_name, status = askForModelFileGivenOntologyLocation(models_file,
                                      left_icon="reject",
                                      left_tooltip="reject",
                                      alternative=False)
    if status == "exit":
      print("exit -- there was no model given")
      sys.exit()

    self.ui.ontology_name_label.setText('{}'.format(self.ontology_name))
    self.ui.model_name_label.setText('{}'.format(self.model_name))

    self.model_loc = DIRECTORIES["model_location"] % (self.ontology_name, self.model_name)
    self.model_file = FILES["model_file"] % (self.ontology_name, self.model_name)
    self.cases_location = DIRECTORIES["cases_location"] % (self.ontology_name,
                                                           self.model_name)

    self.case_name, status = askForCasefileGivenLocation(self.cases_location,
                                                           left_icon="reject",
                                                           left_tooltip="reject",
                                                           right_icon="accept",
                                                           right_tooltip="accept") #, alternative = False)
    if status == "exit":
      sys.exit()
    if status == "new":
      pass
    if status == "existent":
      pass


    # # couple up model container
    # self.networks = self.ontology.list_leave_networks
    # self.model_container = ModelContainer(self.networks, self.ontology)
    # self.model_container.makeFromFile(self.model_file)

    # get flat topology
    model_flat_file = FILES["model_flat_file"] % (self.ontology_name, self.model_name)
    self.model_case_file = FILES["model_case_file"] %(self.ontology_name, self.model_name, self.case_name)
    # nodes
    # arcs
    # named_networks
    # typed_toke_domains
    # typed_token_incidence_matrix
    # typed_token_lists
    self.model_flat = getData(model_flat_file)
    del self.model_flat["named_networks"]["network__named_network"]   # Note: not needed here -- maybe somewhere else?

    # make incidence lists
    incidence_lists,incidence_lists_transfer_mechanism = self.__makeIncidenceLists()


    print("debugging ")



    # self.case_name, new_case = afc(self.cases_location, alternative = False)

    # self.model = path.join(models_file, self.mod_name)

    # self.model_file_name = '{}.json'.format(self.mod_name)

    # self.fill_language_selection()
    # message = '<b>Set up</b> <br />Ontology: {}<br />Model: {}'
    # display = message.format(self.ontology_name, self.mod_name)
    # self.ui.message_box.setText(display)
    # self.language = None
    # self.already_compiled = self.check_for_model_existance()
    # if not self.already_compiled:
      # self.setup_new_model_structure()
    # self.upload_topology()

    # self.mr = ModelRenderer(self.ontology, self.mod_name, self.case_name, self.ui)

  def __makeIncidenceLists(self):
    nodes = self.model_flat["nodes"]
    arcs = self.model_flat["arcs"]
    tokens = self.ontology.tokens
    incidence_lists_tokens = {}
    incidence_lists_transfer_mechanism = {}
    indices_arc_token_typed_token = {}
    for token in tokens:
      incidence_lists_tokens[token] = []
      incidence_lists_transfer_mechanism[token] = {}
      indices_arc_token_typed_token[token] = {}
      for a in arcs:
        if arcs[a]["token"] == token:
          source = arcs[a]["source"]
          sink = arcs[a]["sink"]
          incidence_lists_tokens[token].append((source, sink))
          mechanism = arcs[a]["mechanism"]
          if arcs[a]["mechanism"]:
            if mechanism not in incidence_lists_transfer_mechanism[token]:
              incidence_lists_transfer_mechanism[token][mechanism] = []
            incidence_lists_transfer_mechanism[token][mechanism].append((source, sink))

    return incidence_lists_tokens, incidence_lists_transfer_mechanism





  def generateColouredIncidenceMatrices(self):
    pass

  def fill_language_selection(self):
    languages = ['-'] + LANGUAGES['code_generation']

    self.ui.output_language_box.clear()
    self.ui.output_language_box.addItems(languages)

  # @QtCore.pyqtSignature('QString')
  def on_output_language_box_activated(self, language):
    self.ui.message_box.setText('Change output language to {}'.format(language))
    self.language = language
    self.mr.setup_system(language)

  def on_produce_model_button_pressed(self):
    self.mr.generate_output()

  def closeEvent(self, event):
    self.deleteLater()
    return
