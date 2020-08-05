from os import path, mkdir

from PyQt4 import QtCore, QtGui

from Common.common_resources import getOntologyName
from Common.common_resources import askForModelFileGivenOntologyLocation as afm
from Common.ontology_container import OntologyContainer
from TaskBuilder.z_ModelFactory.editor_model_factory_gui import Ui_MainWindow

# from jinja2 import Environment             # sudo apt-get install python-jinja2
# from jinja2 import FileSystemLoader

# from Common.common_resources import invertDict
# from Common.ontology_container import OntologyContainer
# from Common.resource_initialisation import DIRECTORIES

# from OntologyEquationEditor.resources import FILE_EXTENSIONS
from OntologyBuilder.OntologyEquationEditor.resources import LANGUAGES
# from OntologyEquationEditor.resources import CODE


# from ModelFactory.model_framework import Model
from TaskBuilder.z_ModelFactory.model_integration import ModelFactory

FOLDERS = ['variables', 'model_json', 'equations']


class Ui_ModelFactory(QtGui.QMainWindow):
  def __init__(self):
    QtGui.QMainWindow.__init__(self)
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    # attach ontology
    # self.ontology_name = 'SOFC_04'
    self.ontology_name = getOntologyName()
    self.ontology = OntologyContainer(self.ontology_name)
    self.ontology_location = self.ontology.onto_path
    # data_file_resources = checkAndFixResources(self.ontology_name, stage="ontology-stage-2")
    # self.self.ontology_location = data_file_resources["self.ontology_location"]
    # self.mod_name = 'SOFC-GT-SIMPLIFIED'
    self.mod_name = afm(self.ontology_location + '/models')[0]
    self.model_loc = '{}/models/{}'.format(self.ontology_location, self.mod_name)

    self.ui.ontology_name_label.setText('{}'.format(self.ontology_name))
    # self.ontology = OntologyContainer(data_file_resources["ontology_location"])

    self.ui.model_name_label.setText('{}'.format(self.mod_name))
    self.model_file_name = '{}.json'.format(self.mod_name)

    self.fill_language_selection()
    message = '<b>Set up</b> <br />Ontology: {}<br />Model: {}'
    display = message.format(self.ontology_name, self.mod_name)
    self.ui.message_box.setText(display)
    self.language = None
    self.already_compiled = self.check_for_model_existance()
    if not self.already_compiled:
      self.setup_new_model_structure()
    self.upload_topology()

  def fill_language_selection(self):
    languages = ['-'] + LANGUAGES['code_generation']
    self.ui.output_language_box.clear()
    self.ui.output_language_box.addItems(languages)

  @QtCore.pyqtSignature('QString')
  def on_output_language_box_activated(self, language):
    self.ui.message_box.setText('Change output language to {}'.format(language))
    self.language = language

  def on_produce_model_button_pressed(self):
    if self.language:
      self.factory = ModelFactory(self.ontology,
                                  self.mod_name,
                                  self.language,
                                  self.model_loc)
      self.factory.produce_code()
      self.ui.message_box.setText(self.factory.file_loc())
    else:
      self.ui.message_box.setText('Have not selected an output language!')
      print("Language not selected")
      return

  def upload_topology(self):
    pass
    # loc = path.join(self.model_loc, 'figures', 'topo.png')
    # pixmap = QtGui.QPixmap(loc)
    # pixmap4 = pixmap.scaled(490, 350, QtCore.Qt.KeepAspectRatio)
    # scene = QtGui.QGraphicsScene()
    # scene.addPixmap(pixmap4)
    # self.ui.display_topology.setScene(scene)
    # self.ui.display_topology.show()

  def check_for_file_existance(self, file):
    return path.isfile(file)

  def check_for_model_existance(self):
    """
    Check if model already compiled or this is the first version
    """
    loc = '{}/models/{}'.format(self.ontology_location, self.mod_name)
    exist = path.isdir(loc)
    return exist

  def setup_new_model_structure(self):
    self.ui.message_box.setText('Setting up new model structure')
    loc = '{}/models/{}'.format(self.ontology_location, self.mod_name)
    try:
      mkdir(loc)
    except FileExistsError:
      print("Directory ", loc,  " already exists")
    for folder in FOLDERS:
      place = '{}/{}'.format(loc, folder)
      try:
        mkdir(place)
      except FileExistsError:
        print("Directory ", place,  " already exists")

  def closeEvent(self, event):
    # Had problems with this one.
    self.deleteLater()
    # self.close()l
    return
