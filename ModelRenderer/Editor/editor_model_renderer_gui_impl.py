# from collections import OrderedDict

from PyQt4 import QtCore, QtGui

from Common.common_resources import getOntologyName
from Common.common_resources import askForModelFileGivenOntologyLocation as afm
from Common.common_resources import askForCasefileGivenLocation as afc
from Common.resource_initialisation import DIRECTORIES
from Common.ontology_container import OntologyContainer
from TaskBuilder.ModelRenderer.Editor.editor_model_factory_gui import Ui_MainWindow
from OntologyBuilder.OntologyEquationEditor.resources import LANGUAGES

from TaskBuilder.ModelRenderer.main import ModelRenderer
# FOLDERS = ['variables', 'model_json', 'equations']


class Ui_ModelFactory(QtGui.QMainWindow):
  def __init__(self):
    QtGui.QMainWindow.__init__(self)
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    self.ontology_name = getOntologyName()
    self.ontology = OntologyContainer(self.ontology_name)
    self.ontology_location = self.ontology.onto_path

    models_file = DIRECTORIES["model_library_location"] % self.ontology_name
    self.mod_name = afm(models_file)[0]
    # self.model_loc = '{}/models/{}'.format(self.ontology_location, self.mod_name)
    # print('JALLA')

    self.ui.ontology_name_label.setText('{}'.format(self.ontology_name))
    self.ui.model_name_label.setText('{}'.format(self.mod_name))

    self.model_loc = DIRECTORIES["model_location"] % (self.ontology_name, self.mod_name)

    self.cases_location = DIRECTORIES["cases_location"] % (self.ontology_name,
                                                           self.mod_name)
    self.case_name, new_case = afc(self.cases_location, alternative = False)

    # self.model = path.join(models_file, self.mod_name)

    # self.model_file_name = '{}.json'.format(self.mod_name)

    self.fill_language_selection()
    # message = '<b>Set up</b> <br />Ontology: {}<br />Model: {}'
    # display = message.format(self.ontology_name, self.mod_name)
    # self.ui.message_box.setText(display)
    # self.language = None
    # self.already_compiled = self.check_for_model_existance()
    # if not self.already_compiled:
      # self.setup_new_model_structure()
    # self.upload_topology()

    self.mr = ModelRenderer(self.ontology, self.mod_name, self.case_name, self.ui)

  def fill_language_selection(self):
    languages = ['-'] + LANGUAGES['code_generation']

    self.ui.output_language_box.clear()
    self.ui.output_language_box.addItems(languages)

  @QtCore.pyqtSignature('QString')
  def on_output_language_box_activated(self, language):
    self.ui.message_box.setText('Change output language to {}'.format(language))
    self.language = language
    self.mr.setup_system(language)

  def on_produce_model_button_pressed(self):
    self.mr.generate_output()

  def closeEvent(self, event):
    self.deleteLater()
    return
