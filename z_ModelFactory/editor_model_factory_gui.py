# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor_model_factory_gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 537)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.message_box = QtWidgets.QTextBrowser(self.centralwidget)
        self.message_box.setGeometry(QtCore.QRect(50, 410, 491, 61))
        self.message_box.setObjectName("message_box")
        self.output_language_box = QtWidgets.QComboBox(self.centralwidget)
        self.output_language_box.setGeometry(QtCore.QRect(740, 60, 211, 25))
        self.output_language_box.setObjectName("output_language_box")
        self.ontology_name_label = QtWidgets.QLabel(self.centralwidget)
        self.ontology_name_label.setGeometry(QtCore.QRect(0, 0, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.ontology_name_label.setFont(font)
        self.ontology_name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ontology_name_label.setObjectName("ontology_name_label")
        self.model_name_label = QtWidgets.QLabel(self.centralwidget)
        self.model_name_label.setGeometry(QtCore.QRect(290, 0, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.model_name_label.setFont(font)
        self.model_name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.model_name_label.setObjectName("model_name_label")
        self.produce_model_button = QtWidgets.QPushButton(self.centralwidget)
        self.produce_model_button.setGeometry(QtCore.QRect(792, 440, 141, 25))
        self.produce_model_button.setObjectName("produce_model_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(750, 40, 81, 17))
        self.label.setObjectName("label")
        self.display_topology = QtWidgets.QGraphicsView(self.centralwidget)
        self.display_topology.setGeometry(QtCore.QRect(50, 40, 491, 361))
        self.display_topology.setObjectName("display_topology")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1015, 22))
        self.menubar.setObjectName("menubar")
        self.menuModel_Factory = QtWidgets.QMenu(self.menubar)
        self.menuModel_Factory.setObjectName("menuModel_Factory")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuModel_Factory.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ontology_name_label.setText(_translate("MainWindow", "TextLabel"))
        self.model_name_label.setText(_translate("MainWindow", "TextLabel"))
        self.produce_model_button.setText(_translate("MainWindow", "Produce model"))
        self.label.setText(_translate("MainWindow", "Language"))
        self.menuModel_Factory.setTitle(_translate("MainWindow", "Model Factory"))
