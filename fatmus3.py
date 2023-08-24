# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fatmus.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
# import PyQt5_stylesheets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.vtkCommonColor import vtkNamedColors
from vtk.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty
)



def set_table_data(table_widget, row, column, data):
    # 创建表格�?
    item = QtWidgets.QTableWidgetItem(data)
    # 设置表格项的对齐方式
    item.setTextAlignment(0x0002)  # 居中对齐
    # 将表格项添加到表格部件的指定位置
    table_widget.setItem(row, column, item)


class Ui_MainWindow(object):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.actionsegmente_fat_and_muscle = None
        self.aspect_ratio = 1051 / 645

    def resizeEvent(self, event):
        print("resizeEvent triggered")
        current_size = self.size()
        new_width = current_size.width()
        new_height = int(new_width / self.aspect_ratio)
        self.resize(new_width, new_height)
        view_size = self.view1.size()
        view_w = view_size.width()
        view_h = view_size.height()
        self.vtkWidget.resize(view_w, view_h)

    def __initWidget(self):
        self.view1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 以鼠标所在位置为锚点进行缩放
        self.view1.setTransformationAnchor(self.view1.AnchorUnderMouse)
        self.view2.setTransformationAnchor(self.view2.AnchorUnderMouse)
        self.view3.setTransformationAnchor(self.view3.AnchorUnderMouse)

    def menu_ui(self, MainWindow):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        self.menubar.setFont(font)
        # self.menubar.setStyleSheet('background-color: powderblue')
        # self.menubar.setStyleSheet('background-color: black; color: #FFA500; font-weight: bold;')

        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSegmentation = QtWidgets.QMenu(self.menubar)
        self.menuSegmentation.setObjectName("menuSegmentation")
        self.menu_display_mask = QtWidgets.QMenu(self.menubar)
        self.menu_display_mask.setObjectName("menu_display_mask")
        self.menuclinic_data = QtWidgets.QMenu(self.menubar)
        self.menuclinic_data.setObjectName("menuclinic_data")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # self.statusbar.setStyleSheet('background-color: black; color: #FFA500; font-weight: bold;')
        self.statusbar.setFont(font)

        self.actionopen_file = QtWidgets.QAction(MainWindow)
        self.actionopen_file.setObjectName("actionopen_file")
        self.actionsegmente_spine = QtWidgets.QAction(MainWindow)
        self.actionsegmente_spine.setObjectName("actionsegmente_spine")
        self.actionsegmente_fat_and_muscle = QtWidgets.QAction(MainWindow)
        self.actionsegmente_fat_and_muscle.setObjectName("actionsegmente_fat_and_muscle")
        self.action_display_FM = QtWidgets.QAction(MainWindow)
        self.action_display_FM.setObjectName("action_display_FM")
        self.action_display_S = QtWidgets.QAction(MainWindow)
        self.action_display_S.setObjectName("action_display_S")
        self.action_predict = QtWidgets.QAction(MainWindow)
        self.action_predict.setObjectName("action_predict")

        self.menuFile.addAction(self.actionopen_file)
        self.menuSegmentation.addAction(self.actionsegmente_spine)
        self.menuSegmentation.addAction(self.actionsegmente_fat_and_muscle)
        self.menu_display_mask.addAction(self.action_display_S)
        self.menu_display_mask.addAction(self.action_display_FM)
        self.menuclinic_data.addAction(self.action_predict)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSegmentation.menuAction())
        self.menubar.addAction(self.menu_display_mask.menuAction())
        self.menubar.addAction(self.menuclinic_data.menuAction())

        self.actionopen_file.triggered.connect(MainWindow.showpic)
        self.actionsegmente_spine.triggered.connect(MainWindow.spine_seg)
        self.actionsegmente_fat_and_muscle.triggered.connect(MainWindow.fatmus_seg)
        self.action_display_FM.triggered.connect(MainWindow.show_fatmus_mask)
        self.action_display_S.triggered.connect(MainWindow.show_spine_mask)
        self.action_predict.triggered.connect(MainWindow.predictf)

    def setupUi(self, MainWindow):
        print('begin-------------------------------------')
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resizeEvent()
        MainWindow.setGeometry(100, 100, 1082, 663)
        MainWindow.setWindowTitle("Predict")
        MainWindow.setAutoFillBackground(True)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        # self.centralwidget.setGeometry(100, 100, 1200, 840)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menu_ui(MainWindow)
        allLayout = QtWidgets.QHBoxLayout()
        # views ------------------------------------------------------------------------------------
        viewwidget = QtWidgets.QWidget()
        viewLayout = QtWidgets.QGridLayout()

        self.scene1 = QtWidgets.QGraphicsScene()
        self.view1 = QtWidgets.QGraphicsView()
        self.scene2 = QtWidgets.QGraphicsScene()
        self.view2 = QtWidgets.QGraphicsView()
        self.scene3 = QtWidgets.QGraphicsScene()
        self.view3 = QtWidgets.QGraphicsView()
        self._color_background = QtGui.QColor('#000000')
        self.scene1.setBackgroundBrush(self._color_background)
        self.scene2.setBackgroundBrush(self._color_background)
        self.scene3.setBackgroundBrush(self._color_background)
        self.view1.setScene(self.scene1)
        self.view2.setScene(self.scene2)
        self.view3.setScene(self.scene3)
        self.view1.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.view2.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.view3.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.__initWidget()

        # vtkRenderer ------------------------------------------------------------------------------------
        self.pic_box_vface = QtWidgets.QGroupBox(self.centralwidget)
        self.pic_box_vface.setStyleSheet(
            'border-width: 0px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgba(0, 0, 0, 0) ;')

        self.ren = vtkRenderer()
        self.vtkWidget = QVTKRenderWindowInteractor(self.pic_box_vface)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.colors = vtkNamedColors()
        self.vtkpros = {}
        self.colors.SetColor('BkgColor', [0, 0, 0, 255])
        self.ren.SetBackground(self.colors.GetColor3d('BkgColor'))
        self.iren.Initialize()

        viewLayout.addWidget(self.view1, 0, 0)
        viewLayout.addWidget(self.view2, 0, 1)
        viewLayout.addWidget(self.view3, 1, 0)
        viewLayout.addWidget(self.pic_box_vface, 1, 1)
        viewLayout.addWidget(self.statusbar, 2, 0, 1, 2)

        self.statusbar.showMessage('Ready')
        viewwidget.setLayout(viewLayout)

        # INPUT Box of info_layout--------------------------------------------------------------------------------------
        input_wdg = QtWidgets.QFrame()
        input_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        input_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        input_layout = QtWidgets.QGridLayout()
        input_label = QtWidgets.QLabel('Input')
        # input_label.setStyleSheet('background-color: white;')
        clinic_data_label = QtWidgets.QLabel('Clinic data:')
        Age_label = QtWidgets.QLabel('1. Age           ')
        Sex_label = QtWidgets.QLabel('2. Sex           ')
        NLR_label = QtWidgets.QLabel('3. NLR           ')
        ALT_label = QtWidgets.QLabel('4. ALT           ')
        AFP_label = QtWidgets.QLabel('  5. AFP           ')
        HBV_label = QtWidgets.QLabel('  6. HBV           ')
        BCLC_label = QtWidgets.QLabel('  7. BCLC          ')
        CP_label = QtWidgets.QLabel('  8. Child-Pugh    ')
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        input_label.setFont(font)
        # input_label.setStyleSheet('background-color: white;')
        # input_label.setAlignment(Qt.AlignCenter)

        self.Age_line = QtWidgets.QLineEdit()
        self.Sex_line = QtWidgets.QLineEdit()
        self.NLR_line = QtWidgets.QLineEdit()
        self.ALT_line = QtWidgets.QLineEdit()
        self.AFP_line = QtWidgets.QLineEdit()
        self.HBV_line = QtWidgets.QLineEdit()
        self.BCLC_line = QtWidgets.QLineEdit()
        self.CP_line = QtWidgets.QLineEdit()

        line_edit_style = "QLineEdit { border-radius: 3px; background-color: rgb(220, 220, 220)}"
        self.Age_line.setStyleSheet(line_edit_style)
        self.Sex_line.setStyleSheet(line_edit_style)
        self.NLR_line.setStyleSheet(line_edit_style)
        self.ALT_line.setStyleSheet(line_edit_style)
        self.AFP_line.setStyleSheet(line_edit_style)
        self.HBV_line.setStyleSheet(line_edit_style)
        self.BCLC_line.setStyleSheet(line_edit_style)
        self.CP_line.setStyleSheet(line_edit_style)

        self.Age_line.setFixedHeight(37)
        self.Sex_line.setFixedHeight(37)
        self.NLR_line.setFixedHeight(37)
        self.ALT_line.setFixedHeight(37)
        self.AFP_line.setFixedHeight(37)
        self.HBV_line.setFixedHeight(37)
        self.BCLC_line.setFixedHeight(37)
        self.CP_line.setFixedHeight(37)
        font_color = "QLineEdit {color: rgb(20,20,20); background-color:rgb(220,220,220);}"
        self.Age_line.setStyleSheet(font_color)
        self.Sex_line.setStyleSheet(font_color)
        self.NLR_line.setStyleSheet(font_color)
        self.ALT_line.setStyleSheet(font_color)
        self.AFP_line.setStyleSheet(font_color)
        self.HBV_line.setStyleSheet(font_color)
        self.BCLC_line.setStyleSheet(font_color)
        self.CP_line.setStyleSheet(font_color)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        Age_label.setFont(font)
        Sex_label.setFont(font)
        NLR_label.setFont(font)
        ALT_label.setFont(font)
        AFP_label.setFont(font)
        HBV_label.setFont(font)
        BCLC_label.setFont(font)
        CP_label.setFont(font)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Age_line.setFont(font)
        self.Sex_line.setFont(font)
        self.NLR_line.setFont(font)
        self.ALT_line.setFont(font)
        self.AFP_line.setFont(font)
        self.HBV_line.setFont(font)
        self.BCLC_line.setFont(font)
        self.CP_line.setFont(font)
        input_layout.addWidget(input_label, 0, 0, 1, 4, Qt.AlignCenter)
        input_layout.addWidget(clinic_data_label, 1, 0)
        input_layout.addWidget(Age_label, 2, 0)
        input_layout.addWidget(Sex_label, 3, 0)
        input_layout.addWidget(NLR_label, 4, 0)
        input_layout.addWidget(ALT_label, 5, 0)

        input_layout.addWidget(self.Age_line, 2, 1)
        input_layout.addWidget(self.Sex_line, 3, 1)
        input_layout.addWidget(self.NLR_line, 4, 1)
        input_layout.addWidget(self.ALT_line, 5, 1)

        input_layout.addWidget(AFP_label, 2, 2)
        input_layout.addWidget(HBV_label, 3, 2)
        input_layout.addWidget(BCLC_label, 4, 2)
        input_layout.addWidget(CP_label, 5, 2)

        input_layout.addWidget(self.AFP_line, 2, 3)
        input_layout.addWidget(self.HBV_line, 3, 3)
        input_layout.addWidget(self.BCLC_line, 4, 3)
        input_layout.addWidget(self.CP_line, 5, 3)

        input_layout.setRowStretch(0,1)
        input_layout.setRowStretch(1,1)
        input_layout.setRowStretch(2,1)
        input_layout.setRowStretch(3,1)
        input_layout.setRowStretch(4,1)
        input_layout.setRowStretch(5,1)
        input_wdg.setLayout(input_layout)

        # OUTPUT Box of info_layout-------------------------------------------------------------------------------------
        output_wdg = QtWidgets.QFrame()
        output_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        output_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        output_layout = QtWidgets.QGridLayout()
        output_label = QtWidgets.QLabel('Output')
        data_3d_label = QtWidgets.QLabel('3D data:')
        AMV_label = QtWidgets.QLabel('1. AM-volume')
        BMV_label = QtWidgets.QLabel('2. BM-volume')
        SFV_label = QtWidgets.QLabel('3. SF-volume')
        VFV_label = QtWidgets.QLabel('4. VF-volume')
        AM_mean_label = QtWidgets.QLabel('5. AM-mean CT')
        AM_itq_label = QtWidgets.QLabel('6. AM-interquartile CT')
        AM_mg_label = QtWidgets.QLabel('7. AM-min gradient CT')
        VF_mg_label = QtWidgets.QLabel('8. VF-min gradient CT')
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        output_label.setFont(font)
        # output_label.setAlignment(Qt.AlignCenter)

        self.AMV_line = QtWidgets.QLineEdit()
        self.BMV_line = QtWidgets.QLineEdit()
        self.SFV_line = QtWidgets.QLineEdit()
        self.VFV_line = QtWidgets.QLineEdit()
        self.AM_mean_line = QtWidgets.QLineEdit()
        self.AM_itq_line = QtWidgets.QLineEdit()
        self.AM_mg_line = QtWidgets.QLineEdit()
        self.VF_mg_line = QtWidgets.QLineEdit()

        self.AMV_line.setFixedHeight(37)
        self.BMV_line.setFixedHeight(37)
        self.SFV_line.setFixedHeight(37)
        self.VFV_line.setFixedHeight(37)
        self.AM_mean_line.setFixedHeight(37)
        self.AM_itq_line.setFixedHeight(37)
        self.AM_mg_line.setFixedHeight(37)
        self.VF_mg_line.setFixedHeight(37)

        self.AMV_line.setStyleSheet(line_edit_style)
        self.BMV_line.setStyleSheet(line_edit_style)
        self.SFV_line.setStyleSheet(line_edit_style)
        self.VFV_line.setStyleSheet(line_edit_style)
        self.AM_mean_line.setStyleSheet(line_edit_style)
        self.AM_itq_line.setStyleSheet(line_edit_style)
        self.AM_mg_line.setStyleSheet(line_edit_style)
        self.VF_mg_line.setStyleSheet(line_edit_style)
        font_color = "QLineEdit {color: rgb(20,20,20); background-color:rgb(220,220,220);}"
        self.AMV_line.setStyleSheet(font_color)
        self.BMV_line.setStyleSheet(font_color)
        self.SFV_line.setStyleSheet(font_color)
        self.VFV_line.setStyleSheet(font_color)
        self.AM_mean_line.setStyleSheet(font_color)
        self.AM_itq_line.setStyleSheet(font_color)
        self.AM_mg_line.setStyleSheet(font_color)
        self.VF_mg_line.setStyleSheet(font_color)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        AMV_label.setFont(font)
        BMV_label.setFont(font)
        SFV_label.setFont(font)
        VFV_label.setFont(font)
        AM_mean_label.setFont(font)
        AM_itq_label.setFont(font)
        AM_mg_label.setFont(font)
        VF_mg_label.setFont(font)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.AMV_line.setFont(font)
        self.BMV_line.setFont(font)
        self.SFV_line.setFont(font)
        self.VFV_line.setFont(font)
        self.AM_mean_line.setFont(font)
        self.AM_itq_line.setFont(font)
        self.AM_mg_line.setFont(font)
        self.VF_mg_line.setFont(font)
        output_layout.addWidget(output_label, 0, 0, 1, 4, Qt.AlignCenter)
        output_layout.addWidget(data_3d_label, 1, 0)
        output_layout.addWidget(AMV_label, 2, 0)
        output_layout.addWidget(BMV_label, 3, 0)
        output_layout.addWidget(SFV_label, 4, 0)
        output_layout.addWidget(VFV_label, 5, 0)

        output_layout.addWidget(self.AMV_line, 2, 1)
        output_layout.addWidget(self.BMV_line, 3, 1)
        output_layout.addWidget(self.SFV_line, 4, 1)
        output_layout.addWidget(self.VFV_line, 5, 1)

        output_layout.addWidget(AM_mean_label, 2, 2)
        output_layout.addWidget(AM_itq_label, 3, 2)
        output_layout.addWidget(AM_mg_label, 4, 2)
        output_layout.addWidget(VF_mg_label, 5, 2)

        output_layout.addWidget(self.AM_mean_line, 2, 3)
        output_layout.addWidget(self.AM_itq_line, 3, 3)
        output_layout.addWidget(self.AM_mg_line, 4, 3)
        output_layout.addWidget(self.VF_mg_line, 5, 3)
        output_wdg.setLayout(output_layout)

        # Box of "Prediction 3-year survival prob:"---------------------------------------------------------------------
        predict_layout = QtWidgets.QHBoxLayout()
        predict_wdg = QtWidgets.QFrame()
        predict_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        predict_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        predict_label = QtWidgets.QLabel('Prediction Probility:')

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        predict_label.setFont(font)

        self.plotresult = QtWidgets.QTextBrowser()
        self.plotresult.setStyleSheet(
            'border-width: 1px;border-style: solid;background-color: rgb(220, 220, 220);')
        # Disable word wrap to show the entire text without wrapping
        self.plotresult.setLineWrapMode(QtWidgets.QTextBrowser.NoWrap)
        # Remove the vertical scrollbar
        self.plotresult.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plotresult.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plotresult.setFixedSize(140, 37)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.plotresult.setFont(font)

        predict_layout.addWidget(predict_label)
        predict_layout.addWidget(self.plotresult)
        predict_layout.setAlignment(self.plotresult, Qt.AlignLeft)  # 将部件向左对?
        predict_wdg.setLayout(predict_layout)

        # info_layout ---------------------------------------------------------------------------------------------------
        info_layout = QtWidgets.QVBoxLayout()
        info_wdg = QtWidgets.QFrame()
        info_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        info_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        # info_wdg.setStyleSheet('background-color: rgb(25,35,45);')
        info_layout.addWidget(input_wdg)
        info_layout.addWidget(output_wdg)
        info_layout.addWidget(predict_wdg)
        info_layout.setStretch(2, 1)
        info_layout.setStretch(0, 3)
        info_layout.setStretch(1, 3)
        info_wdg.setLayout(info_layout)

        # all layout ---------------------------------------------------------------------------------------------------
        allLayout.addWidget(viewwidget)
        allLayout.addWidget(info_wdg)
        allLayout.setStretch(0, 5)
        allLayout.setStretch(1, 4)

        self.centralwidget.setLayout(allLayout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSegmentation.setTitle(_translate("MainWindow", "Segmentation"))
        self.menu_display_mask.setTitle(_translate("MainWindow", "Display"))
        self.menuclinic_data.setTitle(_translate("MainWindow", "Predict"))
        self.actionopen_file.setText(_translate("MainWindow", "Open File"))
        self.actionsegmente_spine.setText(_translate("MainWindow", "Segment Spine"))
        self.actionsegmente_fat_and_muscle.setText(_translate("MainWindow", "Segment Fat and Muscle"))
        self.action_display_FM.setText(_translate("MainWindow", "Display Fat and Muscle"))
        self.action_display_S.setText(_translate("MainWindow", "Display Spine"))
        self.action_predict.setText(_translate("MainWindow", "Predict 3-year survival probability"))




class Window_FM(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window_FM, self).__init__()

        self.setupUi()


if __name__ == "__main__":
    import sys
    from New_App_fatmus3 import Window_FM2
    from qt_material import apply_stylesheet

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    # app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style='style_Classic'))
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    widget = Window_FM2()

    widget.show()
    sys.exit(app.exec_())
