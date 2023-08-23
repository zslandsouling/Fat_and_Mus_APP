import shutil
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QFileDialog, QApplication)
from PyQt5.QtGui import QPixmap
import joblib

from sklearn.calibration import CalibratedClassifierCV
import nibabel as nib
from fatmus3 import Ui_MainWindow

import vtk
import sys
import pandas as pd
from radiomics import featureextractor
import six
import SimpleITK as itk
from qimage2ndarray import array2qimage
import skimage.transform as st
import torch
from utils import *
import cv2
from cv2 import applyColorMap

# import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
import math
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score, roc_curve
from vtk.vtkCommonColor import vtkNamedColors
from vtk.vtkCommonDataModel import vtkPiecewiseFunction
from vtk.vtkIOImage import vtkMetaImageReader, vtkNIFTIImageReader
from vtk.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtk.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from PyQt5.QtGui import *
from PyQt5.QtCore import *

gpu_id = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Window_FM2(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window_FM2, self).__init__()
        self.setupUi(self)
        self.img = None
        self.showmask = None
        self.prinimg = None
        self.printlivermask = None
        self.mask = None
        self.space = None
        # self.net = nnNet_withclion(channel=1, numclass=1, numword=60, f=9, lian=2).to(device)
        # self.load_GPUS(self.net)
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.fatmus_mask = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.face_flage = 0

        self.model = joblib.load('platts_scaling_3d_clinic_3.pickle')
        self.model2 = joblib.load('model_.pickle').astype(np.float)
        # with open('standard.pickle', 'rb') as intp: pickle.load(intp)
        self.stander = joblib.load("standard.pickle")
        # torch.cuda.empty_cache()
        self.view1.setMouseTracking(True)
        self.view2.setMouseTracking(True)
        self.view3.setMouseTracking(True)

        self.view1.installEventFilter(self)
        self.view2.installEventFilter(self)
        self.view3.installEventFilter(self)

        self.leng_img = -100
        self.width_img = -100
        self.high_img = -100

        self.right_press_flag = False
        self.left_press_flag = False

        self.face_w = 272
        self.face_h = 272

        self.file_name = None
        self.file_path = None
        self.spine_mask_path = os.path.join('data', 'ori', 'spine')
        self.fatmus_mask_path = os.path.join('data', 'ori', 'fatmus')
        self.img_path = os.path.join('data', 'ori', 'img')
        self.crop_spine_mask_path = os.path.join('data', 'crop', 'spine')
        self.crop_fatmus_mask_path = os.path.join('data', 'crop', 'fatmus')
        self.crop_img_path = os.path.join('data', 'crop', 'img')
        os.makedirs(self.spine_mask_path, exist_ok=True)
        os.makedirs(self.fatmus_mask_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)
        os.makedirs(self.crop_spine_mask_path, exist_ok=True)
        os.makedirs(self.crop_fatmus_mask_path, exist_ok=True)
        os.makedirs(self.crop_img_path, exist_ok=True)
        self.spine_mask_file = ''
        self.fatmus_mask_file = ''
        self.img_file = ''
        self.crop_spine_mask_file = ''
        self.crop_fatmus_mask_file = ''
        self.crop_img_file = ''

        self.volumes = {}
        self.volume_path = ''
        self.volume_old = None

        self.afpg2 = None
        self.afpg1 = None
        self.am_mean = None
        self.am_25var = None
        self.vf_25var = None

        self.scale_ratio = 1

        self.scene1.mouseDoubleClickEvent = self.pointselect1
        self.scene2.mouseDoubleClickEvent = self.pointselect2
        self.scene3.mouseDoubleClickEvent = self.pointselect3
        self.pen = QtGui.QPen(QtCore.Qt.green)
        self.pen2 = QtGui.QPen(QtCore.Qt.red, 4)
        self.pen3 = QtGui.QPen(QtCore.Qt.red)
        self.x_line1 = QtWidgets.QGraphicsLineItem()
        self.x_line2 = QtWidgets.QGraphicsLineItem()
        self.x_line1.setPen(self.pen)
        self.x_line2.setPen(self.pen)
        self.y_line1 = QtWidgets.QGraphicsLineItem()
        self.y_line2 = QtWidgets.QGraphicsLineItem()
        self.y_line1.setPen(self.pen)
        self.y_line2.setPen(self.pen)
        self.z_line1 = QtWidgets.QGraphicsLineItem()
        self.z_line2 = QtWidgets.QGraphicsLineItem()
        self.z_line1.setPen(self.pen)
        self.z_line2.setPen(self.pen)

        self.x_point1 = QtWidgets.QGraphicsEllipseItem()
        self.x_point2 = QtWidgets.QGraphicsEllipseItem()
        self.x_point1.setPen(self.pen2)
        self.x_point2.setPen(self.pen2)
        self.y_point1 = QtWidgets.QGraphicsEllipseItem()
        self.y_point2 = QtWidgets.QGraphicsEllipseItem()
        self.y_point1.setPen(self.pen2)
        self.y_point2.setPen(self.pen2)
        self.z_point1 = QtWidgets.QGraphicsEllipseItem()
        self.z_point2 = QtWidgets.QGraphicsEllipseItem()
        self.z_point1.setPen(self.pen2)
        self.z_point2.setPen(self.pen2)
        self.x_point_flag = 1
        self.y_point_flag = 1
        self.z_point_flag = 1
        self.x_point2line = QtWidgets.QGraphicsLineItem()
        self.x_point2line.setPen(self.pen3)
        self.y_point2line = QtWidgets.QGraphicsLineItem()
        self.y_point2line.setPen(self.pen3)
        self.z_point2line = QtWidgets.QGraphicsLineItem()
        self.z_point2line.setPen(self.pen3)

        self.x_x = None
        self.x_y = None
        self.y_x = None
        self.y_y = None
        self.z_x = None
        self.z_y = None

        self.pixmapItem1 = None
        self.pixmapItem2 = None
        self.pixmapItem3 = None

    def pointselect1(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = event.scenePos().y()
            self.y_x = event.scenePos().x()
            self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.z_x = int(round(self.face_w * (event.scenePos().y() / self.face_h), 0))
            self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.width_img = int(round((event.scenePos().y() / self.face_h) * self.width_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.x_point_flag == 1:
                self.draw_point_x(self.x_point1, event.scenePos().x(), event.scenePos().y())
                self.point_x1_x = event.scenePos().x()
                self.point_x1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.x_point_flag = 2
            elif self.x_point_flag == 2:
                self.draw_point_x(self.x_point2, event.scenePos().x(), event.scenePos().y())
                self.point_x2_x = event.scenePos().x()
                self.point_x2_y = event.scenePos().y()
                self.drawline(self.scene1, self.x_point2line, self.point_x1_x, self.point_x1_y,
                              self.point_x2_x, self.point_x2_y)

                self.x_distance_x = abs(self.point_x1_x - self.point_x2_x) / self.face_w * self.high_max * self.space[0]
                self.x_distance_y = abs(self.point_x1_y - self.point_x2_y) / self.face_h * self.width_max * self.space[
                    1]
                self.x_distance = math.sqrt(self.x_distance_y ** 2 + self.x_distance_x ** 2)
                self.distance.setText(f"{self.x_distance:>.8f}")
                self.x_point_flag = 1

    def pointselect2(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
            self.y_x = event.scenePos().x()
            self.y_y = event.scenePos().y()
            self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.y_point_flag == 1:
                self.draw_point_y(self.y_point1, event.scenePos().x(), event.scenePos().y())
                self.point_y1_x = event.scenePos().x()
                self.point_y1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.y_point_flag = 2
            elif self.y_point_flag == 2:
                self.draw_point_y(self.y_point2, event.scenePos().x(), event.scenePos().y())
                self.point_y2_x = event.scenePos().x()
                self.point_y2_y = event.scenePos().y()
                self.drawline(self.scene2, self.y_point2line, self.point_y1_x, self.point_y1_y,
                              self.point_y2_x, self.point_y2_y)

                self.y_distance_x = abs(self.point_y1_x - self.point_y2_x) / self.face_w * self.high_max * self.space[0]
                self.y_distance_y = abs(self.point_y1_y - self.point_y2_y) / self.face_h * self.leng_max * self.space[2]
                self.y_distance = math.sqrt(self.y_distance_y ** 2 + self.y_distance_x ** 2)
                self.distance.setText(f"{self.y_distance:>.8f}")
                self.y_point_flag = 1

    def pointselect3(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.x_y = int(round(self.face_h * (event.scenePos().x() / self.face_w), 0))
            self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.y_y = event.scenePos().y()
            self.z_x = event.scenePos().x()
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.width_img = int(round((event.scenePos().x() / self.face_w) * self.width_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.z_point_flag == 1:
                self.draw_point_z(self.z_point1, event.scenePos().x(), event.scenePos().y())
                self.point_z1_x = event.scenePos().x()
                self.point_z1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point2)
                self.z_point_flag = 2
            elif self.z_point_flag == 2:
                self.draw_point_z(self.z_point2, event.scenePos().x(), event.scenePos().y())
                self.point_z2_x = event.scenePos().x()
                self.point_z2_y = event.scenePos().y()
                self.drawline(self.scene3, self.z_point2line, self.point_z1_x, self.point_z1_y,
                              self.point_z2_x, self.point_z2_y)
                self.z_distance_x = abs(self.point_z1_x - self.point_z2_x) / self.face_w * self.width_max * self.space[
                    1]
                self.z_distance_y = abs(self.point_z1_y - self.point_z2_y) / self.face_h * self.leng_max * self.space[2]
                self.z_distance = math.sqrt(self.z_distance_y ** 2 + self.z_distance_x ** 2)
                self.distance.setText(f"{self.z_distance:>.8f}")
                self.z_point_flag = 1

    def draw_point_x(self, item, x, y):
        self.scene1.removeItem(item)
        self.scene1.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene1.addItem(item)
        self.scene1.addItem(item)

    def drawline(self, scene, item, x1, y1, x2, y2):
        item.setLine(QtCore.QLineF(QtCore.QPointF(x1, y1),
                                   QtCore.QPointF(x2, y2)))
        scene.addItem(item)

    def draw_point_y(self, item, x, y):
        self.scene2.removeItem(item)
        self.scene2.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene2.addItem(item)
        self.scene2.addItem(item)

    def draw_point_z(self, item, x, y):
        self.scene3.removeItem(item)
        self.scene3.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene3.addItem(item)
        self.scene3.addItem(item)

    def draw_line(self, x_x, x_y, y_x, y_y, z_x, z_y):
        self.scene1.removeItem(self.x_line1)
        self.scene1.removeItem(self.x_line2)
        self.scene2.removeItem(self.y_line1)
        self.scene2.removeItem(self.y_line2)
        self.scene3.removeItem(self.z_line1)
        self.scene3.removeItem(self.z_line2)
        self.x_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x_x), 0),
                                           QtCore.QPointF(int(x_x), self.scene1.height())))
        self.x_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(x_y)),
                                           QtCore.QPointF(self.scene1.width(), int(x_y))))
        self.y_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(y_x), 0),
                                           QtCore.QPointF(int(y_x), self.scene2.height())))
        self.y_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y_y)),
                                           QtCore.QPointF(self.scene2.width(), int(y_y))))
        self.z_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(z_x), 0),
                                           QtCore.QPointF(int(z_x), self.scene3.height())))
        self.z_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(z_y)),
                                           QtCore.QPointF(self.scene3.width(), int(z_y))))
        self.scene1.addItem(self.x_line1)
        self.scene1.addItem(self.x_line2)
        self.scene2.addItem(self.y_line1)
        self.scene2.addItem(self.y_line2)
        self.scene3.addItem(self.z_line1)
        self.scene3.addItem(self.z_line2)

    def cleardistancef(self):
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)
        self.distance.clear()

    def clearallf(self):
        self.spine_mask = None
        self.fatmus_mask = None
        self.crop_spine_mask = None
        self.crop_fatmus_mask = None
        self.img = None
        self.crop_img = None
        self.printlivermask = None
        self.space = None
        self.numprint = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.plotresult.clear()
        self.Age_line.clear()
        self.Sex_line.clear()
        self.NLR_line.clear()
        self.ALT_line.clear()
        self.AFP_line.clear()
        self.HBV_line.clear()
        self.BCLC_line.clear()
        self.CP_line.clear()
        self.AMV_line.clear()
        self.BMV_line.clear()
        self.SFV_line.clear()
        self.VFV_line.clear()
        self.AM_mean_line.clear()
        self.AM_itq_line.clear()
        self.AM_mg_line.clear()
        self.VF_mg_line.clear()
        self.afpg2 = None
        self.afpg1 = None
        self.am_mean = None
        self.am_25var = None
        self.vf_25var = None

    def __setDragEnabled(self, isEnabled: bool):
        """ 设置拖拽是否启动 """
        self.view1.setDragMode(self.view1.ScrollHandDrag if isEnabled else self.view1.NoDrag)
        self.view2.setDragMode(self.view2.ScrollHandDrag if isEnabled else self.view2.NoDrag)
        self.view3.setDragMode(self.view3.ScrollHandDrag if isEnabled else self.view3.NoDrag)

    def __isEnableDrag(self, pixmap):
        """ 根据图片的尺寸决定是否启动拖拽功能 """
        if self.prinimg is not None:
            v = pixmap.width() > self.face_w
            h = pixmap.height() > self.face_h
            return v or h

    def showpic_xyz(self, x, y, z, w_size, h_size):
        if self.prinimg is not None:
            if self.prinimg.ndim == 3:
                image_axi = array2qimage(np.expand_dims(self.prinimg[x, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_axi = array2qimage(self.prinimg[x, ...])

            pixmap_axi = QPixmap.fromImage(image_axi).scaled(w_size, h_size)
            self.pixmapItem1 = QtWidgets.QGraphicsPixmapItem(pixmap_axi)
            self.scene1.addItem(self.pixmapItem1)
            self.view1.setSceneRect(QtCore.QRectF(pixmap_axi.rect()))
            self.view1.setScene(self.scene1)

            if self.prinimg.ndim == 3:
                image_cor = array2qimage(np.expand_dims(self.prinimg[:, y, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_cor = array2qimage(self.prinimg[:, y, ...])
            pixmap_cor = QPixmap.fromImage(image_cor).scaled(w_size, h_size)
            self.pixmapItem2 = QtWidgets.QGraphicsPixmapItem(pixmap_cor)
            self.scene2.addItem(self.pixmapItem2)
            self.view2.setSceneRect(QtCore.QRectF(pixmap_cor.rect()))
            self.view2.setScene(self.scene2)

            if self.prinimg.ndim == 3:
                image_sag = array2qimage(np.expand_dims(self.prinimg[:, :, z, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_sag = array2qimage(self.prinimg[:, :, z, ...])
            pixmap_sag = QPixmap.fromImage(image_sag).scaled(w_size, h_size)
            self.pixmapItem3 = QtWidgets.QGraphicsPixmapItem(pixmap_sag)
            self.scene3.addItem(self.pixmapItem3)
            self.view3.setSceneRect(QtCore.QRectF(pixmap_sag.rect()))
            self.view3.setScene(self.scene3)

            self.__setDragEnabled(self.__isEnableDrag(pixmap_axi))

    def showpic(self):
        # fname = QFileDialog.getOpenFileName(self, '加载图片', 'C:\\')

        fname = QFileDialog().getOpenFileName(self, caption='Load CT image',
                                              directory='data',
                                              filter="Image(*.nii *.nii.gz)")
        self.file_path = fname[0]
        self.parent_file = os.path.abspath(os.path.join(self.file_path, ".."))
        if self.prinimg is not None:
            self.clearallf()
        if len(fname[1]) != 0:
            self.statusbar.showMessage("Loading the image...")
            self.file_name = fname[0].split('/')[-1]
            self.img_file = os.path.join(self.img_path, self.file_name)
            self.fatmus_mask_file = os.path.join(self.fatmus_mask_path, self.file_name)
            self.spine_mask_file = os.path.join(self.spine_mask_path, self.file_name)
            if not os.path.exists(self.img_file):
                shutil.copy(self.file_path, self.img_file)
            print(self.file_name)
            img = itk.ReadImage(self.img_file)
            self.space = img.GetSpacing()
            # img = itk.GetArrayFromImage(img)
            self.img = itk.GetArrayFromImage(img)
            img = np.clip(self.img, -17.0, 201.0)
            img = np.flip(img, axis=0)
            self.prinimg = (img - 99.40078) / 39.392952
            self.ori_prinimg = self.prinimg
            self.leng_max, self.width_max, self.high_max = self.img.shape
            self.face_w = 272
            self.face_h = 272

            self.leng_img = int(self.leng_max / 2)
            self.width_img = int(self.width_max / 2)
            self.high_img = int(self.high_max / 2)
            self.showpic_xyz(int(self.leng_max / 2), int(self.width_max / 2), int(self.high_max / 2), self.face_w,
                             self.face_h)
            self.x_x = self.face_w // 2
            self.x_y = self.face_h // 2
            self.y_x = self.face_w // 2
            self.y_y = self.face_h // 2
            self.z_x = self.face_w // 2
            self.z_y = self.face_h // 2
            self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

            # self.coord.setText(f"Path: {self.file_path}")

            reader = vtkNIFTIImageReader()
            reader.SetFileName(self.img_file)
            reader.Update()

            volumeMapper = vtkGPUVolumeRayCastMapper()
            volumeMapper.SetInputData(reader.GetOutput())

            volumeProperty = vtkVolumeProperty()
            volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
            volumeProperty.ShadeOn()  # 打开或者关闭阴影
            volumeProperty.SetAmbient(0.4)
            volumeProperty.SetDiffuse(0.6)  # 漫反射
            volumeProperty.SetSpecular(0.2)  # 镜面反射
            # 设置不透明度
            compositeOpacity = vtkPiecewiseFunction()
            compositeOpacity.AddPoint(70, 0.00)
            compositeOpacity.AddPoint(90, 0.4)
            compositeOpacity.AddPoint(180, 0.6)
            volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度
            # 设置梯度不透明属性
            volumeGradientOpacity = vtkPiecewiseFunction()
            volumeGradientOpacity.AddPoint(10, 0.0)
            volumeGradientOpacity.AddPoint(90, 0.5)
            volumeGradientOpacity.AddPoint(100, 1.0)

            # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
            # 设置颜色属性
            color = vtkColorTransferFunction()
            color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
            color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
            color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
            volumeProperty.SetColor(color)

            volume = vtkVolume()  # 和vtkActor作用一致
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProperty)

            if self.volume_old is not None:
                self.ren.RemoveViewProp(self.volume_old)
            self.ren.AddViewProp(volume)
            self.volume_old = volume
            camera = self.ren.GetActiveCamera()
            c = volume.GetCenter()

            camera.SetViewUp(0, 0, 1)
            camera.SetViewAngle(60)
            camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
            camera.SetFocalPoint(c[0], c[1] - 200, c[2])
            camera.Azimuth(40.0)
            camera.Elevation(10.0)
            # self.render_window.Render()
            self.iren.Initialize()

            view_size = self.view1.size()
            view_w = view_size.width()
            view_h = view_size.height()
            self.vtkWidget.resize(view_w, view_h)
            # self.iren.Start()

    # def radiomics_link(self):
    #     self.statusbar.showMessage("Calculate the radiomics features")
    #     self.show_message_radiomics()
    #     if self.crop_fatmus_mask is None and self.crop_spine_mask is None:
    #         if os.path.exists(self.fatmus_mask_file) == False and self.fatmus_mask is None:
    #             self.get_fatmus_mask()
    #         elif os.path.exists(self.fatmus_mask_file) == True and self.fatmus_mask is None:
    #             fatmus_mask = itk.ReadImage(self.spine_mask_file)
    #             self.fatmus_mask = itk.GetArrayFromImage(fatmus_mask)
    #             self.fatmus_mask = np.where(self.fatmus_mask != 0, 1, 0)
    #     img = itk.ReadImage(self.filename)
    #     image_space = img.GetSpacing()
    #     image_direction = img.GetDirection()
    #     image_origin = img.GetOrigin()
    #     position_1 = np.where(self.fatmus_mask != 0)
    #     new_img = np.zeros_like(itk.GetArrayFromImage(img))
    #     new_img[position_1] = 1
    #
    #     src_new = itk.GetImageFromArray(new_img)
    #     src_new.SetSpacing(image_space)
    #     src_new.SetOrigin(image_origin)
    #     src_new.SetDirection(image_direction)
    #     itk.WriteImage(src_new, self.filename.split('.nii.gz')[0] + '_cat.nii.gz')
    #
    #     save_curdata, name = self.catch_features(self.filename, self.filename.split('.nii.gz')[0] + '_cat.nii.gz')
    #     pos = np.where((name == 'wavelet-LHH_glrlm_RunVariance') | (name == 'wavelet-HLH_glcm_MaximumProbability')
    #                    | (name == 'wavelet-LmeiHH_glrlm_LongRunEmphasis') | (
    #                            name == 'wavelet-LHL_glcm_DifferenceEntropy'))
    #     out = []
    #     for kk in pos[0]:
    #         out.append(save_curdata[kk])
    #     self.clion_radiomics = out
    #     self.statusbar.showMessage("The radiomics features have been extracted")
    #     # os.remove(self.filename.split('.nii.gz')[0] + '_cat.nii.gz')
    #
    # def catch_radiomics_features(self, imagePath, maskPath):
    #     if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
    #         raise Exception(
    #             'Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    #     settings = {}
    #     settings['binWidth'] = 25  # 5
    #     settings['sigma'] = [1, 3, 5]
    #     settings['Interpolator'] = itk.sitkBSpline
    #     settings['resampledPixelSpacing'] = [0.7, 0.7, 5]  # 3,3,3
    #     settings['voxelArrayShift'] = 1000  # 300
    #     settings['normalize'] = True
    #     settings['normalizeScale'] = 100
    #     extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    #     extractor.enableImageTypeByName('LoG')
    #     extractor.enableImageTypeByName('Wavelet')
    #     extractor.enableImageTypeByName('Gradient')
    #     extractor.enableAllFeatures()
    #     extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile',
    #                                                '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange',
    #                                                'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
    #                                                'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis',
    #                                                'Variance', 'Uniformity'])
    #     extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio',
    #                                           'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion',
    #                                           'Maximum3DDiameter', 'Maximum2DDiameterSlice', 'Maximum2DDiameterColumn',
    #                                           'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
    #                                           'LeastAxisLength', 'Elongation', 'Flatness'])
    #     feature_cur = []
    #     feature_name = []
    #     result = extractor.execute(imagePath, maskPath, label=1)
    #     for key, value in six.iteritems(result):
    #         # print('\t', key, ':', value)
    #         feature_name.append(key)
    #         feature_cur.append(value)
    #     # print(len(feature_cur[37:]))
    #     name = feature_name[37:]
    #     name = np.array(name)
    #     '''
    #     flag=1
    #     if flag:
    #         name = np.array(feature_name)
    #         name_df = pd.DataFrame(name)
    #         writer = pd.ExcelWriter('key.xlsx')
    #         name_df.to_excel(writer)
    #         writer.save()
    #         flag = 0
    #     '''
    #     for i in range(len(feature_cur[37:])):
    #         # if type(feature_cur[i+22]) != type(feature_cur[30]):
    #         feature_cur[i + 37] = float(feature_cur[i + 37])
    #     return feature_cur[37:], name

    def show_fatmus_mask(self):

        if self.img is None:
            self.statusbar.showMessage("Please load a image, first.")
        else:
            fname = QFileDialog.getOpenFileName(self, caption='Load fat and muscle mask', directory='data',
                                                filter="Image(*.nii *.nii.gz)")
            if len(fname[1]) != 0:
                self.showmask = read_nii(fname[0])
                if self.showmask.shape == self.img.shape:
                    self.mask_np = read_nii(fname[0])
                    self.img_np = read_nii(self.img_file)
                    self.new_img = self.mask_np * self.img_np
                    ori2new_nii(fname[0], self.new_img, os.path.join('./data', 'new_fatmus.nii.gz'))
                    if self.prinimg is not None:
                        self.printmask_fatmus(0.5)
                        # self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                        self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                        reader = vtkNIFTIImageReader()
                        # reader.SetFileName(os.path.join('./data', 'new_fatmus.nii.gz'))
                        reader.SetFileName(fname[0])
                        reader.Update()
                        volumeMapper = vtkGPUVolumeRayCastMapper()
                        volumeMapper.SetInputData(reader.GetOutput())

                        volumeProperty = vtkVolumeProperty()
                        volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                        volumeProperty.ShadeOn()  # 打开或者关闭阴影
                        volumeProperty.SetAmbient(0.4)
                        volumeProperty.SetDiffuse(0.6)  # 漫反射
                        volumeProperty.SetSpecular(0.2)  # 镜面反射
                        # 设置不透明度
                        compositeOpacity = vtkPiecewiseFunction()
                        compositeOpacity.AddPoint(0, 0)
                        compositeOpacity.AddPoint(1, 0.2)
                        compositeOpacity.AddPoint(2, 0.2)
                        compositeOpacity.AddPoint(3, 0.2)
                        compositeOpacity.AddPoint(4, 0.2)
                        volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                        # 设置梯度不透明属性
                        volumeGradientOpacity = vtkPiecewiseFunction()
                        volumeGradientOpacity.AddPoint(0, 0.0)
                        volumeGradientOpacity.AddPoint(1, 0.5)
                        volumeGradientOpacity.AddPoint(5, 1.0)

                        # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                        # 设置颜色属性
                        color = vtkColorTransferFunction()
                        color.AddRGBPoint(0, 0, 0, 0)
                        color.AddRGBPoint(1, 0.8, 0.8, 0.8)
                        color.AddRGBPoint(2, 0.4, 0.4, 0.4)
                        color.AddRGBPoint(3, 0.8, 0.52, 0.3)
                        color.AddRGBPoint(4, 0.8, 0.8, 0.3)
                        volumeProperty.SetColor(color)

                        volume = vtkVolume()  # 和vtkActor作用一致
                        volume.SetMapper(volumeMapper)
                        volume.SetProperty(volumeProperty)
                        if self.volume_old is not None:
                            self.ren.RemoveViewProp(self.volume_old)
                        self.ren.AddViewProp(volume)
                        self.volume_old = volume
                        # self.volume_path = fname[0]
                        camera = self.ren.GetActiveCamera()
                        c = volume.GetCenter()
                        camera.SetViewUp(0, 0, 1)
                        camera.SetViewAngle(60)
                        camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                        camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                        camera.Azimuth(30.0)
                        camera.Elevation(30.0)
                        self.iren.Initialize()
                        os.remove(os.path.join('./data', 'new_fatmus.nii.gz'))
                        self.statusbar.showMessage("The mask of fat and muscle has been displayed.")
                else:
                    # self.coord.setText('请加载与图对应的分割结果')
                    self.statusbar.showMessage("Please load a correspronding mask of fat and muscle.")

    def show_spine_mask(self):

        if self.img is None:
            self.statusbar.showMessage("Please load a image, first.")
        else:
            fname = QFileDialog.getOpenFileName(self, caption='Load spine mask', directory='data',
                                                filter="Image(*.nii *.nii.gz)")
            if len(fname[1]) != 0:
                spine_mask = itk.ReadImage(fname[0])
                self.showmask = itk.GetArrayFromImage(spine_mask)
                if self.showmask.shape == self.img.shape:
                    self.mask_np = read_nii(fname[0])
                    self.img_np = read_nii(self.img_file)
                    self.new_img = self.mask_np * self.img_np
                    ori2new_nii(fname[0], self.new_img, os.path.join('./data', 'new_spine.nii.gz'))

                    if self.prinimg is not None:
                        self.printmask_spine(0.5)
                        # self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                        self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                        reader = vtkNIFTIImageReader()
                        reader.SetFileName(os.path.join('./data', 'new_spine.nii.gz'))
                        reader.Update()
                        volumeMapper = vtkGPUVolumeRayCastMapper()
                        volumeMapper.SetInputData(reader.GetOutput())

                        volumeProperty = vtkVolumeProperty()
                        volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                        volumeProperty.ShadeOn()  # 打开或者关闭阴影
                        volumeProperty.SetAmbient(0.4)
                        volumeProperty.SetDiffuse(0.6)  # 漫反射
                        volumeProperty.SetSpecular(0.2)  # 镜面反射
                        # 设置不透明度
                        compositeOpacity = vtkPiecewiseFunction()
                        compositeOpacity.AddPoint(0, 0.00)
                        compositeOpacity.AddPoint(19, 1)
                        volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                        # 设置颜色属性
                        color = vtkColorTransferFunction()
                        color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                        color.AddRGBPoint(19.0, 1.0, 1.0, 1)
                        color.AddRGBPoint(20.0, 0.8, 0.8, 0.8)
                        color.AddRGBPoint(21.0, 0.6, 0.6, 0.6)
                        color.AddRGBPoint(22.0, 0.4, 0.4, 0.4)
                        volumeProperty.SetColor(color)

                        volume = vtkVolume()  # 和vtkActor作用一致
                        volume.SetMapper(volumeMapper)
                        volume.SetProperty(volumeProperty)
                        if self.volume_old is not None:
                            self.ren.RemoveViewProp(self.volume_old)
                        self.ren.AddViewProp(volume)
                        self.volume_old = volume
                        # self.volume_path = fname[0]
                        camera = self.ren.GetActiveCamera()
                        c = volume.GetCenter()
                        camera.SetViewUp(0, 0, 1)
                        camera.SetViewAngle(60)
                        camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                        camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                        camera.Azimuth(30.0)
                        camera.Elevation(30.0)
                        self.iren.Initialize()
                        os.remove(os.path.join('./data', 'new_spine.nii.gz'))
                        self.statusbar.showMessage("The mask of spine has been displayed.")
                else:
                    # self.coord.setText('请加载与图对应的分割结果')
                    self.statusbar.showMessage("Please load a correspronding mask of spine.")

    def cal_fatmus_vol(self):
        excel_3d_data = []
        data_3d_name = ['SF-volume', 'VF-volume', 'AM-volume', 'BM-volume',  # 3
                        'SF-HU_mean', 'SF-HU_25per', 'SF-HU_75per',  # 6
                        'SF-HU_gap', 'SF-HU_25perVar', 'SF-HU_75perVar',  # 9
                        'VF-HU_mean', 'VF-HU_25per', 'VF-HU_75per',  # 12
                        'VF-HU_gap', 'VF-HU_25perVar', 'VF-HU_75perVar',  # 15
                        'AM-HU_mean', 'AM-HU_25per', 'AM-HU_75per',  # 18
                        'AM-HU_gap', 'AM-HU_25perVar', 'AM-HU_75perVar',
                        'BM-HU_mean', 'BM-HU_25per', 'BM-HU_75per',
                        'BM-HU_gap', 'BM-HU_25perVar', 'BM-HU_75perVar']
        self.statusbar.showMessage("Calculating the 3D data of fat and muscle.")
        self.show_message_compute_volume()

        if os.path.exists(self.crop_fatmus_mask_file) == False and self.crop_fatmus_mask is None:
            self.fatmus_seg()
            self.spine_seg()
        elif os.path.exists(self.crop_fatmus_mask_file):
            crop_fatmus_mask = itk.ReadImage(self.crop_fatmus_mask_file)
            self.crop_fatmus_mask = itk.GetArrayFromImage(crop_fatmus_mask)
        if os.path.exists(self.crop_img_file) == True:
            crop_img = itk.ReadImage(self.crop_img_file)
            self.crop_img = itk.GetArrayFromImage(crop_img)
        space = itk.ReadImage(self.crop_img_file).GetSpacing()
        for l in range(1, 5):
            label_mask = self.crop_fatmus_mask.copy()
            label_mask[label_mask != l] = 0
            label_mask[label_mask == l] = 1
            vol = np.sum(label_mask) * space[0] * space[1] * space[2] / 1000
            excel_3d_data.append(vol)
        excel_3d_data = np.around(excel_3d_data, decimals=4)
        excel_3d_data = excel_3d_data.tolist()
        image_flat = self.crop_img.flatten()
        mask_flat = self.crop_fatmus_mask.flatten()

        for l in range(1, 5):
            mask_flat_one = np.zeros(mask_flat.shape)
            mask_flat_one[mask_flat == l] = 1
            mask_indices = np.nonzero(mask_flat_one)
            mask_voxel_values = image_flat[mask_indices]
            excel_3d_data.append(np.mean(mask_voxel_values))
            excel_3d_data.append(np.percentile(mask_voxel_values, 25))
            excel_3d_data.append(np.percentile(mask_voxel_values, 75))
            excel_3d_data.append(np.percentile(mask_voxel_values, 75) - np.percentile(mask_voxel_values, 25))
            excel_3d_data.append(np.percentile(mask_voxel_values, 25) / (
                    np.percentile(mask_voxel_values, 75) - np.percentile(mask_voxel_values, 25)))
            excel_3d_data.append(np.percentile(mask_voxel_values, 75) / (
                    np.percentile(mask_voxel_values, 75) - np.percentile(mask_voxel_values, 25)))

        excel_3d_data = np.around(excel_3d_data, decimals=2)

        self.vf_25var = excel_3d_data[15]
        self.am_mean = excel_3d_data[16]
        self.am_25var = excel_3d_data[21]

        self.AMV_line.setText(str(excel_3d_data[2]))
        self.BMV_line.setText(str(excel_3d_data[3]))
        self.SFV_line.setText(str(excel_3d_data[0]))
        self.VFV_line.setText(str(excel_3d_data[1]))
        self.AM_mean_line.setText(str(excel_3d_data[16]))
        self.AM_itq_line.setText(str(excel_3d_data[19]))
        self.AM_mg_line.setText(str(excel_3d_data[20]))
        self.VF_mg_line.setText(str(excel_3d_data[14]))

        self.AMV_line.setEnabled(False)
        self.BMV_line.setEnabled(False)
        self.SFV_line.setEnabled(False)
        self.VFV_line.setEnabled(False)
        self.AM_mean_line.setEnabled(False)
        self.AM_itq_line.setEnabled(False)
        self.AM_mg_line.setEnabled(False)
        self.VF_mg_line.setEnabled(False)
        self.statusbar.showMessage("Calculated the 3D data of fat and muscle.")
        return excel_3d_data, data_3d_name

    def show_mask(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load mask', directory='data',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0 and self.img is not None:
            img = itk.ReadImage(fname[0])
            self.showmask = itk.GetArrayFromImage(img)
            if self.showmask.shape == self.img.shape:
                self.affine = nib.load(self.img_path).affine
                self.mask_np = nib.load(fname[0]).get_fdata()
                self.img_np = nib.load(self.img_path).get_fdata()
                self.new_img = self.mask_np * self.img_np
                self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
                nib.save(self.new_img, os.path.join('data', 'new_img.nii.gz'))
                if self.prinimg is not None:
                    self.printmask_fatmus(0.5)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                    reader = vtkNIFTIImageReader()
                    reader.SetFileName(os.path.join('data', 'new_img.nii.gz'))
                    reader.Update()
                    volumeMapper = vtkGPUVolumeRayCastMapper()
                    volumeMapper.SetInputData(reader.GetOutput())

                    volumeProperty = vtkVolumeProperty()
                    volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                    volumeProperty.ShadeOn()  # 打开或者关闭阴影
                    volumeProperty.SetAmbient(0.4)
                    volumeProperty.SetDiffuse(0.6)  # 漫反射
                    volumeProperty.SetSpecular(0.2)  # 镜面反射
                    # 设置不透明度
                    compositeOpacity = vtkPiecewiseFunction()
                    compositeOpacity.AddPoint(70, 0.00)
                    compositeOpacity.AddPoint(90, 0.4)
                    compositeOpacity.AddPoint(180, 0.6)
                    volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                    # 设置梯度不透明属性
                    volumeGradientOpacity = vtkPiecewiseFunction()
                    volumeGradientOpacity.AddPoint(10, 0.0)
                    volumeGradientOpacity.AddPoint(90, 0.5)
                    volumeGradientOpacity.AddPoint(100, 1.0)

                    # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                    # 设置颜色属性
                    color = vtkColorTransferFunction()
                    color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                    color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                    color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                    volumeProperty.SetColor(color)

                    volume = vtkVolume()  # 和vtkActor作用一致
                    volume.SetMapper(volumeMapper)
                    volume.SetProperty(volumeProperty)
                    if self.volume_old is not None:
                        self.ren.RemoveViewProp(self.volume_old)
                    self.ren.AddViewProp(volume)
                    self.volume_old = volume
                    # self.volume_path = fname[0]
                    camera = self.ren.GetActiveCamera()
                    c = volume.GetCenter()

                    camera.SetViewUp(0, 0, 1)
                    camera.SetViewAngle(60)
                    camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                    camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                    # camera.SetPosition(c[0], c[1] - 800, c[2] - 200)
                    # camera.SetFocalPoint(c[0], c[1], c[2])
                    camera.Azimuth(30.0)
                    camera.Elevation(30.0)
                    self.iren.Initialize()
                    os.remove(os.path.join('data', 'new_img.nii.gz'))
                    self.statusbar.showMessage("The segmentation result has been displayed")
            else:
                # self.coord.setText('请加载与图对应的分割结果')
                self.statusbar.showMessage("Please load a correspronding segmentation result")
        elif self.img is None:
            # self.coord.setText('请先导入图像')
            self.statusbar.showMessage("Please load a image first")

    def clinicf_read(self):
        self.statusbar.showMessage("Please input data as required.")

    def showmask_path(self, path):
        self.showmask = read_nii(path)
        if self.showmask.shape == self.img.shape:
            self.mask_np = read_nii(path)
            self.img_np = read_nii(self.img_file)
            self.new_img = self.mask_np * self.img_np
            ori2new_nii(path, self.new_img, os.path.join('data', 'new_fatmus.nii.gz'))

            if self.prinimg is not None:
                self.printmask_fatmus(0.5)
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                reader = vtkNIFTIImageReader()
                reader.SetFileName(os.path.join('data', 'new_fatmus.nii.gz'))
                reader.Update()

                volumeMapper = vtkGPUVolumeRayCastMapper()
                volumeMapper.SetInputData(reader.GetOutput())
                volumeProperty = vtkVolumeProperty()
                volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                volumeProperty.ShadeOn()  # 打开或者关闭阴影
                volumeProperty.SetAmbient(0.4)
                volumeProperty.SetDiffuse(0.6)  # 漫反射
                volumeProperty.SetSpecular(0.2)  # 镜面反射
                # 设置不透明度
                compositeOpacity = vtkPiecewiseFunction()
                compositeOpacity.AddPoint(70, 0.00)
                compositeOpacity.AddPoint(90, 0.4)
                compositeOpacity.AddPoint(180, 0.6)
                volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                # 设置梯度不透明属性
                volumeGradientOpacity = vtkPiecewiseFunction()
                volumeGradientOpacity.AddPoint(10, 0.0)
                volumeGradientOpacity.AddPoint(90, 0.5)
                volumeGradientOpacity.AddPoint(100, 1.0)

                # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                # 设置颜色属性
                color = vtkColorTransferFunction()
                color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                volumeProperty.SetColor(color)

                volume = vtkVolume()  # 和vtkActor作用一致
                volume.SetMapper(volumeMapper)
                volume.SetProperty(volumeProperty)
                if self.volume_old is not None:
                    self.ren.RemoveViewProp(self.volume_old)
                self.ren.AddViewProp(volume)
                self.volume_old = volume
                # self.volume_path = fname[0]
                camera = self.ren.GetActiveCamera()
                c = volume.GetCenter()
                camera.SetViewUp(0, 0, 1)
                # camera.SetPosition(c[0], c[1] - 800, c[2] - 200)
                # camera.SetFocalPoint(c[0], c[1], c[2])
                camera.SetViewAngle(60)
                camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                camera.Azimuth(30.0)
                camera.Elevation(30.0)
                self.iren.Initialize()
                os.remove(os.path.join('data', 'new_fatmus.nii.gz'))
                self.cal_fatmus_vol()

    def printmask_fatmus(self, alpha):
        new_prinimg = []
        # 设置每个标记类别的颜色
        color_map = {
            1: [255, 0, 0],  # 标记0的颜色为红色
            2: [0, 255, 0],  # 标记1的颜色为绿色
            3: [0, 0, 255],  # 标记2的颜色为蓝色
            4: [255, 255, 0],  # 标记3的颜色为黄色
            0: [0, 0, 0]  # 标记4的颜色为紫色
        }
        for i in range(self.leng_max):
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max - 1) - i, ...]
            imgone = self.normalize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            # 为每个标记类别设置颜色
            mask_colored = np.zeros_like(maskone_)
            for k in range(3):
                for label in color_map:
                    mask_one_k = maskone_[:, :, k]
                    mask_colored_k = mask_colored[:, :, k]
                    mask_colored_k[mask_one_k == label] = color_map[label][k]
                    mask_colored[:, :, k] = mask_colored_k
            # 将颜色叠加到图像上
            cam_img = alpha * mask_colored + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg
        return new_prinimg

    def printmask_spine(self, alpha):
        new_prinimg = []
        # 设置每个标记类别的颜色
        color_map = {
            19: [255, 0, 0],  # 标记0的颜色为红色
            20: [0, 255, 0],  # 标记1的颜色为绿色
            21: [0, 0, 255],  # 标记2的颜色为蓝色
            22: [255, 255, 0],  # 标记3的颜色为黄色
            0: [0, 0, 0]  # 标记4的颜色为紫色
        }
        for i in range(self.leng_max):
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max - 1) - i, ...]
            imgone = self.normalize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            # 为每个标记类别设置颜色
            mask_colored = np.zeros_like(maskone_)
            for k in range(3):
                for label in color_map:
                    mask_one_k = maskone_[:, :, k]
                    mask_colored_k = mask_colored[:, :, k]
                    mask_colored_k[mask_one_k == label] = color_map[label][k]
                    mask_colored[:, :, k] = mask_colored_k
            # 将颜色叠加到图像上
            cam_img = alpha * mask_colored + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg
        return new_prinimg


    def is_number(self, str):
        try:
            # 因为使用float有一个例外是'NaN'
            if str == 'NaN':
                return False
            float(str)
            return True
        except ValueError:
            return False

    def show_message_predict(self):
        QMessageBox.information(self, "Note", "The prediction is about to be made and will take longer if no "
                                              "segmentation results are entered and automatic segmentation is not performed",
                                QMessageBox.Yes)

    def show_message_compute_volume(self):
        QMessageBox.information(self, "Note", "If the mask is not imported in advance and there are no related "
                                              "files in the directory, the segmentation will be performed automatically, "
                                              "which will take a longer time",
                                QMessageBox.Yes)

    def show_message_incompelete_data(self):
        QMessageBox.information(self, "Note", " Incomplete input data, please input data as required.", QMessageBox.Yes)

    def show_message_fatmus(self):
        QMessageBox.information(self, "Note", "Fat and muscle will be extracted. This will spread a few seconds",
                                QMessageBox.Yes)

    def show_message_spine(self):
        QMessageBox.information(self, "Note", "Spine will be extracted. This will spread a few seconds",
                                QMessageBox.Yes)

    def predictf(self):
        column_index = 1  # 假设要获取第二列的值
        # 获取指定列的值
        values = []
        input_clinic = True

        self.Age = self.Age_line.text()
        self.Sex = self.Sex_line.text()
        self.NLR = self.NLR_line.text()
        self.ALT = self.ALT_line.text()
        self.AFP = self.AFP_line.text()
        self.HBV = self.HBV_line.text()
        self.BCLC = self.BCLC_line.text()
        self.CP = self.CP_line.text()
        values = [self.Age,self.Sex,self.NLR,self.ALT, self.AFP,self.HBV,self.BCLC,self.CP]
        for nn in values:
            if nn is None:
                input_clinic = False
        if input_clinic:
            print(values)
            afpg = self.AFP
            if self.is_number(afpg):
                if int(afpg) == 1:
                    self.afpg1 = 1
                    self.afpg2 = 0
                elif int(afpg) == 2:
                    self.afpg2 = 1
                    self.afpg1 = 0
                elif int(afpg) == 0:
                    self.afpg1 = 0
                    self.afpg2 = 0
                else:
                    self.statusbar.showMessage("AFP_G value is error.")
            else:
                self.afpg1 = None
                self.afpg2 = None
            if self.afpg1 is not None and self.afpg2 is not None and self.vf_25var is not None \
                    and self.am_25var is not None and self.am_mean is not None:
                beta0 = 5.564
                beta_afpg1 = -0.008
                beta_afpg2 = 0.898
                beta_vf_25var = 1.637
                beta_am_mean = -0.172
                beta_am_25var = 1.513
                expon_num = math.exp(
                    beta0 + beta_afpg1 * self.afpg1 + beta_afpg2 * self.afpg2 + beta_vf_25var * self.vf_25var + beta_am_25var * self.am_25var + beta_am_mean * self.am_mean)
                pred = expon_num / (1 + expon_num)
                if pred > 0.1165:
                    self.plotresult.setTextColor(QtCore.Qt.red)
                elif pred <= 0.1165:
                    self.plotresult.setTextColor(QtCore.Qt.red)
                self.plotresult.setText("{0:0.5f}".format(pred))
                self.statusbar.showMessage("The prediction is finished.")
            else:
                # self.coord.setText(f"输入不完整,请按要求导入数据")
                self.show_message_incompelete_data()
                self.statusbar.showMessage("Incomplete input data, please input data as required.")
        else:
            # self.coord.setText(f"输入不完整,请按要求导入数据")
            self.show_message_incompelete_data()
            self.statusbar.showMessage("Incomplete input data, please input data as required.")

    def spine_seg(self):
        if self.img is not None:
            self.spine_mask_file = os.path.join(self.spine_mask_path, self.file_name)
            if not os.path.exists(self.spine_mask_file):
                nnunet_spine_nii_path = os.path.join('data', 'nnunet_in', 'spine_0_0000.nii.gz')
                nnunet_spine_mask_path = os.path.join('data', 'nnunet_out', 'spine_0.nii.gz')
                if os.path.exists(nnunet_spine_nii_path):
                    os.remove(nnunet_spine_nii_path)
                shutil.copy(self.img_file, nnunet_spine_nii_path)
                self.statusbar.showMessage("Start to segment spine, please wait a moment")
                self.show_message_spine()
                os.system("nnUNet_predict -i ./data/nnunet_in "
                          "-o ./data/nnunet_out -t 057 -f 0  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
                shutil.copy(nnunet_spine_mask_path, self.spine_mask_file)
                os.remove(nnunet_spine_nii_path)
                os.remove(nnunet_spine_mask_path)
                os.remove(os.path.join('data', 'nnunet_out', 'plans.pkl'))
                self.statusbar.showMessage("Spine segmentation has been done")
            else:
                self.statusbar.showMessage("Spine segmentation has been existed.")

            if os.path.exists(self.fatmus_mask_file):
                self.spine_crop_img()
            if os.path.exists(self.crop_fatmus_mask_file):
                self.showmask_path(self.crop_fatmus_mask_file)
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load an image first')

    def fatmus_seg(self):
        if self.img is not None:
            self.fatmus_mask_file = os.path.join(self.fatmus_mask_path, self.file_name)
            if not os.path.exists(self.fatmus_mask_file):
                nnunet_fatmus_nii_path = os.path.join('data', 'nnunet_in', 'fatmus_0_0000.nii.gz')
                nnunet_fatmus_mask_path = os.path.join('data', 'nnunet_out', 'fatmus_0.nii.gz')
                if os.path.exists(nnunet_fatmus_nii_path):
                    os.remove(nnunet_fatmus_nii_path)

                shutil.copy(self.img_file, nnunet_fatmus_nii_path)
                self.statusbar.showMessage("Start to segment spine, please wait a moment")
                self.show_message_fatmus()
                os.system("nnUNet_predict -i data/nnunet_in "
                          "-o data/nnunet_out -t 006 -f 1 -m 2d -tr nnUNetTrainerV2 -chk model_best")
                shutil.copy(nnunet_fatmus_mask_path, self.fatmus_mask_file)
                os.remove(nnunet_fatmus_nii_path)
                os.remove(nnunet_fatmus_mask_path)
                os.remove(os.path.join('data', 'nnunet_out', 'plans.pkl'))
                fatmus_mask = itk.ReadImage(self.fatmus_mask_file)
                self.fatmus_mask = itk.GetArrayFromImage(fatmus_mask)
                self.statusbar.showMessage("Fat and muscle segmentation has been done")
            else:
                self.statusbar.showMessage("Fat and muscle segmentation has been done")

            if os.path.exists(self.spine_mask_file):
                self.spine_crop_img()
            if os.path.exists(self.crop_fatmus_mask_file):
                self.showmask_path(self.crop_fatmus_mask_file)
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load an image first')

    def spine_crop_img(self):
        begin_slice = 1000
        end_slice = 0
        l3 = 22
        t12 = 19
        if self.img is not None:
            self.crop_spine_mask_file = os.path.join(self.crop_spine_mask_path, self.file_name)
            self.crop_fatmus_mask_file = os.path.join(self.crop_fatmus_mask_path, self.file_name)
            self.crop_img_file = os.path.join(self.crop_img_path, self.file_name)
            if not os.path.exists(self.crop_spine_mask_file) or not os.path.exists(self.crop_fatmus_mask_file):

                spine_mask_data = read_nii(self.spine_mask_file)
                origin_nii_data = read_nii(self.img_file)
                fatmus_mask_data = read_nii(self.fatmus_mask_file)
                new_img = np.zeros(origin_nii_data.shape)
                new_spine_mask = np.zeros(spine_mask_data.shape)
                new_fatmus_mask = np.zeros(fatmus_mask_data.shape)

                if t12 in spine_mask_data and l3 in spine_mask_data:
                    spine_mask_data = remove_archs_spine_mask(spine_mask_data)
                    spine_mask_data[spine_mask_data < t12] = 0
                    spine_mask_data[spine_mask_data > l3] = 0
                    spine_mask_data_copy = spine_mask_data.copy()
                    spine_mask_data_copy[spine_mask_data_copy != 0] = 1
                    for d in range(spine_mask_data_copy.shape[0]):
                        if 1 in spine_mask_data_copy[d, :, :]:
                            if d < begin_slice:
                                begin_slice = d
                            if d > end_slice:
                                end_slice = d
                    if end_slice > begin_slice:
                        new_spine_mask[begin_slice:end_slice + 1] = spine_mask_data[begin_slice:end_slice + 1]
                        new_fatmus_mask[begin_slice:end_slice + 1] = fatmus_mask_data[begin_slice:end_slice + 1]
                        new_img[begin_slice:end_slice + 1] = origin_nii_data[begin_slice:end_slice + 1]
                    else:
                        new_spine_mask[end_slice:begin_slice + 1] = spine_mask_data[end_slice:begin_slice + 1]
                        new_fatmus_mask[end_slice:begin_slice + 1] = fatmus_mask_data[end_slice:begin_slice + 1]
                        new_img[end_slice:begin_slice + 1] = origin_nii_data[end_slice:begin_slice + 1]
                        a = begin_slice
                        begin_slice = end_slice
                        end_slice = a
                    ori2new_nii(self.spine_mask_file, new_spine_mask, self.crop_spine_mask_file)
                    ori2new_nii(self.fatmus_mask_file, new_fatmus_mask, self.crop_fatmus_mask_file)
                    ori2new_nii(self.img_file, new_img, self.crop_img_file)


                else:
                    self.statusbar.showMessage(
                        "This image does not contain T12 or L3, which can not calculate the volume of fat and muscle correctly. Please change another image.")

        else:
            self.statusbar.showMessage('Please load an image first')


    # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def normalize(self, x):
        maa = x.max()
        mii = x.min()
        x = (x - mii) * 255 / (maa - mii)
        return x

    def eventFilter(self, source, event):
        if source is self.view1:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 1
                return True

        elif source is self.view2:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 2
                return True


        elif source is self.view3:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 3
                # print('3')
                return True
        else:
            self.face_flage = 0

        return False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img + 1
                if self.leng_img >= self.leng_max:
                    self.leng_img = self.leng_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img + 1
                if self.width_img >= self.width_max:
                    self.width_img = self.width_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img + 1
                if self.high_img >= self.high_max:
                    self.high_img = self.high_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


        elif event.angleDelta().y() < 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img - 1
                if self.leng_img < 0:
                    self.leng_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img - 1
                if self.width_img < 0:
                    self.width_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)



            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img - 1
                if self.high_img < 0:
                    self.high_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


if __name__ == '__main__':
    # gpu_id = "4"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from qt_material import apply_stylesheet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app = QApplication(sys.argv)
    # apply_stylesheet(app, theme='dark_lightgreen.xml')
    import qdarkstyle

    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    mw = Window_FM2()
    mw.show()
    sys.exit(app.exec_())
