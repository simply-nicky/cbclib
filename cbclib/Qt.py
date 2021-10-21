"""
Qt.py - PyQt based classes for 3D visualisation using OpenGl library
"""
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL.GL import glEnable, glBlendFunc, glBegin, glColor4f, glVertex3f, glEnd, glHint
from OpenGL.GL import (GL_LINE_SMOOTH, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                       GL_LINE_SMOOTH_HINT, GL_NICEST, GL_LINES)
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLSurfacePlotItem, GLVolumeItem
from pyqtgraph import ImageView, glColor

def make_app():
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    app.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

    return app

class Viewer2D(QtWidgets.QMainWindow):
    def __init__(self, data, label, levels, parent=None, size=(640, 480)):
        QtWidgets.QMainWindow.__init__(parent=parent, size=QtCore.QSize(size[0], size[1]))
        self.setWindowTitle('CBC Viewer')
        self.update_ui(data, label, levels)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.show()

    def update_ui(self, data, label, levels):
        self.layout = QtGui.QVBoxLayout()
        _label_widget = QtGui.QLabel(label)
        _label_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(_label_widget)
        _image_view = ImageView()
        _image_view.setPredefinedGradient('thermal')
        _image_view.setImage(img=data, levels=levels)
        self.layout.addWidget(_image_view)

    def show_data(self, data, label, levels=(0, 100)):
        app = make_app()
        viewer = Viewer2D(data=data, label=label, levels=levels)
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            app.exec_()

class Grid(GLGraphicsItem):
    """
    Coordinates grid class for pyqtgraph package

    size - grid size
    color - grid color
    antialias - flag to turn on antialiasing
    """
    def __init__(self, size=None, color=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if color is None:
            color = (255, 255, 255, 80)
        self.set_color(color)
        if size is None:
            size = QtGui.QVector3D(1, 1, 0)
        self.set_size(size=size)
        self.set_spacing(0.05, 0.05)

    def set_color(self, color):
        """
        Set the color of the grid. Arguments are the same as those accepted by
        pyqtgraph.glColor method
        """
        self.color = glColor(color)
        self.update()

    def set_size(self, x=None, y=None, size=None):
        """
        Set grid size

        x, y - grid size
        """
        if size is not None:
            x = size.x()
            y = size.y()
        self.__size = [x, y]
        self.update()

    def size(self):
        """
        Return grid size
        """
        return self.__size[:]

    def set_spacing(self, x=None, y=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
        if spacing is not None:
            x, y = spacing.x(), spacing.y()
        self.__spacing = [x, y]
        self.update()

    def set_grid(self, x=None, y=None, size=None, ratio=20):
        """
        Set grid size and spacing

        x,y - grid size
        ratio - overall grid to spacing ratio
        """
        if size is not None:
            x = size.x()
            y = size.y()
        self.set_size(x, y)
        self.set_spacing(x / ratio, x / ratio)

    def spacing(self):
        """
        Return grid spacing
        """
        return self.__spacing[:]

    def paint(self):
        """
        Draw grid object
        """
        self.setupGLState()
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glBegin(GL_LINES)
        _x, _y = self.size()
        _xs, _ys = self.spacing()
        _x_vals = np.arange(0, _x + _xs * 0.001, _xs)
        _y_vals = np.arange(0, _y + _ys * 0.001, _ys)
        glColor4f(*self.color)
        for x in _x_vals:
            glVertex3f(x, _y_vals[0], 0)
            glVertex3f(x, _y_vals[-1], 0)
        for y in _y_vals:
            glVertex3f(_x_vals[0], y, 0)
            glVertex3f(_x_vals[-1], y, 0)
        glEnd()

class Viewer3D(GLViewWidget):
    """
    Abstract 3D viewer

    title - plot title
    origin - axes origin
    roi - viewer region of interest
    size - viewer window size
    """
    def __init__(self, title='Plot3D', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        GLViewWidget.__init__(self, parent)
        self.resize(size[0], size[1])
        self.setWindowTitle(title)
        self.origin, self.roi = origin, roi
        self.set_camera()
        self.make_grid()
        
    def make_grid(self):
        """
        Draw axes grid
        """
        self.gx = Grid(color=(255, 255, 255, 50))
        self.gx.set_grid(self.roi[0], self.roi[2])
        self.gx.rotate(90, 1, 0, 0)
        self.gx.translate(*self.origin)
        self.addItem(self.gx)
        self.gy = Grid(color=(255, 255, 255, 50))
        self.gy.set_grid(self.roi[2], self.roi[1])
        self.gy.rotate(90, 0, -1, 0)
        self.gy.translate(*self.origin)
        self.addItem(self.gy)
        self.gz = Grid(color=(255, 255, 255, 50))
        self.gz.set_grid(self.roi[0], self.roi[1])
        self.gz.translate(*self.origin)
        self.addItem(self.gz)

    def set_grid(self, origin, roi):
        """
        Set axes grid origin and region of interest
        """
        self.gx.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gx.set_grid(roi[0], roi[2])
        self.gy.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gy.set_grid(roi[2], roi[1])
        self.gz.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gz.set_grid(roi[0], roi[1])

    def set_grid_color(self, color):
        """
        Set axes grid color
        """
        self.gx.set_color(color)
        self.gy.set_color(color)
        self.gz.set_color(color)
        self.update()

    def set_camera(self):
        """
        Set viewer camera position
        """
        self.opts['center'] = QtGui.QVector3D(self.origin[0] + self.roi[0] / 2,
                                              self.origin[1] + self.roi[1] / 2,
                                              self.origin[2] + self.roi[2] / 2)
        self.opts['distance'] = max(self.roi) * 2
        self.update()

class ScatterViewer(Viewer3D):
    """
    3D scatter data viewer class

    title - plot title
    origin - axes origin
    roi - viewer region of interest
    size - viewer window size    
    """
    def __init__(self, title='Scatter Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        Viewer3D.__init__(self, title, origin, roi, size, parent)
        self.s_p = GLScatterPlotItem()
        self.s_p.setGLOptions('translucent')
        self.addItem(self.s_p)

    def set_data(self, pos, color=(1.0, 1.0, 1.0, 0.5), size=10):
        """
        Update the data displayed by this item. All arguments are optional.
        For example, it is allowed to update spot positions while leaving
        colors unchanged, etc.

        ====================  ==================================================
        **Arguments:**
        pos                   (N,3) array of floats specifying point locations.
        color                 (N,4) array of floats (0.0-1.0) specifying
                              spot colors OR a tuple of floats specifying
                              a single color for all spots.
        size                  (N,) array of floats specifying spot sizes or
                              a single value to apply to all spots.
        ====================  ==================================================
        """
        kwds = {'pos': pos, 'color': color, 'size': size}
        self.s_p.setData(**kwds)
        origin = pos.min(axis=0)
        roi = pos.max(axis=0) - origin
        self.set_grid(origin, roi)
        self.origin, self.roi = origin, roi
        self.set_camera()

class VolumeViewer(Viewer3D):
    """
    3D volumetric data viewer class

    title - plot title
    origin - axes origin
    roi - viewer region of interest
    size - viewer window size    
    """
    def __init__(self, title='Volume Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        Viewer3D.__init__(self, title, origin, roi, size, parent)
        self.v_i = GLVolumeItem(data=None)
        self.addItem(self.v_i)

    def set_data(self, data, smooth=True, sliceDensity=1):
        """
        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 4D numpy array (x, y, z, RGBA) with dtype=ubyte.
        sliceDensity    Density of slices to render through the volume. A value of 1 means one slice per voxel.
        smooth          (bool) If True, the volume slices are rendered with linear interpolation.
        ==============  =======================================================================================
        """
        self.v_i.sliceDensity, self.v_i.smooth = sliceDensity, smooth
        self.v_i.setData(data)
        roi = data.shape[:-1]
        self.set_grid(self.origin, roi)
        self.roi = roi
        self.set_camera()

class SurfaceViewer(Viewer3D):
    """
    3D surface viewer class

    title - plot title
    origin - axes origin
    roi - viewer region of interest
    size - viewer window size    
    """
    def __init__(self, title='Volume Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        Viewer3D.__init__(self, title, origin, roi, size, parent)
        self.s_w = GLSurfacePlotItem(data=None)
        self.addItem(self.s_w)

    def set_data(self, x, y, z, color=(1.0, 1.0, 1.0, 0.5)):
        """
        Update the data in this surface plot

        ==============  =====================================================================
        **Arguments:**
        x, y            1D arrays of values specifying the x,y positions of vertexes in the
                        grid. If these are omitted, then the values will be assumed to be
                        integers.
        z               2D array of height values for each grid vertex.
        color           (width, height, 4) array of vertex colors.
        ==============  =====================================================================

        All arguments are optional.

        Note that if vertex positions are updated, the normal vectors for each triangle must
        be recomputed. This is somewhat expensive if the surface was initialized with smooth=False
        and very expensive if smooth=True. For faster performance, initialize with
        computeNormals=False and use per-vertex colors or a normal-independent shader program.
        """
        colors = np.array(color)
        if colors.shape == (4,):
            colors = colors * np.ones(z.shape + (4,))
        kwds = {'x': x, 'y': y, 'z': z, 'colors': colors}
        self.s_w.setData(**kwds)
        origin = np.array([x.min(), y.min(), z.min()])
        roi = np.array([x.max(), y.max(), z.max()]) - origin
        self.set_grid(origin, roi)
        self.origin, self.roi = origin, roi
        self.set_camera()

def vol_data(data, col=np.array([255, 255, 255]), alpha=1.0):
    """
    Generate volumetric data from 3d numpy array

    data - numpy array
    col - volumetric data color
    alpha - data transparency (0.0 - 1.0)
    """
    voldata = np.empty(data.shape + (4,), dtype=np.ubyte)
    adata = np.log(data - data.min() + 1)
    voldata[..., 0:3] = col
    voldata[..., 3] = adata * (255 / adata.max() * alpha)
    return voldata
