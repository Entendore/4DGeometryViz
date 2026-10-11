import sys
import time
import math
import numpy as np
from itertools import permutations, combinations
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QComboBox, QPushButton, QGroupBox, QCheckBox,
    QSplitter, QFileDialog, QTabWidget, QTextEdit
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QSurfaceFormat, QPainter, QPen
from OpenGL.GL import *
from OpenGL.GLU import *
from PySide6.QtOpenGLWidgets import QOpenGLWidget


# ═══════════════════════════════════════════════════════════════
# SHAPE DATABASE
# ═══════════════════════════════════════════════════════════════
SHAPE_INFO = {
    "Tesseract": {"schlafli": "{4,3,3}", "cells": "8 cubes", "vertices": 16, "edges": 32, "faces": 24, "symmetry": "B₄ (order 384)", "description": "The 4D hypercube. Generalization of the cube to 4 dimensions."},
    "5-cell": {"schlafli": "{3,3,3}", "cells": "5 tetrahedra", "vertices": 5, "edges": 10, "faces": 10, "symmetry": "A₄ (order 120)", "description": "The 4D simplex. Simplest regular 4-polytope."},
    "16-cell": {"schlafli": "{3,3,4}", "cells": "16 tetrahedra", "vertices": 8, "edges": 24, "faces": 32, "symmetry": "B₄ (order 384)", "description": "The 4D cross-polytope. Dual of the tesseract."},
    "24-cell": {"schlafli": "{3,4,3}", "cells": "24 octahedra", "vertices": 24, "edges": 96, "faces": 96, "symmetry": "F₄ (order 1152)", "description": "Unique to 4D with no 3D analog. Self-dual."},
    "600-cell": {"schlafli": "{3,3,5}", "cells": "600 tetrahedra", "vertices": 120, "edges": 720, "faces": 1200, "symmetry": "H₄ (order 14400)", "description": "The 4D analog of the icosahedron."},
    "Clifford Torus": {"schlafli": "—", "cells": "—", "vertices": 400, "edges": 800, "faces": 400, "symmetry": "T²", "description": "A flat torus embedded in 4D space. The product of two circles."},
    "Runcinated 5-cell": {"schlafli": "t₀,₃{3,3,3}", "cells": "5 tetrahedra + 5 octahedra + 10 triangular prisms", "vertices": 20, "edges": 60, "faces": 70, "symmetry": "A₄ (order 120)", "description": "Uniform 4-polytope created by expanding the 5-cell."},
    "Rectified Tesseract": {"schlafli": "r{4,3,3}", "cells": "8 cuboctahedra + 16 tetrahedra", "vertices": 32, "edges": 96, "faces": 96, "symmetry": "B₄ (order 384)", "description": "Created by truncating tesseract vertices at edge midpoints."},
    "Truncated Tesseract": {"schlafli": "t{4,3,3}", "cells": "8 truncated cubes + 16 tetrahedra", "vertices": 64, "edges": 128, "faces": 64, "symmetry": "B₄ (order 384)", "description": "Truncating tesseract vertices at 1/3 of edge length."},
    "3-3 Duoprism": {"schlafli": "{3}×{3}", "cells": "6 triangular prisms", "vertices": 9, "edges": 18, "faces": 6, "symmetry": "D₃×D₃ (order 72)", "description": "Cartesian product of two triangles."},
    "4-3 Duoprism": {"schlafli": "{4}×{3}", "cells": "12 prisms", "vertices": 12, "edges": 24, "faces": 7, "symmetry": "D₄×D₃ (order 192)", "description": "Cartesian product of a square and a triangle."},
    "5-5 Duoprism": {"schlafli": "{5}×{5}", "cells": "10 pentagonal prisms", "vertices": 25, "edges": 50, "faces": 10, "symmetry": "D₅×D₅ (order 200)", "description": "Cartesian product of two pentagons."},
    "Dodecahedral Subset": {"schlafli": "—", "cells": "—", "vertices": 20, "edges": 30, "faces": 0, "symmetry": "H₃ subset", "description": "3D dodecahedron embedded in 4D space (w=0 plane)."}
}


# ═══════════════════════════════════════════════════════════════
# 4D SHAPE GENERATORS
# ═══════════════════════════════════════════════════════════════
class FourDShapes:
    @staticmethod
    def tesseract():
        vertices = np.array([[x, y, z, w] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]], dtype=np.float64)
        edges = [[i, j] for i in range(16) for j in range(i+1, 16) if np.sum(vertices[i] != vertices[j]) == 1]
        faces = []
        for d1 in range(4):
            for d2 in range(d1+1, 4):
                fixed = [i for i in range(4) if i not in [d1, d2]]
                for f1 in [-1, 1]:
                    for f2 in [-1, 1]:
                        fv = []
                        for v1 in [-1, 1]:
                            for v2 in [-1, 1]:
                                v = [0,0,0,0]; v[d1] = v1; v[d2] = v2; v[fixed[0]] = f1; v[fixed[1]] = f2
                                idx = np.where((vertices == v).all(axis=1))[0][0]
                                fv.append(idx)
                        faces.append(fv)
        return vertices, edges, faces, "Tesseract"

    @staticmethod
    def simplex_4d():
        vertices = np.array([[1, 1, 1, -1], [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [0, 0, 0, 1]], dtype=np.float64) * 0.6
        edges = [[i, j] for i in range(5) for j in range(i+1, 5)]
        faces = [[i, j, k] for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
        return vertices, edges, faces, "5-cell"

    @staticmethod
    def cell_16():
        vertices = []
        for i in range(4):
            v = [0,0,0,0]; v[i] = 1; vertices.append(v)
            v = [0,0,0,0]; v[i] = -1; vertices.append(v)
        vertices = np.array(vertices, dtype=np.float64)
        edges = [[i, j] for i in range(8) for j in range(i+1, 8) if not np.allclose(vertices[i], -vertices[j])]
        faces = []
        for i in range(8):
            for j in range(i+1, 8):
                if np.allclose(vertices[j], -vertices[i]): continue
                for k in range(j+1, 8):
                    if np.allclose(vertices[k], -vertices[i]): continue
                    if np.allclose(vertices[k], -vertices[j]): continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "16-cell"

    @staticmethod
    def cell_24():
        vertices = []
        for i in range(4):
            for j in range(i+1, 4):
                for c1 in [1, -1]:
                    for c2 in [1, -1]:
                        v = [0,0,0,0]; v[i] = c1; v[j] = c2
                        vertices.append(v)
        vertices = np.array(vertices, dtype=np.float64)
        edges = [[i, j] for i in range(24) for j in range(i+1, 24) if abs(np.linalg.norm(vertices[i]-vertices[j]) - np.sqrt(2)) < 0.01]
        edge_set = set(tuple(sorted(e)) for e in edges)
        faces = []
        for i in range(24):
            for j in range(i+1, 24):
                if tuple(sorted([i,j])) not in edge_set: continue
                for k in range(j+1, 24):
                    if tuple(sorted([i,k])) not in edge_set: continue
                    if tuple(sorted([j,k])) not in edge_set: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "24-cell"

    @staticmethod
    def cell_600():
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for s1 in [0.5, -0.5]:
            for s2 in [0.5, -0.5]:
                for s3 in [0.5, -0.5]:
                    for s4 in [0.5, -0.5]:
                        vertices.append([s1, s2, s3, s4])
        for i in range(4):
            for s in [1, -1]:
                v = [0,0,0,0]; v[i] = s; vertices.append(v)
        base_vals = [phi/2, 0.5, 1/(2*phi), 0]
        def is_even_perm(p):
            inv = sum(1 for i in range(len(p)) for j in range(i+1, len(p)) if p[i] > p[j])
            return inv % 2 == 0
        even_perms = [p for p in permutations(range(4)) if is_even_perm(p)]
        for p in even_perms:
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    for s3 in [1, -1]:
                        v = [0,0,0,0]
                        v[p[0]] = s1 * base_vals[0]; v[p[1]] = s2 * base_vals[1]; v[p[2]] = s3 * base_vals[2]
                        vertices.append(v)
        vertices = np.unique(np.array(vertices, dtype=np.float64), axis=0)
        target = 1/phi
        edges = [[i, j] for i in range(len(vertices)) for j in range(i+1, len(vertices)) if abs(np.linalg.norm(vertices[i]-vertices[j]) - target) < 0.01]
        edge_set = set(tuple(sorted(e)) for e in edges)
        faces = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if tuple(sorted([i,j])) not in edge_set: continue
                for k in range(j+1, len(vertices)):
                    if tuple(sorted([i,k])) not in edge_set: continue
                    if tuple(sorted([j,k])) not in edge_set: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "600-cell"

    @staticmethod
    def clifford_torus():
        n = 20
        vertices = []
        for i in range(n):
            a1 = 2 * np.pi * i / n
            for j in range(n):
                a2 = 2 * np.pi * j / n
                vertices.append([np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)])
        vertices = np.array(vertices, dtype=np.float64)
        
        edges = []
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                edges.append([idx, i * n + (j+1)%n])
                edges.append([idx, ((i+1)%n) * n + j])
        edges = [list(e) for e in set(tuple(sorted(e)) for e in edges)]
        
        faces = []
        for i in range(n):
            for j in range(n):
                v0 = i * n + j
                v1 = i * n + (j+1)%n
                v2 = ((i+1)%n) * n + (j+1)%n
                v3 = ((i+1)%n) * n + j
                faces.append([v0, v1, v2, v3])
        return vertices, edges, faces, "Clifford Torus"

    @staticmethod
    def runcinated_5_cell():
        v5 = np.array([
            [1, 1, 1, -1], [1, -1, -1, -1], [-1, 1, -1, -1],
            [-1, -1, 1, -1], [0, 0, 0, 1]
        ], dtype=np.float64) * 0.6
        vertices = []
        for i, j in combinations(range(5), 2):
            vertices.append((v5[i] + v5[j]) / 2.0)
        vertices = np.array(vertices, dtype=np.float64)
        
        dists = np.linalg.norm(vertices[:, None] - vertices, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists[dists < np.inf])
        
        edges = [[i, j] for i in range(20) for j in range(i+1, 20) if abs(np.linalg.norm(vertices[i]-vertices[j]) - min_dist) < 0.05]
        edge_set = set(tuple(sorted(e)) for e in edges)
        faces = []
        for i in range(20):
            for j in range(i+1, 20):
                if tuple(sorted([i,j])) not in edge_set: continue
                for k in range(j+1, 20):
                    if tuple(sorted([i,k])) not in edge_set: continue
                    if tuple(sorted([j,k])) not in edge_set: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "Runcinated 5-cell"

    @staticmethod
    def rectified_tesseract():
        vertices = []
        for i in range(4):
            for j in range(i+1, 4):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0,0,0,0]; v[i] = s1; v[j] = s2
                        vertices.append(v)
        vertices = np.array(vertices, dtype=np.float64)
        edges = [[i, j] for i in range(len(vertices)) for j in range(i+1, len(vertices)) if abs(np.linalg.norm(vertices[i]-vertices[j]) - np.sqrt(2)) < 0.01]
        edge_set = set(tuple(sorted(e)) for e in edges)
        faces = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if tuple(sorted([i,j])) not in edge_set: continue
                for k in range(j+1, len(vertices)):
                    if tuple(sorted([i,k])) not in edge_set: continue
                    if tuple(sorted([j,k])) not in edge_set: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "Rectified Tesseract"

    @staticmethod
    def truncated_tesseract():
        vertices = []
        phi3 = 1/3
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    for s4 in [-1, 1]:
                        base = [s1, s2, s3, s4]
                        for dim in range(4):
                            v = base.copy()
                            v[dim] = v[dim] * (1 - phi3)
                            vertices.append(v)
        vertices = np.unique(np.array(vertices, dtype=np.float64), axis=0)
        dists = np.linalg.norm(vertices[:, None] - vertices, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists[dists < np.inf])
        edges = [[i, j] for i in range(len(vertices)) for j in range(i+1, len(vertices)) if abs(np.linalg.norm(vertices[i]-vertices[j]) - min_dist) < 0.05]
        faces = []
        edge_set = set(tuple(sorted(e)) for e in edges)
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if tuple(sorted([i,j])) not in edge_set: continue
                for k in range(j+1, len(vertices)):
                    if tuple(sorted([i,k])) not in edge_set: continue
                    if tuple(sorted([j,k])) not in edge_set: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "Truncated Tesseract"

    @staticmethod
    def duoprism_3_3():
        vertices = []
        for i in range(3):
            a1 = 2*np.pi*i/3
            for j in range(3):
                a2 = 2*np.pi*j/3
                vertices.append([np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)])
        vertices = np.array(vertices, dtype=np.float64)
        edges = [[i, j] for i in range(9) for j in range(i+1, 9) if np.linalg.norm(vertices[i]-vertices[j]) < 1.5]
        faces = [[0,1,4,3], [1,2,5,4], [2,0,3,5], [0,1,7,6], [1,2,8,7], [2,0,6,8]]
        return vertices, edges, faces, "3-3 Duoprism"

    @staticmethod
    def duoprism_4_3():
        vertices = []
        for i in range(4):
            a1 = 2*np.pi*i/4
            for j in range(3):
                a2 = 2*np.pi*j/3
                vertices.append([np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)])
        vertices = np.array(vertices, dtype=np.float64) * 0.8
        edges = []
        for j in range(3):
            for i in range(4):
                edges.append([j*4+i, j*4+((i+1)%4)])
        for i in range(4):
            for j in range(3):
                edges.append([j*4+i, ((j+1)%3)*4+i])
        edges = [list(e) for e in set(tuple(sorted(e)) for e in edges)]
        faces = []
        for j in range(3):
            faces.append([j*4, j*4+1, j*4+2, j*4+3])
        for i in range(4):
            faces.append([i, i+4, i+8])
        return vertices, edges, faces, "4-3 Duoprism"

    @staticmethod
    def duoprism_5_5():
        vertices = []
        for i in range(5):
            a1 = 2*np.pi*i/5
            for j in range(5):
                a2 = 2*np.pi*j/5
                vertices.append([np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)])
        vertices = np.array(vertices, dtype=np.float64) * 0.8
        edges = []
        for j in range(5):
            for i in range(5):
                edges.append([j*5+i, j*5+((i+1)%5)])
        for i in range(5):
            for j in range(5):
                edges.append([j*5+i, ((j+1)%5)*5+i])
        edges = [list(e) for e in set(tuple(sorted(e)) for e in edges)]
        faces = []
        for j in range(5):
            faces.append([j*5+i for i in range(5)])
        for i in range(5):
            faces.append([j*5+i for j in range(5)])
        return vertices, edges, faces, "5-5 Duoprism"

    @staticmethod
    def hyperdodecahedron():
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for x in [1, -1]:
            for y in [1, -1]:
                for z in [1, -1]:
                    for w in [1, -1]:
                        vertices.append([x, y, z, w])
        for perm in [[phi, 1/phi, 0, 0], [-phi, 1/phi, 0, 0]]:
            for _ in range(4):
                perm = perm[1:] + perm[:1]
                vertices.append(perm.copy())
        vertices = np.array(vertices[:20], dtype=np.float64)
        edges = [[i, j] for i in range(len(vertices)) for j in range(i+1, len(vertices)) if np.linalg.norm(vertices[i]-vertices[j]) < 2.5]
        return vertices, edges, [], "Dodecahedral Subset"


# ═══════════════════════════════════════════════════════════════
# QUATERNION MATH UTILITIES
# ═══════════════════════════════════════════════════════════════
def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm > 0: return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def axis_angle_to_quat(axis, angle):
    axis = normalize_quat(np.append(0, axis))[1:] 
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0: return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / axis_norm
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])

def quat_to_rot_matrix(q):
    w, x, y, z = normalize_quat(q)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def get_4d_rot_matrix(qL, qR):
    qL = normalize_quat(qL)
    qR = normalize_quat(qR)
    wL, xL, yL, zL = qL
    wR, xR, yR, zR = qR
    
    L = np.array([
        [ wL, -xL, -yL, -zL],
        [ xL,  wL, -zL,  yL],
        [ yL,  zL,  wL, -xL],
        [ zL, -yL,  xL,  wL]
    ])
    R = np.array([
        [ wR, -xR, -yR, -zR],
        [ xR,  wR,  zR, -yR],
        [ yR, -zR,  wR,  xR],
        [ zR,  yR, -xR,  wR]
    ])
    M = L @ R
    P = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    return P @ M @ P.T


# ═══════════════════════════════════════════════════════════════
# 4D RENDERER
# ═══════════════════════════════════════════════════════════════
class FourDRenderer(QOpenGLWidget):
    WIRE = 0; SOLID = 1; POINTS = 2; HYBRID = 3; HOLOGRAM = 4
    SINGLE = 0; QUAD = 1; STEREO = 2; VR_STEREO = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = np.array([])
        self.edges = []
        self.faces = []
        self.shape_name = ""

        self.rotation_angles = [0.0] * 6
        self.use_quaternion_spin = False
        
        self.q4d_L = np.array([1.0, 0.0, 0.0, 0.0])
        self.q4d_R = np.array([1.0, 0.0, 0.0, 0.0])
        self.q4d_L_input = np.array([0.0, 1.0, 0.0, 0.0])
        self.q4d_R_input = np.array([0.0, 0.0, 1.0, 0.0])
        self.q4d_L_speed = 2.0
        self.q4d_R_speed = 3.0

        self.cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.cam_pan = [0.0, 0.0, 0.0]

        self.projection_distance = 3.0
        self.scale = 1.0
        self.render_mode = 0
        self.projection_mode = 0
        self.view_mode = 0
        self.normalize_4d = False

        self.w_slice_enabled = False
        self.w_slice_value = 0.0
        self.w_slice_auto_sweep = False
        self.w_slice_sweep_dir = 2

        self.show_vertices = True
        self.show_axes = True
        self.show_legend = True
        self.show_fps = True
        self.show_labels = False
        self.show_grid = False
        self.show_shadow = False
        self.show_glow = False
        self.show_fog = True
        self.show_true_slice = True
        self.use_smooth_shading = True
        
        # Trails
        self.use_trails = False
        self.trail_history = []
        self.trail_length = 15
        
        self.color_mode = 0
        self.line_width = 2.0
        self.face_opacity = 0.75
        self.bg_color = [0.02, 0.02, 0.08, 1.0]

        self.zoom = 6.0
        self.auto_rotate = False
        self.auto_rotate_speed = 0.5
        
        self.last_mouse_pos = None
        self.setMinimumSize(600, 500)

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(lambda: self.update() if self.show_fps else None)
        self.fps_timer.start(500)
        
        self.selected_vertex = -1
        
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_rotate_step)
        self.auto_timer.start(16)

    def set_shape(self, vertices, edges, faces, name):
        self.vertices = vertices.copy()
        self.edges = edges
        self.faces = faces
        self.shape_name = name
        self.selected_vertex = -1
        self.trail_history.clear()
        self.update()

    def set_rotation(self, plane_index, angle):
        self.rotation_angles[plane_index] = angle
        self.update()

    def set_quaternion_spin_enabled(self, enabled):
        self.use_quaternion_spin = enabled
        if enabled:
            self.q4d_L = np.array([1.0, 0.0, 0.0, 0.0])
            self.q4d_R = np.array([1.0, 0.0, 0.0, 0.0])
        self.update()

    def set_quaternion_input(self, is_left, axis, val):
        if is_left:
            self.q4d_L_input[axis+1] = val / 100.0
        else:
            self.q4d_R_input[axis+1] = val / 100.0

    def set_quaternion_speed(self, is_left, val):
        if is_left:
            self.q4d_L_speed = val / 10.0
        else:
            self.q4d_R_speed = val / 10.0

    def set_projection_distance(self, distance):
        self.projection_distance = distance
        self.update()

    def set_projection_mode(self, mode):
        self.projection_mode = mode
        self.update()

    def set_scale(self, scale):
        self.scale = scale
        self.update()

    def set_line_width(self, width):
        self.line_width = width
        self.update()

    def set_render_mode(self, mode):
        self.render_mode = mode
        self.update()

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.update()

    def set_w_slice(self, enabled, value):
        self.w_slice_enabled = enabled
        self.w_slice_value = value
        self.update()

    def set_color_mode(self, mode):
        self.color_mode = mode
        self.update()

    def set_face_opacity(self, opacity):
        self.face_opacity = opacity
        self.update()

    def set_bg_color(self, r, g, b):
        self.bg_color = [r, g, b, 1.0]
        self.update()

    def set_show_legend(self, show):
        self.show_legend = show
        self.update()

    def set_show_fps(self, show):
        self.show_fps = show
        self.update()

    def set_show_labels(self, show):
        self.show_labels = show
        self.update()
        
    def set_show_grid(self, show):
        self.show_grid = show
        self.update()
        
    def set_show_shadow(self, show):
        self.show_shadow = show
        self.update()
        
    def set_show_glow(self, show):
        self.show_glow = show
        self.update()
        
    def set_show_fog(self, show):
        self.show_fog = show
        self.update()
        
    def set_show_true_slice(self, show):
        self.show_true_slice = show
        self.update()
        
    def set_auto_rotate(self, enabled):
        self.auto_rotate = enabled
        self.update()
        
    def set_w_sweep(self, enabled):
        self.w_slice_auto_sweep = enabled

    def set_normalize_4d(self, enabled):
        self.normalize_4d = enabled
        self.update()

    def set_smooth_shading(self, enabled):
        self.use_smooth_shading = enabled
        self.update()

    def set_trails(self, enabled):
        self.use_trails = enabled
        if not enabled: self.trail_history.clear()
        self.update()

    def reset_view(self):
        self.cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.cam_pan = [0.0, 0.0, 0.0]
        self.zoom = 6.0
        self.rotation_angles = [0.0] * 6
        self.q4d_L = np.array([1.0, 0.0, 0.0, 0.0])
        self.q4d_R = np.array([1.0, 0.0, 0.0, 0.0])
        self.trail_history.clear()
        self.update()

    def rotate_4d_euler(self, vertices, angle, plane):
        if len(vertices) == 0: return vertices
        angle_rad = np.radians(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        matrices = [
            np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]),
            np.array([[c,0,-s,0],[0,1,0,0],[s,0,c,0],[0,0,0,1]]),
            np.array([[c,0,0,-s],[0,1,0,0],[0,0,1,0],[s,0,0,c]]),
            np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]]),
            np.array([[1,0,0,0],[0,c,0,-s],[0,0,1,0],[0,s,0,c]]),
            np.array([[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]])
        ]
        return vertices @ matrices[plane].T

    def project_4d_to_3d(self, vertices):
        if len(vertices) == 0: return np.array([]), np.array([])
        result = []
        w_coords = []
        for v in vertices:
            w = v[3]
            w_coords.append(w)
            if self.projection_mode == 0:
                denom = self.projection_distance - w
                if abs(denom) < 0.01: denom = 0.01
                factor = self.projection_distance / denom
            else:
                denom = 2.0 - w
                if abs(denom) < 0.01: denom = 0.01
                factor = 2.0 / denom
            result.append([v[0]*factor, v[1]*factor, v[2]*factor])
        return np.array(result), np.array(w_coords)

    def get_color(self, w, min_w, max_w):
        if self.color_mode == 4: return (0.8, 0.8, 0.9, 0.9)
        if max_w - min_w < 0.001: t = 0.5
        else: t = (w - min_w) / (max_w - min_w)

        if self.color_mode == 0: return (t, 0.2 + 0.6*(1-abs(2*t-1)), 1-t, 0.9)
        elif self.color_mode == 1:
            h = t * 6; c = 0.9; x = c*(1-abs(h%2-1))
            if h < 1: r,g,b = c,x,0
            elif h < 2: r,g,b = x,c,0
            elif h < 3: r,g,b = 0,c,x
            elif h < 4: r,g,b = 0,x,c
            elif h < 5: r,g,b = x,0,c
            else: r,g,b = c,0,x
            return (r+0.1, g+0.1, b+0.1, 0.9)
        elif self.color_mode == 2:
            intensity = 0.5 + 0.5*t
            if t < 0.5: return (0.2, intensity, intensity*0.8, 0.9)
            else: return (intensity, 0.3, 0.5+0.5*(1-t), 0.9)
        else:
            intensity = 0.3 + 0.7*t
            return (intensity*0.7, intensity*0.85, intensity, 0.9)

    def initializeGL(self):
        glClearColor(*self.bg_color)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, -3.0, -10.0, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClearColor(*self.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.show_fog:
            glEnable(GL_FOG)
            glFogi(GL_FOG_MODE, GL_EXP2)
            glFogf(GL_FOG_DENSITY, 0.12)
            glFogfv(GL_FOG_COLOR, [self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0])
        else:
            glDisable(GL_FOG)

        glLoadIdentity()
        if self.view_mode == 1: self._render_quad_view()
        elif self.view_mode == 2: self._render_stereo()
        elif self.view_mode == 3: self._render_vr_stereo()
        else: self._render_single()

        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        self.draw_overlay()

    def _render_single(self):
        w, h = self.width(), self.height()
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        aspect = w / h if h > 0 else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        self._setup_camera()
        self._draw_scene()

    def _render_quad_view(self):
        w, h = self.width(), self.height()
        hw, hh = w // 2, h // 2
        glViewport(0, hh, hw, hh)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45.0, hw/hh if hh > 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        self._setup_camera(); self._draw_scene()
        
        glViewport(hw, hh, hw, hh)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-3, 3, -3, 3, -100, 100)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        gluLookAt(0, 5, 0, 0, 0, 0, 0, 0, -1)
        self._draw_scene()
        
        glViewport(0, 0, hw, hh)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-3, 3, -3, 3, -100, 100)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        self._draw_scene()
        
        glViewport(hw, 0, hw, hh)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-3, 3, -3, 3, -100, 100)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        gluLookAt(5, 0, 0, 0, 0, 0, 0, 1, 0)
        self._draw_scene()

    def _render_stereo(self):
        w, h = self.width(), self.height()
        eye_sep = 0.15
        
        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        aspect = w / h if h > 0 else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(eye_sep/2, 0, -self.zoom)
        glTranslatef(*self.cam_pan)
        glMultMatrixf(quat_to_rot_matrix(self.cam_quat).T)
        self._draw_scene()
        
        glClear(GL_DEPTH_BUFFER_BIT)
        glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(-eye_sep/2, 0, -self.zoom)
        glTranslatef(*self.cam_pan)
        glMultMatrixf(quat_to_rot_matrix(self.cam_quat).T)
        self._draw_scene()
        
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

    def _render_vr_stereo(self):
        w, h = self.width(), self.height()
        hw = w // 2
        eye_sep = 0.15
        
        glViewport(0, 0, hw, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        aspect = hw / h if h > 0 else 1
        gluPerspective(90.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(eye_sep/2, 0, -self.zoom)
        glTranslatef(*self.cam_pan)
        glMultMatrixf(quat_to_rot_matrix(self.cam_quat).T)
        self._draw_scene()
        
        glViewport(hw, 0, hw, h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        aspect = hw / h if h > 0 else 1
        gluPerspective(90.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(-eye_sep/2, 0, -self.zoom)
        glTranslatef(*self.cam_pan)
        glMultMatrixf(quat_to_rot_matrix(self.cam_quat).T)
        self._draw_scene()

    def _setup_camera(self):
        glTranslatef(0, 0, -self.zoom)
        glTranslatef(*self.cam_pan)
        glMultMatrixf(quat_to_rot_matrix(self.cam_quat).T)

    def _draw_scene(self):
        glDisable(GL_LIGHTING)
        if self.show_grid: self._draw_ground_grid()
        if self.show_axes: self.draw_axes()

        if len(self.vertices) == 0: return

        if self.use_quaternion_spin:
            M = get_4d_rot_matrix(self.q4d_L, self.q4d_R)
            rotated = self.vertices @ M.T
        else:
            rotated = self.vertices.copy()
            for i, angle in enumerate(self.rotation_angles):
                rotated = self.rotate_4d_euler(rotated, angle, i)

        if self.normalize_4d and len(rotated) > 0:
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            norms[norms == 0] = 1
            rotated = rotated / norms

        projected, w_coords = self.project_4d_to_3d(rotated)
        if len(projected) == 0: return
        projected *= self.scale

        # Update Trails
        if self.use_trails:
            self.trail_history.append(projected.copy())
            if len(self.trail_history) > self.trail_length:
                self.trail_history.pop(0)

        min_w, max_w = (np.min(w_coords), np.max(w_coords)) if len(w_coords) > 0 else (0, 1)
        visibility = np.ones(len(projected), dtype=bool)
        if self.w_slice_enabled and not self.show_true_slice:
            visibility = rotated[:, 3] <= self.w_slice_value

        # Draw Trails
        if self.use_trails and len(self.trail_history) > 1:
            self._draw_trails(min_w, max_w)

        if self.show_shadow: self._draw_shadow(projected, w_coords, min_w, max_w, visibility)
        
        if self.render_mode in [1, 3, 4] and self.faces: 
            self._draw_solid_faces(projected, w_coords, min_w, max_w, visibility)
        if self.render_mode in [0, 3, 4]: 
            self._draw_wireframe(projected, w_coords, min_w, max_w, visibility, rotated)
        if self.show_vertices and self.render_mode != 2: self._draw_vertices(projected, w_coords, min_w, max_w, visibility, rotated)
        if self.render_mode == 2: self._draw_points_only(projected, w_coords, min_w, max_w, visibility, rotated)
        if self.show_glow: self._draw_glow(projected, w_coords, min_w, max_w, visibility, rotated)
        
        if self.w_slice_enabled and self.show_true_slice and self.faces:
            self._draw_true_slice(projected, rotated)
        elif self.w_slice_enabled:
            self._draw_slice_points(projected, rotated)

        if self.show_labels: self._draw_vertex_labels(projected, w_coords, min_w, max_w, visibility, rotated)

    def _draw_trails(self, min_w, max_w):
        glDisable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i, hist in enumerate(self.trail_history[:-1]):
            alpha = (i / len(self.trail_history)) * 0.6
            for v in hist:
                # Use a generic color based on history index, since w_coords aren't stored for history
                # To keep it simple, use a bright fading white/cyan
                glColor4f(0.5, 0.8, 1.0, alpha)
                glVertex3fv(v)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _draw_ground_grid(self):
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glColor4f(0.15, 0.15, 0.15, 0.5)
            glVertex3f(i, -2, -5); glVertex3f(i, -2, 5)
            glVertex3f(-5, -2, i); glVertex3f(5, -2, i)
        glEnd()

    def _draw_shadow(self, projected, w_coords, min_w, max_w, visibility):
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(0, -2, 0)
        glBegin(GL_POLYGON)
        for face in self.faces:
            face_indices = np.array(face)
            if not np.any(visibility[face_indices]): continue
            for idx in face_indices:
                glColor4f(0, 0, 0, 0.3)
                glVertex3f(projected[idx][0], 0, projected[idx][2])
        glEnd()
        glPopMatrix()

    def _compute_vertex_normals(self, projected, visibility):
        v_normals = np.zeros_like(projected)
        for face in self.faces:
            face_indices = np.array(face)
            if not np.any(visibility[face_indices]): continue
            pts = [projected[idx] for idx in face_indices]
            v0, v1, v2 = pts[0], pts[1], pts[2]
            normal = np.cross(v1 - v0, v2 - v0)
            face_center = np.mean(pts, axis=0)
            if np.dot(normal, face_center) < 0:
                normal = -normal
            for idx in face_indices:
                v_normals[idx] += normal
        norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return v_normals / norms

    def _draw_solid_faces(self, projected, w_coords, min_w, max_w, visibility):
        glEnable(GL_LIGHTING)
        
        v_normals = None
        if self.use_smooth_shading:
            v_normals = self._compute_vertex_normals(projected, visibility)

        face_data = []
        for face in self.faces:
            face_indices = np.array(face)
            if not np.any(visibility[face_indices]): continue
            avg_z = np.mean([projected[idx][2] for idx in face_indices])
            face_data.append((avg_z, face))
        face_data.sort(key=lambda x: -x[0])

        for _, face in face_data:
            face_indices = np.array(face)
            pts = [projected[idx] for idx in face_indices]
            v0, v1, v2 = pts[0], pts[1], pts[2]
            normal = np.cross(v1 - v0, v2 - v0)
            face_center = np.mean(pts, axis=0)

            if np.dot(normal, face_center) < 0:
                draw_indices = face_indices[::-1]
                normal = -normal
            else:
                draw_indices = face_indices

            norm_len = np.linalg.norm(normal)
            if norm_len > 0: normal /= norm_len

            avg_w = np.mean([w_coords[idx] for idx in face_indices])
            c = self.get_color(avg_w, min_w, max_w)

            glBegin(GL_POLYGON)
            if self.use_smooth_shading and v_normals is not None:
                for idx in draw_indices:
                    glNormal3fv(v_normals[idx])
                    c_v = self.get_color(w_coords[idx], min_w, max_w)
                    if self.render_mode == 4: # Hologram
                        glColor4f(c_v[0]*0.1, c_v[1]*0.1, c_v[2]*0.1, self.face_opacity * 0.5)
                    else:
                        glColor4f(c_v[0], c_v[1], c_v[2], self.face_opacity)
                    glVertex3fv(projected[idx])
            else:
                glNormal3fv(normal)
                if self.render_mode == 4:
                    glColor4f(c[0]*0.1, c[1]*0.1, c[2]*0.1, self.face_opacity * 0.5)
                else:
                    glColor4f(c[0], c[1], c[2], self.face_opacity)
                for idx in draw_indices: glVertex3fv(projected[idx])
            glEnd()

        glDisable(GL_LIGHTING)
        
        # Hologram edge glow
        if self.render_mode == 4:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glLineWidth(self.line_width * 1.5)
            glBegin(GL_LINES)
            for face in self.faces:
                face_indices = np.array(face)
                if not np.any(visibility[face_indices]): continue
                for i in range(len(face_indices)):
                    idx1 = face_indices[i]
                    idx2 = face_indices[(i+1)%len(face_indices)]
                    c1 = self.get_color(w_coords[idx1], min_w, max_w)
                    c2 = self.get_color(w_coords[idx2], min_w, max_w)
                    glColor4f(c1[0], c1[1], c1[2], 0.9)
                    glVertex3fv(projected[idx1])
                    glColor4f(c2[0], c2[1], c2[2], 0.9)
                    glVertex3fv(projected[idx2])
            glEnd()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        elif self.render_mode == 1 or self.render_mode == 3:
            glLineWidth(1.0)
            for face in self.faces:
                face_indices = np.array(face)
                if not np.any(visibility[face_indices]): continue
                glBegin(GL_LINE_LOOP)
                for idx in face_indices:
                    c = self.get_color(w_coords[idx], min_w, max_w)
                    glColor4f(c[0]*0.4, c[1]*0.4, c[2]*0.4, 0.8)
                    glVertex3fv(projected[idx])
                glEnd()

    def _draw_wireframe(self, projected, w_coords, min_w, max_w, visibility, rotated):
        glDisable(GL_LIGHTING)
        glLineWidth(self.line_width)
        glBegin(GL_LINES)
        for edge in self.edges:
            i, j = edge
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value and rotated[j, 3] > self.w_slice_value: continue
            
            c1 = self.get_color(w_coords[i], min_w, max_w)
            c2 = self.get_color(w_coords[j], min_w, max_w)
            if self.selected_vertex >= 0 and (i == self.selected_vertex or j == self.selected_vertex):
                c1 = (1, 1, 0, 1); c2 = (1, 1, 0, 1)
                
            glColor4f(*c1); glVertex3fv(projected[i])
            glColor4f(*c2); glVertex3fv(projected[j])
        glEnd()

    def _draw_vertices(self, projected, w_coords, min_w, max_w, visibility, rotated):
        glDisable(GL_LIGHTING)
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for i, v in enumerate(projected):
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value: continue
            c = self.get_color(w_coords[i], min_w, max_w)
            if i == self.selected_vertex:
                glColor4f(1, 1, 0, 1); glPointSize(12.0)
            else:
                glColor4f(min(1, c[0]*1.4), min(1, c[1]*1.4), min(1, c[2]*1.4), 1.0); glPointSize(6.0)
            glVertex3fv(v)
        glEnd()

    def _draw_points_only(self, projected, w_coords, min_w, max_w, visibility, rotated):
        glDisable(GL_LIGHTING)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        for i, v in enumerate(projected):
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value: continue
            c = self.get_color(w_coords[i], min_w, max_w)
            if i == self.selected_vertex: glColor4f(1, 1, 0, 1)
            else: glColor4f(min(1, c[0]*1.4), min(1, c[1]*1.4), min(1, c[2]*1.4), 1.0)
            glVertex3fv(v)
        glEnd()

    def _draw_glow(self, projected, w_coords, min_w, max_w, visibility, rotated):
        glDisable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(20.0)
        glBegin(GL_POINTS)
        for i, v in enumerate(projected):
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value: continue
            c = self.get_color(w_coords[i], min_w, max_w)
            glColor4f(c[0]*0.3, c[1]*0.3, c[2]*0.3, 0.3)
            glVertex3fv(v)
        glEnd()
        glPointSize(12.0)
        glBegin(GL_POINTS)
        for i, v in enumerate(projected):
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value: continue
            c = self.get_color(w_coords[i], min_w, max_w)
            glColor4f(c[0]*0.5, c[1]*0.5, c[2]*0.5, 0.4)
            glVertex3fv(v)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _draw_true_slice(self, projected, rotated):
        c = self.w_slice_value
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor4f(0, 1, 1, 0.9)
        for face in self.faces:
            pts_3d = []
            for i in range(len(face)):
                idx1 = face[i]
                idx2 = face[(i+1)%len(face)]
                w1 = rotated[idx1, 3]
                w2 = rotated[idx2, 3]
                if (w1 <= c < w2) or (w2 <= c < w1):
                    t = (c - w1) / (w2 - w1) if (w2-w1)!=0 else 0
                    p = projected[idx1] * (1-t) + projected[idx2] * t
                    pts_3d.append(p)
            if len(pts_3d) == 2:
                glVertex3fv(pts_3d[0])
                glVertex3fv(pts_3d[1])
        glEnd()
        
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor4f(1, 1, 0, 1)
        for face in self.faces:
            pts_3d = []
            for i in range(len(face)):
                idx1 = face[i]
                idx2 = face[(i+1)%len(face)]
                w1 = rotated[idx1, 3]
                w2 = rotated[idx2, 3]
                if (w1 <= c < w2) or (w2 <= c < w1):
                    t = (c - w1) / (w2 - w1) if (w2-w1)!=0 else 0
                    p = projected[idx1] * (1-t) + projected[idx2] * t
                    pts_3d.append(p)
            for p in pts_3d:
                glVertex3fv(p)
        glEnd()

    def _draw_slice_points(self, projected, rotated):
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor4f(1, 1, 0, 1)
        for edge in self.edges:
            i, j = edge
            w1, w2 = rotated[i, 3], rotated[j, 3]
            if (w1 <= self.w_slice_value < w2) or (w2 <= self.w_slice_value < w1):
                t = (self.w_slice_value - w1) / (w2 - w1) if (w2-w1) != 0 else 0
                p = projected[i] * (1-t) + projected[j] * t
                glVertex3fv(p)
        glEnd()

    def _draw_vertex_labels(self, projected, w_coords, min_w, max_w, visibility, rotated):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(255, 255, 0))
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        for i, v in enumerate(projected):
            if self.w_slice_enabled and not self.show_true_slice:
                if rotated[i, 3] > self.w_slice_value: continue
            win_x, win_y, win_z = gluProject(v[0], v[1], v[2], modelview, projection, viewport)
            if win_z < 1.0:
                painter.drawText(int(win_x) + 8, int(self.height() - win_y) - 8, str(i))
        painter.end()

    def draw_overlay(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.show_fps:
            painter.setPen(QColor(0, 255, 100))
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            painter.drawText(10, 20, f"FPS: {self.fps:.1f}")
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(10, 38, f"Shape: {self.shape_name}")
            painter.drawText(10, 52, f"V:{len(self.vertices)}  E:{len(self.edges)}  F:{len(self.faces)}")
            painter.drawText(10, 66, f"Zoom: {self.zoom:.1f}  Pan: {self.cam_pan[0]:.1f}, {self.cam_pan[1]:.1f}")
            
            if self.use_quaternion_spin:
                painter.setPen(QColor(255, 165, 0))
                painter.drawText(10, 80, "4D ROT: DUAL QUATERNION ENGINE")
                painter.drawText(10, 94, f"qL: [{self.q4d_L[0]:.2f}, {self.q4d_L[1]:.2f}, {self.q4d_L[2]:.2f}, {self.q4d_L[3]:.2f}]")
                painter.drawText(10, 108, f"qR: [{self.q4d_R[0]:.2f}, {self.q4d_R[1]:.2f}, {self.q4d_R[2]:.2f}, {self.q4d_R[3]:.2f}]")
            else:
                painter.setPen(QColor(180, 180, 180))
                painter.drawText(10, 80, "4D ROT: 6-PLANE EULER")

            if self.w_slice_enabled:
                painter.setPen(QColor(0, 255, 255))
                mode = "TRUE 3D WIREFRAME" if self.show_true_slice else "EDGE POINTS"
                painter.drawText(10, 122, f"W-Slice: {self.w_slice_value:.2f} ({mode})")
            
            if self.selected_vertex >= 0 and self.selected_vertex < len(self.vertices):
                v = self.vertices[self.selected_vertex]
                painter.setPen(QColor(255, 255, 0))
                painter.setFont(QFont("Consolas", 9, QFont.Bold))
                painter.drawText(10, 136, f"V[{self.selected_vertex}]: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}, {v[3]:.2f})")

        if self.show_legend and len(self.vertices) > 0:
            self.draw_color_legend(painter)

        if self.view_mode == 1:
            painter.setPen(QColor(200, 200, 200, 150))
            painter.setFont(QFont("Arial", 9))
            w, h = self.width(), self.height()
            hw, hh = w // 2, h // 2
            painter.drawText(10, hh - 5, "Perspective")
            painter.drawText(hw + 10, hh - 5, "Top (Y-axis)")
            painter.drawText(10, h - 5, "Front (Z-axis)")
            painter.drawText(hw + 10, h - 5, "Side (X-axis)")

        painter.setPen(QColor(100, 100, 100, 180))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(self.width() - 340, self.height() - 10,
                        "R:Reset | S:Save | +/-:Zoom | Space:Anim | 1-4:View | F:Fog | C:Slice | Esc:Quit")
        painter.end()

    def draw_color_legend(self, painter):
        legend_x = self.width() - 180
        legend_y = 90
        legend_w = 30
        legend_h = 160
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(legend_x - 5, legend_y - 10, "W-axis")
        for i in range(legend_h):
            t = 1.0 - i / legend_h
            c = self.get_color(t, 0.0, 1.0)
            painter.fillRect(legend_x, legend_y + i, legend_w, 1, QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255)))
        painter.setPen(QColor(150, 150, 150))
        painter.drawRect(legend_x, legend_y, legend_w, legend_h)
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(legend_x + legend_w + 5, legend_y + 8, "Max W")
        painter.drawText(legend_x + legend_w + 5, legend_y + legend_h, "Min W")
        if self.w_slice_enabled:
            rotated = self.vertices.copy()
            if self.use_quaternion_spin:
                M = get_4d_rot_matrix(self.q4d_L, self.q4d_R)
                rotated = rotated @ M.T
            else:
                for i, angle in enumerate(self.rotation_angles):
                    rotated = self.rotate_4d_euler(rotated, angle, i)
            if self.normalize_4d:
                norms = np.linalg.norm(rotated, axis=1, keepdims=True)
                norms[norms == 0] = 1
                rotated = rotated / norms
            w_coords = rotated[:, 3]
            min_w, max_w = np.min(w_coords), np.max(w_coords)
            if max_w - min_w > 0.001:
                t = (self.w_slice_value - min_w) / (max_w - min_w)
                slice_y = legend_y + int((1 - t) * legend_h)
                painter.setPen(QPen(QColor(0, 255, 255), 2))
                painter.drawLine(legend_x - 5, slice_y, legend_x + legend_w + 5, slice_y)
                painter.setPen(QColor(0, 255, 255))
                painter.drawText(legend_x + legend_w + 5, slice_y + 4, f"W={self.w_slice_value:.2f}")

    def draw_axes(self):
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(0.8, 0.2, 0.2, 0.6); glVertex3f(-4, 0, 0); glVertex3f(4, 0, 0)
        glColor4f(0.2, 0.8, 0.2, 0.6); glVertex3f(0, -4, 0); glVertex3f(0, 4, 0)
        glColor4f(0.2, 0.2, 0.8, 0.6); glVertex3f(0, 0, -4); glVertex3f(0, 0, 4)
        glEnd()

    def save_image(self, filename):
        self.grabFramebuffer().save(filename)

    def export_obj(self, filename):
        if len(self.vertices) == 0: return
        if self.use_quaternion_spin:
            M = get_4d_rot_matrix(self.q4d_L, self.q4d_R)
            rotated = self.vertices @ M.T
        else:
            rotated = self.vertices.copy()
            for i, angle in enumerate(self.rotation_angles):
                rotated = self.rotate_4d_euler(rotated, angle, i)
        projected, w_coords = self.project_4d_to_3d(rotated)
        projected *= self.scale
        with open(filename, 'w') as f:
            f.write(f"# 4D Shape: {self.shape_name}\n# Exported from 4D Visualization Suite\n\n")
            for v in projected: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for edge in self.edges: f.write(f"l {edge[0]+1} {edge[1]+1}\n")
            if self.faces:
                f.write("\n")
                for face in self.faces:
                    indices = " ".join(str(i+1) for i in face)
                    f.write(f"f {indices}\n")

    def _auto_rotate_step(self):
        if self.auto_rotate:
            q_rot = axis_angle_to_quat([0, 1, 0], 0.02)
            self.cam_quat = quat_mult(q_rot, self.cam_quat)
        
        if self.w_slice_enabled and self.w_slice_auto_sweep:
            new_val = self.w_slice_value + (self.w_slice_sweep_dir * 0.05)
            if new_val > 2.0:
                new_val = 2.0
                self.w_slice_sweep_dir = -2
            elif new_val < -2.0:
                new_val = -2.0
                self.w_slice_sweep_dir = 2
            self.w_slice_value = new_val
            if hasattr(self, 'parent_widget'): 
                self.parent_widget.update_slice_ui(new_val)

        if self.use_quaternion_spin:
            qL_rot = axis_angle_to_quat(self.q4d_L_input[1:], self.q4d_L_speed * 0.02)
            qR_rot = axis_angle_to_quat(self.q4d_R_input[1:], self.q4d_R_speed * 0.02)
            self.q4d_L = quat_mult(qL_rot, self.q4d_L)
            self.q4d_R = quat_mult(qR_rot, self.q4d_R)

        self.update()

    def get_arcball_vector(self, x, y):
        w, h = self.width(), self.height()
        v = np.array([
            (2.0 * x - w) / w,
            (h - 2.0 * y) / h,
            0.0
        ])
        v_norm = np.linalg.norm(v)
        if v_norm > 1.0:
            v = v / v_norm
        else:
            v[2] = np.sqrt(1.0 - v_norm**2)
        return v

    def pick_vertex(self, x, y):
        if len(self.vertices) == 0: return -1
        if self.use_quaternion_spin:
            M = get_4d_rot_matrix(self.q4d_L, self.q4d_R)
            rotated = self.vertices @ M.T
        else:
            rotated = self.vertices.copy()
            for i, angle in enumerate(self.rotation_angles):
                rotated = self.rotate_4d_euler(rotated, angle, i)
        projected, w_coords = self.project_4d_to_3d(rotated)
        projected *= self.scale
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        min_dist = float('inf')
        closest_idx = -1
        for i, v in enumerate(projected):
            win_x, win_y, win_z = gluProject(v[0], v[1], v[2], modelview, projection, viewport)
            screen_y = self.height() - win_y
            dist = math.sqrt((win_x - x)**2 + (screen_y - y)**2)
            if dist < 15 and dist < min_dist and win_z < 1.0:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            idx = self.pick_vertex(event.position().x(), event.position().y())
            if idx >= 0:
                self.selected_vertex = idx if idx != self.selected_vertex else -1
                self.update()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            if event.buttons() & Qt.LeftButton:
                va = self.get_arcball_vector(self.last_mouse_pos.x(), self.last_mouse_pos.y())
                vb = self.get_arcball_vector(event.position().x(), event.position().y())
                angle = np.arccos(min(1.0, np.dot(va, vb)))
                axis = np.cross(va, vb)
                if np.linalg.norm(axis) > 0.001:
                    q_rot = axis_angle_to_quat(axis, angle * 2.0)
                    self.cam_quat = quat_mult(q_rot, self.cam_quat)
            elif event.buttons() & Qt.RightButton:
                dy = event.position().y() - self.last_mouse_pos.y()
                self.zoom += dy * 0.05
                self.zoom = max(2.0, min(20.0, self.zoom))
            elif event.buttons() & Qt.MiddleButton:
                dx = event.position().x() - self.last_mouse_pos.x()
                dy = event.position().y() - self.last_mouse_pos.y()
                self.cam_pan[0] += dx * 0.01
                self.cam_pan[1] -= dy * 0.01
            self.update()
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event): self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom -= delta * 0.005
        self.zoom = max(2.0, min(20.0, self.zoom))
        self.update()


# ═══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4D Visualization Suite v5.0 - Holographic & Hyperspherical")
        self.setMinimumSize(1400, 900)
        self.animation_active = [False] * 6
        self.animation_speed = 1

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.tabs = QTabWidget()
        self.tabs.setMaximumWidth(460)
        
        self.renderer = FourDRenderer()
        self.renderer.parent_widget = self
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.renderer)
        splitter.addWidget(self.tabs)
        splitter.setSizes([1000, 460])
        layout.addWidget(splitter)

        self.tabs.addTab(self._create_main_tab(), "Main")
        self.tabs.addTab(self._create_rotation_tab(), "Rotation & Slice")
        self.tabs.addTab(self._create_display_tab(), "Display & FX")
        self.tabs.addTab(self._create_info_tab(), "Info")

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.setInterval(16)
        self.load_shape("Clifford Torus")
        self.setFocusPolicy(Qt.StrongFocus)

    def _create_main_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("4D Visualization Controls")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00BFFF; padding: 8px; background: rgba(0,50,100,0.3); border-radius: 5px;")
        layout.addWidget(title)

        shape_group = QGroupBox("Shape Selection")
        shape_layout = QVBoxLayout(shape_group)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems([
            "Tesseract", "5-cell", "16-cell", "24-cell", "600-cell", "Clifford Torus",
            "Runcinated 5-cell", "Rectified Tesseract", "Truncated Tesseract",
            "3-3 Duoprism", "4-3 Duoprism", "5-5 Duoprism", "Dodecahedral Subset"
        ])
        self.shape_combo.setCurrentText("Clifford Torus")
        self.shape_combo.currentTextChanged.connect(self.load_shape)
        shape_layout.addWidget(self.shape_combo)
        self.shape_info = QLabel()
        self.shape_info.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        self.shape_info.setWordWrap(True)
        shape_layout.addWidget(self.shape_info)
        layout.addWidget(shape_group)

        view_group = QGroupBox("View Mode")
        view_layout = QVBoxLayout(view_group)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Single View", "Quad View", "Stereo Anaglyph", "VR Side-by-Side"])
        self.view_combo.currentIndexChanged.connect(self.renderer.set_view_mode)
        view_layout.addWidget(self.view_combo)
        
        self.auto_rot_cb = QCheckBox("Auto-Rotate 3D Camera")
        self.auto_rot_cb.toggled.connect(self.renderer.set_auto_rotate)
        view_layout.addWidget(self.auto_rot_cb)
        layout.addWidget(view_group)

        render_group = QGroupBox("Render Settings")
        render_layout = QVBoxLayout(render_group)
        for label_text, combo_items, handler in [
            ("Mode:", ["Wireframe", "Solid", "Points Only", "Hybrid (Solid+Wire)", "Hologram"], self.renderer.set_render_mode),
            ("Projection:", ["Perspective", "Stereographic"], self.renderer.set_projection_mode),
            ("Color Scheme:", ["Blue-Red", "Rainbow", "Cyan-Magenta", "Ice", "Flat Colors"], self.renderer.set_color_mode),
            ("Background:", ["Dark Blue", "Black", "White", "Dark Gray"], self.change_bg_color),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            combo = QComboBox()
            combo.addItems(combo_items)
            combo.currentIndexChanged.connect(handler)
            row.addWidget(combo)
            render_layout.addLayout(row)
        layout.addWidget(render_group)

        action_group = QGroupBox("Quick Actions")
        action_layout = QVBoxLayout(action_group)
        btn_row1 = QHBoxLayout()
        reset_btn = QPushButton("🔄 Reset All")
        reset_btn.setStyleSheet("background: #c0392b; color: white; padding: 6px; font-weight: bold;")
        reset_btn.clicked.connect(self.reset_all)
        btn_row1.addWidget(reset_btn)
        
        random_btn = QPushButton("🎲 Random Euler")
        random_btn.setStyleSheet("background: #8e44ad; color: white; padding: 6px; font-weight: bold;")
        random_btn.clicked.connect(self.random_rotation)
        btn_row1.addWidget(random_btn)
        action_layout.addLayout(btn_row1)
        
        btn_row2 = QHBoxLayout()
        save_btn = QPushButton("💾 Save PNG")
        save_btn.setStyleSheet("background: #27ae60; color: white; padding: 6px; font-weight: bold;")
        save_btn.clicked.connect(self.save_image)
        btn_row2.addWidget(save_btn)
        
        obj_btn = QPushButton("📐 Export OBJ")
        obj_btn.setStyleSheet("background: #2980b9; color: white; padding: 6px; font-weight: bold;")
        obj_btn.clicked.connect(self.export_obj)
        btn_row2.addWidget(obj_btn)
        action_layout.addLayout(btn_row2)
        layout.addWidget(action_group)
        layout.addStretch()
        return panel

    def _create_rotation_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # QUATERNION ENGINE GROUP
        quat_group = QGroupBox("4D Dual Quaternion Engine")
        quat_layout = QVBoxLayout(quat_group)
        
        self.quat_spin_cb = QCheckBox("Enable Quaternion Spin (Disables 6-Plane Sliders)")
        self.quat_spin_cb.setStyleSheet("font-weight: bold; color: #00BFFF;")
        self.quat_spin_cb.toggled.connect(self.renderer.set_quaternion_spin_enabled)
        self.quat_spin_cb.toggled.connect(self.toggle_quaternion_ui)
        quat_layout.addWidget(self.quat_spin_cb)
        
        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Presets:")
        preset_row.addWidget(preset_lbl)
        preset_combo = QComboBox()
        preset_combo.addItems(["— Select —", "Clifford Spin", "Lissajous Spin", "Dual Axis Spin"])
        preset_combo.currentIndexChanged.connect(self.apply_quat_preset)
        preset_row.addWidget(preset_combo)
        quat_layout.addLayout(preset_row)
        
        # Left Q
        lq_lbl = QLabel("Left Isoclinic Quaternion (qL):")
        quat_layout.addWidget(lq_lbl)
        self.lq_sliders = []
        for i, axis in enumerate(["X", "Y", "Z"]):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Axis {axis}:"))
            s = QSlider(Qt.Horizontal)
            s.setRange(-100, 100)
            s.setValue(100 if i==0 else 0)
            s.valueChanged.connect(lambda v, idx=i: self.renderer.set_quaternion_input(True, idx, v))
            row.addWidget(s)
            self.lq_sliders.append(s)
            quat_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Speed:"))
        self.lq_speed = QSlider(Qt.Horizontal)
        self.lq_speed.setRange(-100, 100)
        self.lq_speed.setValue(20)
        self.lq_speed.valueChanged.connect(lambda v: self.renderer.set_quaternion_speed(True, v))
        row.addWidget(self.lq_speed)
        quat_layout.addLayout(row)

        # Right Q
        rq_lbl = QLabel("Right Isoclinic Quaternion (qR):")
        quat_layout.addWidget(rq_lbl)
        self.rq_sliders = []
        for i, axis in enumerate(["X", "Y", "Z"]):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Axis {axis}:"))
            s = QSlider(Qt.Horizontal)
            s.setRange(-100, 100)
            s.setValue(100 if i==1 else 0)
            s.valueChanged.connect(lambda v, idx=i: self.renderer.set_quaternion_input(False, idx, v))
            row.addWidget(s)
            self.rq_sliders.append(s)
            quat_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Speed:"))
        self.rq_speed = QSlider(Qt.Horizontal)
        self.rq_speed.setRange(-100, 100)
        self.rq_speed.setValue(30)
        self.rq_speed.valueChanged.connect(lambda v: self.renderer.set_quaternion_speed(False, v))
        row.addWidget(self.rq_speed)
        quat_layout.addLayout(row)
        
        layout.addWidget(quat_group)

        # EULER ROTATION GROUP
        rot_group = QGroupBox("4D Euler Rotation Planes")
        rot_layout = QVBoxLayout(rot_group)
        self.rotation_sliders = []
        self.rotation_value_labels = []
        self.anim_cbs = []
        names = ["XY", "XZ", "XW", "YZ", "YW", "ZW"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        for i, name in enumerate(names):
            row = QHBoxLayout()
            lbl = QLabel(f"{name}:")
            lbl.setStyleSheet(f"color: {colors[i]}; font-weight: bold; min-width: 30px;")
            row.addWidget(lbl)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 360)
            slider.valueChanged.connect(lambda v, idx=i: self.update_rotation(idx, v))
            row.addWidget(slider)
            self.rotation_sliders.append(slider)
            val = QLabel("0°"); val.setMinimumWidth(35)
            row.addWidget(val)
            self.rotation_value_labels.append(val)
            cb = QCheckBox("▶")
            cb.setToolTip(f"Animate {name} rotation")
            cb.toggled.connect(lambda c, idx=i: self.toggle_anim(idx, c))
            row.addWidget(cb)
            self.anim_cbs.append(cb)
            rot_layout.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Anim Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.update_anim_speed)
        speed_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("1x")
        speed_row.addWidget(self.speed_label)
        rot_layout.addLayout(speed_row)
        
        toggle_all_btn = QPushButton("▶ Toggle All Animations")
        toggle_all_btn.setStyleSheet("background: #2c3e50; color: white; padding: 6px;")
        toggle_all_btn.clicked.connect(self.toggle_all_animations)
        rot_layout.addWidget(toggle_all_btn)
        layout.addWidget(rot_group)

        slice_group = QGroupBox("W-Slice (3D Cross-section)")
        slice_layout = QVBoxLayout(slice_group)
        self.slice_cb = QCheckBox("Enable W-slice")
        self.slice_cb.toggled.connect(self.toggle_w_slice)
        slice_layout.addWidget(self.slice_cb)
        
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("W value:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(-200, 200)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.update_w_slice_value)
        sl_row.addWidget(self.slice_slider)
        self.slice_lbl = QLabel("0.00")
        sl_row.addWidget(self.slice_lbl)
        slice_layout.addLayout(sl_row)
        
        self.true_slice_cb = QCheckBox("True 3D Wireframe Slice (Exact Math)")
        self.true_slice_cb.setChecked(True)
        self.true_slice_cb.toggled.connect(self.renderer.set_show_true_slice)
        self.true_slice_cb.toggled.connect(self.renderer.update)
        slice_layout.addWidget(self.true_slice_cb)
        
        self.sweep_cb = QCheckBox("Auto-Sweep Slice")
        self.sweep_cb.toggled.connect(self.renderer.set_w_sweep)
        slice_layout.addWidget(self.sweep_cb)
        layout.addWidget(slice_group)
        layout.addStretch()
        return panel

    def _create_display_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        app_group = QGroupBox("Appearance")
        app_layout = QVBoxLayout(app_group)
        for label_text, slider_min, slider_max, default, divisor, fmt, handler in [
            ("Dist:", 20, 100, 30, 10.0, "{:.1f}", self.update_dist),
            ("Scale:", 20, 200, 100, 100.0, "{:.2f}", self.update_scale),
            ("Line W:", 1, 10, 2, 1.0, "{:.1f}", self.update_line_width),
            ("Opacity:", 10, 100, 75, 100.0, "{:.2f}", self.update_opacity),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(slider_min, slider_max)
            slider.setValue(default)
            slider.valueChanged.connect(handler)
            row.addWidget(slider)
            lbl = QLabel(fmt.format(default / divisor))
            row.addWidget(lbl)
            if label_text == "Dist:": self.dist_slider = slider; self.dist_lbl = lbl
            elif label_text == "Scale:": self.scale_slider = slider; self.scale_lbl = lbl
            elif label_text == "Line W:": self.lw_slider = slider; self.lw_lbl = lbl
            elif label_text == "Opacity:": self.op_slider = slider; self.op_lbl = lbl
            app_layout.addLayout(row)
        layout.addWidget(app_group)

        disp_group = QGroupBox("Display & Visual Effects")
        disp_layout = QVBoxLayout(disp_group)
        for text, attr, default in [
            ("Show Vertices", 'show_vertices', True),
            ("Show Axes", 'show_axes', True),
            ("Show Color Legend", 'show_legend', True),
            ("Show FPS/Stats", 'show_fps', True),
            ("Show Vertex Labels", 'show_labels', False),
            ("Show Ground Grid", 'show_grid', False),
            ("Show Shadow", 'show_shadow', False),
            ("Show Vertex Glow", 'show_glow', False),
            ("Atmospheric Fog", 'show_fog', True),
            ("Smooth Shading (Gouraud)", 'use_smooth_shading', True),
            ("4D Motion Trails", 'use_trails', False),
            ("Normalize 4D Radius (3-Sphere)", 'normalize_4d', False),
        ]:
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.toggled.connect(lambda c, a=attr: setattr(self.renderer, a, c))
            cb.toggled.connect(self.renderer.update)
            disp_layout.addWidget(cb)
            if attr == 'show_vertices': self.vtx_cb = cb
            elif attr == 'show_axes': self.axes_cb = cb
            elif attr == 'show_legend': self.legend_cb = cb
            elif attr == 'show_fps': self.fps_cb = cb
            elif attr == 'show_labels': self.labels_cb = cb
            elif attr == 'show_grid': self.grid_cb = cb
            elif attr == 'show_shadow': self.shadow_cb = cb
            elif attr == 'show_glow': self.glow_cb = cb
            elif attr == 'show_fog': self.fog_cb = cb
            elif attr == 'use_smooth_shading': self.smooth_cb = cb
            elif attr == 'use_trails': self.trails_cb = cb
            elif attr == 'normalize_4d': self.norm_cb = cb
        layout.addWidget(disp_group)
        layout.addStretch()
        return panel

    def _create_info_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        info_group = QGroupBox("Shape Information")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("background: #1a1a1a; color: #cccccc; font-family: Consolas;")
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)
        
        help_group = QGroupBox("Controls & Math Info")
        help_layout = QVBoxLayout(help_group)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(320)
        help_text.setStyleSheet("background: #1a1a1a; color: #aaaaaa;")
        help_text.setHtml("""
<b>Mouse Controls:</b><br>
• Left Drag: Arcball rotate 3D camera<br>
• Right Drag: Zoom<br>
• Middle Drag: Pan Camera (X/Y)<br>
• Ctrl+Click: Select vertex<br><br>

<b>New Visual Features:</b><br>
• <b>Hologram Mode:</b> Dark transparent solids with glowing additive wireframes.<br>
• <b>Smooth Shading:</b> Calculates per-vertex normals for smooth Gouraud lighting.<br>
• <b>4D Motion Trails:</b> Plots the 3D history of 4D vertices.<br>
• <b>Normalize 4D Radius:</b> Projects all vertices to a 3-Sphere (unit radius). Great for viewing topological structure.<br><br>

<b>Keyboard Shortcuts:</b><br>
• R: Reset all<br>
• S: Save image<br>
• +/-: Zoom in/out<br>
• 1/2/3/4: View modes<br>
• F: Toggle Fog<br>
• C: Toggle True Slice<br>
• Esc: Quit<br>
        """)
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)
        return panel

    def load_shape(self, name):
        methods = {
            "Tesseract": FourDShapes.tesseract, "5-cell": FourDShapes.simplex_4d,
            "16-cell": FourDShapes.cell_16, "24-cell": FourDShapes.cell_24,
            "600-cell": FourDShapes.cell_600, "Clifford Torus": FourDShapes.clifford_torus,
            "Runcinated 5-cell": FourDShapes.runcinated_5_cell,
            "Rectified Tesseract": FourDShapes.rectified_tesseract,
            "Truncated Tesseract": FourDShapes.truncated_tesseract,
            "3-3 Duoprism": FourDShapes.duoprism_3_3,
            "4-3 Duoprism": FourDShapes.duoprism_4_3,
            "5-5 Duoprism": FourDShapes.duoprism_5_5,
            "Dodecahedral Subset": FourDShapes.hyperdodecahedron
        }
        if name in methods:
            v, e, f, n = methods[name]()
            self.renderer.set_shape(v, e, f, n)
            self.shape_info.setText(f"● {n}\n   Vertices: {len(v)} | Edges: {len(e)} | Faces: {len(f)}")
            info = SHAPE_INFO.get(name, {})
            html = f"<h2 style='color:#00BFFF;'>{name}</h2>"
            if info:
                html += f"<p><b>Schläfli symbol:</b> {info.get('schlafli', '—')}</p>"
                html += f"<p><b>Cells:</b> {info.get('cells', '—')}</p>"
                html += f"<p><b>Vertices:</b> {info.get('vertices', '—')}</p>"
                html += f"<p><b>Edges:</b> {info.get('edges', '—')}</p>"
                html += f"<p><b>Faces:</b> {info.get('faces', '—')}</p>"
                html += f"<p><b>Symmetry group:</b> {info.get('symmetry', '—')}</p>"
                html += f"<hr><p>{info.get('description', '')}</p>"
            self.info_text.setHtml(html)

    def update_rotation(self, idx, val):
        self.renderer.set_rotation(idx, val)
        self.rotation_value_labels[idx].setText(f"{val}°")

    def toggle_quaternion_ui(self, checked):
        for s in self.rotation_sliders: s.setEnabled(not checked)
        for c in self.anim_cbs: c.setEnabled(not checked)

    def apply_quat_preset(self, idx):
        presets = [
            None,
            # Clifford Spin
            {"lq_axis": [1, 0, 0], "lq_spd": 20, "rq_axis": [1, 0, 0], "rq_spd": -20},
            # Lissajous Spin
            {"lq_axis": [1, 1, 0], "lq_spd": 30, "rq_axis": [1, -1, 0], "rq_spd": 20},
            # Dual Axis Spin
            {"lq_axis": [1, 0, 0], "lq_spd": 40, "rq_axis": [0, 1, 0], "rq_spd": 25},
        ]
        if 0 < idx < len(presets) and presets[idx]:
            p = presets[idx]
            self.lq_sliders[0].setValue(int(p["lq_axis"][0]*100))
            self.lq_sliders[1].setValue(int(p["lq_axis"][1]*100))
            self.lq_sliders[2].setValue(int(p["lq_axis"][2]*100))
            self.lq_speed.setValue(p["lq_spd"])
            
            self.rq_sliders[0].setValue(int(p["rq_axis"][0]*100))
            self.rq_sliders[1].setValue(int(p["rq_axis"][1]*100))
            self.rq_sliders[2].setValue(int(p["rq_axis"][2]*100))
            self.rq_speed.setValue(p["rq_spd"])

    def update_dist(self, val):
        d = val / 10.0; self.renderer.set_projection_distance(d); self.dist_lbl.setText(f"{d:.1f}")

    def update_scale(self, val):
        s = val / 100.0; self.renderer.set_scale(s); self.scale_lbl.setText(f"{s:.2f}")

    def update_line_width(self, val):
        self.renderer.set_line_width(float(val)); self.lw_lbl.setText(f"{val:.1f}")

    def update_opacity(self, val):
        op = val / 100.0; self.renderer.set_face_opacity(op); self.op_lbl.setText(f"{op:.2f}")

    def update_anim_speed(self, val):
        self.animation_speed = val; self.speed_label.setText(f"{val}x")

    def toggle_w_slice(self, checked):
        self.renderer.set_w_slice(checked, self.slice_slider.value() / 100.0)

    def update_w_slice_value(self, val):
        w = val / 100.0; self.slice_lbl.setText(f"{w:.2f}")
        if self.slice_cb.isChecked():
            self.renderer.set_w_slice(True, w)

    def update_slice_ui(self, val):
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(int(val * 100))
        self.slice_lbl.setText(f"{val:.2f}")
        self.slice_slider.blockSignals(False)

    def change_bg_color(self, idx):
        colors = [[0.02,0.02,0.08], [0.0,0.0,0.0], [0.95,0.95,0.95], [0.15,0.15,0.15]]
        self.renderer.set_bg_color(*colors[idx])

    def toggle_anim(self, idx, checked):
        self.animation_active[idx] = checked
        if any(self.animation_active): self.animation_timer.start()
        else: self.animation_timer.stop()

    def toggle_all_animations(self):
        all_active = all(self.animation_active)
        for i in range(6):
            self.anim_cbs[i].setChecked(not all_active)

    def animate(self):
        for i, a in enumerate(self.animation_active):
            if a:
                self.rotation_sliders[i].setValue(
                    (self.rotation_sliders[i].value() + self.animation_speed) % 361)

    def reset_all(self):
        for slider in self.rotation_sliders: slider.setValue(0)
        for cb in self.anim_cbs: cb.setChecked(False)
        self.quat_spin_cb.setChecked(False)
        self.renderer.reset_view()
        self.dist_slider.setValue(30)
        self.scale_slider.setValue(100)

    def random_rotation(self):
        import random
        for slider in self.rotation_sliders:
            slider.setValue(random.randint(0, 360))

    def save_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "4d_render.png",
            "PNG Images (*.png);;JPEG Images (*.jpg);;BMP Images (*.bmp)")
        if filename:
            self.renderer.save_image(filename)

    def export_obj(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export OBJ", "4d_shape.obj",
            "OBJ Files (*.obj)")
        if filename:
            self.renderer.export_obj(filename)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_R: self.reset_all()
        elif key == Qt.Key_S: self.save_image()
        elif key == Qt.Key_Escape: self.close()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.renderer.zoom = max(2.0, self.renderer.zoom - 0.5); self.renderer.update()
        elif key == Qt.Key_Minus:
            self.renderer.zoom = min(20.0, self.renderer.zoom + 0.5); self.renderer.update()
        elif key == Qt.Key_1: self.renderer.set_view_mode(0)
        elif key == Qt.Key_2: self.renderer.set_view_mode(1)
        elif key == Qt.Key_3: self.renderer.set_view_mode(2)
        elif key == Qt.Key_4: self.renderer.set_view_mode(3)
        elif key == Qt.Key_F: self.fog_cb.toggle()
        elif key == Qt.Key_C: self.true_slice_cb.toggle()
        else: super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSamples(4)  # MSAA anti-aliasing
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()