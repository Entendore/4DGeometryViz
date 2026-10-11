import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QComboBox, QPushButton, QGroupBox, QCheckBox,
    QSplitter
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import *
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class FourDShapes:
    """Define 4D shapes with vertices, edges, and faces"""

    @staticmethod
    def tesseract():
        """4D Hypercube"""
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    for w in [-1, 1]:
                        vertices.append([x, y, z, w])
        vertices = np.array(vertices, dtype=np.float64)

        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                diff = np.sum(vertices[i] != vertices[j])
                if diff == 1:
                    edges.append([i, j])

        faces = []
        coords = [0, 1, 2, 3]
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                fixed_dims = [i for i in coords if i not in [d1, d2]]
                for f1 in [-1, 1]:
                    for f2 in [-1, 1]:
                        face_verts = []
                        for v1 in [-1, 1]:
                            for v2 in [-1, 1]:
                                v = [0, 0, 0, 0]
                                v[d1] = v1
                                v[d2] = v2
                                v[fixed_dims[0]] = f1
                                v[fixed_dims[1]] = f2
                                idx = np.where((vertices == v).all(axis=1))[0][0]
                                face_verts.append(idx)
                        faces.append(face_verts)

        return vertices, edges, faces, "Tesseract"

    @staticmethod
    def simplex_4d():
        """4D Simplex"""
        vertices = np.array([
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
            [-1, -1, 1, -1],
            [0, 0, 0, 1]
        ], dtype=np.float64) * 0.6

        edges = [[i, j] for i in range(5) for j in range(i + 1, 5)]
        faces = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    faces.append([i, j, k])
        return vertices, edges, faces, "4D Simplex"

    @staticmethod
    def cell_16():
        """16-cell"""
        vertices = []
        for i in range(4):
            v_pos = [0, 0, 0, 0]
            v_pos[i] = 1
            vertices.append(v_pos)
            v_neg = [0, 0, 0, 0]
            v_neg[i] = -1
            vertices.append(v_neg)
        vertices = np.array(vertices, dtype=np.float64)

        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if not np.allclose(vertices[i], -vertices[j]):
                    edges.append([i, j])

        faces = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.allclose(vertices[j], -vertices[i]): continue
                for k in range(j + 1, 8):
                    if np.allclose(vertices[k], -vertices[i]): continue
                    if np.allclose(vertices[k], -vertices[j]): continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "16-cell"

    @staticmethod
    def cell_24():
        """24-cell"""
        vertices = []
        for i in range(4):
            for j in range(i + 1, 4):
                for c1 in [1, -1]:
                    for c2 in [1, -1]:
                        v = [0, 0, 0, 0]
                        v[i] = c1
                        v[j] = c2
                        vertices.append(v)
        vertices = np.array(vertices, dtype=np.float64)

        edges = []
        for i in range(24):
            for j in range(i + 1, 24):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - np.sqrt(2)) < 0.01:
                    edges.append([i, j])

        faces = []
        for i in range(24):
            for j in range(i + 1, 24):
                if abs(np.linalg.norm(vertices[i] - vertices[j]) - np.sqrt(2)) > 0.01: continue
                for k in range(j + 1, 24):
                    if abs(np.linalg.norm(vertices[i] - vertices[k]) - np.sqrt(2)) > 0.01: continue
                    if abs(np.linalg.norm(vertices[j] - vertices[k]) - np.sqrt(2)) > 0.01: continue
                    faces.append([i, j, k])
        return vertices, edges, faces, "24-cell"

    @staticmethod
    def duoprism_3_3():
        vertices = []
        for i in range(3):
            angle1 = 2 * np.pi * i / 3
            for j in range(3):
                angle2 = 2 * np.pi * j / 3
                vertices.append([
                    np.cos(angle1), np.sin(angle1),
                    np.cos(angle2), np.sin(angle2)
                ])
        vertices = np.array(vertices, dtype=np.float64)

        edges = []
        for i in range(9):
            for j in range(i + 1, 9):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if dist < 1.5:
                    edges.append([i, j])

        faces = [
            [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5],
            [0, 1, 7, 6], [1, 2, 8, 7], [2, 0, 6, 8]
        ]
        return vertices, edges, faces, "3-3 Duoprism"

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
        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if dist < 2.5:
                    edges.append([i, j])
        return vertices, edges, [], "Hyperdodecahedron (Subset)"


class FourDRenderer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = np.array([])
        self.edges = []
        self.faces = []
        self.shape_name = ""

        self.rotation_angles = [0.0] * 6
        self.projection_distance = 3.0
        self.scale = 1.0

        self.render_mode = 0
        self.projection_mode = 0

        self.w_slice_enabled = False
        self.w_slice_value = 0.0

        self.show_vertices = True
        self.show_axes = True
        self.color_mode = 0
        self.line_width = 2.0

        self.rotation_3d = [30.0, 45.0]
        self.zoom = 6.0
        self.last_mouse_pos = None
        self.setMinimumSize(600, 500)

    # ... (Setters omitted for brevity, same as previous) ...
    def set_shape(self, vertices, edges, faces, name):
        self.vertices = vertices.copy()
        self.edges = edges
        self.faces = faces
        self.shape_name = name
        self.update()

    def set_rotation(self, plane_index, angle):
        self.rotation_angles[plane_index] = angle
        self.update()

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

    def set_w_slice(self, enabled, value):
        self.w_slice_enabled = enabled
        self.w_slice_value = value
        self.update()

    def rotate_4d(self, vertices, angle, plane):
        if len(vertices) == 0: return vertices
        angle_rad = np.radians(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        matrices = [
            np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]]),
            np.array([[c, 0, 0, -s], [0, 1, 0, 0], [0, 0, 1, 0], [s, 0, 0, c]]),
            np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0], [0, c, 0, -s], [0, 0, 1, 0], [0, s, 0, c]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]])
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
                result.append([v[0] * factor, v[1] * factor, v[2] * factor])
            else:
                denom = 2.0 - w
                if abs(denom) < 0.01: denom = 0.01
                factor = 2.0 / denom
                result.append([v[0] * factor, v[1] * factor, v[2] * factor])
        return np.array(result), np.array(w_coords)

    def get_color(self, w, min_w, max_w):
        if max_w - min_w < 0.001: t = 0.5
        else: t = (w - min_w) / (max_w - min_w)
        
        if self.color_mode == 0:
            return (t, 0.2 + 0.6 * (1 - abs(2 * t - 1)), 1 - t, 0.9)
        elif self.color_mode == 1:
            h = t * 6; c = 0.9; x = c * (1 - abs(h % 2 - 1))
            if h < 1: r, g, b = c, x, 0
            elif h < 2: r, g, b = x, c, 0
            elif h < 3: r, g, b = 0, c, x
            elif h < 4: r, g, b = 0, x, c
            elif h < 5: r, g, b = x, 0, c
            else: r, g, b = c, 0, x
            return (r+0.1, g+0.1, b+0.1, 0.9)
        else:
            intensity = 0.5 + 0.5 * t
            if t < 0.5: return (0.2, intensity, intensity * 0.8, 0.9)
            else: return (intensity, 0.3, intensity * 0.6, 0.9)

    def initializeGL(self):
        glClearColor(0.02, 0.02, 0.08, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        
        # SOLID RENDERING SETUP
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE) # Handles scaling of normals
        
        # BACK-FACE CULLING
        # This tells OpenGL to ignore triangles that are facing away from the camera.
        # This creates the "Solid" effect.
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # RIGHT-HAND RULE DEFINITION
        # GL_CCW means: if vertices are drawn Counter-Clockwise, the face is Front-Facing.
        glFrontFace(GL_CCW) 

        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0, 0, -self.zoom)
        glRotatef(self.rotation_3d[0], 1, 0, 0)
        glRotatef(self.rotation_3d[1], 0, 1, 0)

        glDisable(GL_LIGHTING)
        if self.show_axes: self.draw_axes()

        if len(self.vertices) == 0: return

        # 4D Math
        rotated = self.vertices.copy()
        for i, angle in enumerate(self.rotation_angles):
            rotated = self.rotate_4d(rotated, angle, i)

        projected, w_coords = self.project_4d_to_3d(rotated)
        if len(projected) == 0: return
        projected *= self.scale
        
        min_w, max_w = (np.min(w_coords), np.max(w_coords)) if len(w_coords) > 0 else (0, 1)

        visibility = np.ones(len(projected), dtype=bool)
        if self.w_slice_enabled:
            visibility = rotated[:, 3] <= self.w_slice_value

        # SOLID FACES RENDERING
        if self.render_mode == 1 and self.faces:
            glEnable(GL_LIGHTING)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            
            for face in self.faces:
                face_indices = np.array(face)
                if not np.any(visibility[face_indices]): continue

                # Get projected vertices
                pts = [projected[idx] for idx in face_indices]
                
                # Calculate Geometric Normal (Right-Hand Rule check)
                # Cross product of (v1-v0) and (v2-v0)
                v0, v1, v2 = pts[0], pts[1], pts[2]
                normal = np.cross(v1 - v0, v2 - v0)
                
                # Calculate Face Center
                face_center = np.mean(pts, axis=0)

                # DYNAMIC ORIENTATION FIX
                # We want the normal to point OUTWARD from the object center (0,0,0).
                # If Normal . FaceCenter < 0, the normal points INWARD (wrong).
                # We must reverse the vertex order to flip the normal to point OUTWARD.
                if np.dot(normal, face_center) < 0:
                    draw_indices = face_indices[::-1] # Reverse order (CW -> CCW relative to outside)
                    normal = -normal # Flip normal for lighting calculation
                else:
                    draw_indices = face_indices # Keep order
                
                # Normalize
                norm_len = np.linalg.norm(normal)
                if norm_len > 0: normal /= norm_len

                glBegin(GL_POLYGON)
                glNormal3fv(normal) # Set the corrected normal
                
                for idx in draw_indices:
                    w = w_coords[idx]
                    c = self.get_color(w, min_w, max_w)
                    glColor4f(c[0], c[1], c[2], 0.85)
                    glVertex3fv(projected[idx])
                glEnd()
            
            glDisable(GL_POLYGON_OFFSET_FILL)

        # WIREFRAME RENDERING
        if self.render_mode == 0 or self.render_mode == 1:
            glDisable(GL_LIGHTING) # Lines don't need lighting
            glLineWidth(self.line_width)
            glBegin(GL_LINES)
            for edge in self.edges:
                i, j = edge
                if self.w_slice_enabled:
                     if rotated[i, 3] > self.w_slice_value and rotated[j, 3] > self.w_slice_value: continue
                
                c1 = self.get_color(w_coords[i], min_w, max_w)
                c2 = self.get_color(w_coords[j], min_w, max_w)
                
                glColor4f(*c1); glVertex3fv(projected[i])
                glColor4f(*c2); glVertex3fv(projected[j])
            glEnd()

        # VERTICES
        if self.show_vertices:
            glDisable(GL_LIGHTING)
            glPointSize(6.0)
            glBegin(GL_POINTS)
            for i, v in enumerate(projected):
                if self.w_slice_enabled and rotated[i, 3] > self.w_slice_value: continue
                c = self.get_color(w_coords[i], min_w, max_w)
                glColor4f(min(1, c[0]*1.4), min(1, c[1]*1.4), min(1, c[2]*1.4), 1.0)
                glVertex3fv(v)
            glEnd()
            
        # SLICE POINTS
        if self.w_slice_enabled:
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glColor4f(1, 1, 1, 1)
            for edge in self.edges:
                i, j = edge
                w1, w2 = rotated[i, 3], rotated[j, 3]
                if (w1 <= self.w_slice_value < w2) or (w2 <= self.w_slice_value < w1):
                    t = (self.w_slice_value - w1) / (w2 - w1) if (w2-w1)!=0 else 0
                    p = projected[i] * (1-t) + projected[j] * t
                    glVertex3fv(p)
            glEnd()

    def draw_axes(self):
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(0.8, 0.2, 0.2, 0.6); glVertex3f(-4, 0, 0); glVertex3f(4, 0, 0)
        glColor4f(0.2, 0.8, 0.2, 0.6); glVertex3f(0, -4, 0); glVertex3f(0, 4, 0)
        glColor4f(0.2, 0.2, 0.8, 0.6); glVertex3f(0, 0, -4); glVertex3f(0, 0, 4)
        glEnd()

    def mousePressEvent(self, event): self.last_mouse_pos = event.position()
    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            if event.buttons() & Qt.LeftButton:
                self.rotation_3d[1] += dx * 0.5
                self.rotation_3d[0] += dy * 0.5
            elif event.buttons() & Qt.RightButton:
                self.zoom += dy * 0.05
                self.zoom = max(2.0, min(20.0, self.zoom))
            self.update()
        self.last_mouse_pos = event.position()
    def mouseReleaseEvent(self, event): self.last_mouse_pos = None
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom -= delta * 0.005
        self.zoom = max(2.0, min(20.0, self.zoom))
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4D Visualization")
        self.setMinimumSize(1400, 900)
        self.animation_active = [False] * 6
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        self.renderer = FourDRenderer()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.renderer)
        splitter.addWidget(self.create_control_panel())
        layout.addWidget(splitter)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.setInterval(16)
        self.load_shape("Tesseract")

    def create_control_panel(self):
        panel = QWidget(); panel.setMinimumWidth(350); layout = QVBoxLayout(panel)
        title = QLabel("4D Controls"); title.setFont(QFont("Arial", 13, QFont.Bold)); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00BFFF; padding: 5px;"); layout.addWidget(title)

        # Shape Selection
        shape_group = QGroupBox("Shape"); shape_layout = QVBoxLayout(shape_group)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Tesseract", "4D Simplex", "16-cell", "24-cell", "3-3 Duoprism", "Hyperdodecahedron"])
        self.shape_combo.currentTextChanged.connect(self.load_shape)
        shape_layout.addWidget(self.shape_combo)
        self.shape_info = QLabel(); self.shape_info.setStyleSheet("color: #AAAAAA;")
        shape_layout.addWidget(self.shape_info); layout.addWidget(shape_group)

        # Render Settings
        render_group = QGroupBox("Render"); render_layout = QVBoxLayout(render_group)
        self.render_combo = QComboBox()
        self.render_combo.addItems(["Wireframe", "Solid", "Points Only"])
        self.render_combo.currentIndexChanged.connect(self.renderer.set_render_mode)
        render_layout.addWidget(self.render_combo)
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["Perspective", "Stereographic"])
        self.proj_combo.currentIndexChanged.connect(self.renderer.set_projection_mode)
        render_layout.addWidget(self.proj_combo)
        layout.addWidget(render_group)

        # Rotation
        rot_group = QGroupBox("4D Rotation"); rot_layout = QVBoxLayout(rot_group)
        self.rotation_sliders = []; self.rotation_value_labels = []; self.anim_cbs = []
        names = ["XY", "XZ", "XW", "YZ", "YW", "ZW"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        for i, name in enumerate(names):
            row = QHBoxLayout()
            lbl = QLabel(f"{name}:"); lbl.setStyleSheet(f"color: {colors[i]}; font-weight: bold;")
            row.addWidget(lbl)
            slider = QSlider(Qt.Horizontal); slider.setRange(0, 360)
            slider.valueChanged.connect(lambda v, idx=i: self.update_rotation(idx, v))
            row.addWidget(slider); self.rotation_sliders.append(slider)
            val = QLabel("0°"); val.setMinimumWidth(35); row.addWidget(val); self.rotation_value_labels.append(val)
            cb = QCheckBox("▶"); cb.toggled.connect(lambda c, idx=i: self.toggle_anim(idx, c))
            row.addWidget(cb); self.anim_cbs.append(cb)
            rot_layout.addLayout(row)
        layout.addWidget(rot_group)

        # Projection Settings
        proj_group = QGroupBox("Projection"); proj_layout = QVBoxLayout(proj_group)
        # Dist
        d_row = QHBoxLayout(); d_row.addWidget(QLabel("Dist:"))
        self.dist_slider = QSlider(Qt.Horizontal); self.dist_slider.setRange(20, 100); self.dist_slider.setValue(30)
        self.dist_slider.valueChanged.connect(self.update_dist)
        d_row.addWidget(self.dist_slider); self.dist_lbl = QLabel("3.0"); d_row.addWidget(self.dist_lbl)
        proj_layout.addLayout(d_row)
        # Scale
        s_row = QHBoxLayout(); s_row.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal); self.scale_slider.setRange(20, 200); self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.update_scale)
        s_row.addWidget(self.scale_slider); self.scale_lbl = QLabel("1.0"); s_row.addWidget(self.scale_lbl)
        proj_layout.addLayout(s_row)
        layout.addWidget(proj_group)

        # Display
        disp_group = QGroupBox("Display"); disp_layout = QVBoxLayout(disp_group)
        self.vtx_cb = QCheckBox("Show Vertices"); self.vtx_cb.setChecked(True)
        self.vtx_cb.toggled.connect(lambda c: setattr(self.renderer, 'show_vertices', c))
        self.vtx_cb.toggled.connect(self.renderer.update); disp_layout.addWidget(self.vtx_cb)
        self.axes_cb = QCheckBox("Show Axes"); self.axes_cb.setChecked(True)
        self.axes_cb.toggled.connect(lambda c: setattr(self.renderer, 'show_axes', c))
        self.axes_cb.toggled.connect(self.renderer.update); disp_layout.addWidget(self.axes_cb)
        layout.addWidget(disp_group)

        return panel

    def load_shape(self, name):
        methods = {
            "Tesseract": FourDShapes.tesseract, "4D Simplex": FourDShapes.simplex_4d,
            "16-cell": FourDShapes.cell_16, "24-cell": FourDShapes.cell_24,
            "3-3 Duoprism": FourDShapes.duoprism_3_3, "Hyperdodecahedron": FourDShapes.hyperdodecahedron
        }
        if name in methods:
            v, e, f, n = methods[name]()
            self.renderer.set_shape(v, e, f, n)
            self.shape_info.setText(f"{n} | V:{len(v)} E:{len(e)} F:{len(f)}")

    def update_rotation(self, idx, val):
        self.renderer.set_rotation(idx, val)
        self.rotation_value_labels[idx].setText(f"{val}°")
    def update_dist(self, val):
        d = val/10.0; self.renderer.set_projection_distance(d); self.dist_lbl.setText(f"{d:.1f}")
    def update_scale(self, val):
        s = val/100.0; self.renderer.set_scale(s); self.scale_lbl.setText(f"{s:.2f}")
    def toggle_anim(self, idx, checked):
        self.animation_active[idx] = checked
        if any(self.animation_active): self.animation_timer.start()
        else: self.animation_timer.stop()
    def animate(self):
        for i, a in enumerate(self.animation_active):
            if a: self.rotation_sliders[i].setValue((self.rotation_sliders[i].value() + 1) % 361)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    fmt = QSurfaceFormat(); fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    window = MainWindow(); window.show()
    sys.exit(app.exec())

if __name__ == "__main__": main()