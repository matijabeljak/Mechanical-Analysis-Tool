import sys
import math
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget, QListWidget,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QToolBar, 
    QAction, QListWidgetItem, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QSplitter, QTextEdit, QDialog
)

# Colors & constants - Enhanced to match spec
COLORS = {
    'background': QColor(250, 250, 250),
    'beam': QColor(80, 80, 80),
    'beam_selected': QColor(255, 180, 0),
    'force': QColor(220, 0, 0),
    'moment': QColor(0, 0, 220),
    'support_pinned': QColor(0, 180, 0),
    'support_roller': QColor(0, 0, 180),
    'node': QColor(0, 0, 0),
    'selected': QColor(255, 180, 0),
    'text': QColor(40, 40, 40),
    'grid': QColor(220, 220, 220),
    'preview': QColor(150, 150, 255),
    'diagram_axial': QColor(220, 0, 0),
    'diagram_shear': QColor(0, 150, 0),
    'diagram_moment': QColor(0, 0, 220),
}
SNAP_PIX = 10
GRID_STEP = 50
DIAGRAM_SAMPLES = 50

import traceback
import sys

def excepthook(exc_type, exc_value, exc_tb):
    print("\n=== Unhandled Exception ===")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Error")
    msg.setText(str(exc_value))
    msg.setDetailedText("\n".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    msg.exec_()

sys.excepthook = excepthook

# ------------------------------------------------------------
# Domain model - Enhanced with better support types
# ------------------------------------------------------------

@dataclass
class Node:
    x: float
    y: float
    id: int = 0
    name: str = ""
    constraints: Tuple[bool, bool, bool] = (False, False, False)
    support_type: str = "none"

@dataclass
class Beam:
    start: int
    end: int
    id: int = 0
    name: str = ""
    E: float = 2e11
    I: float = 1e-6
    A: float = 1e-4
    material: str = "Steel"

@dataclass
class Load:
    type: str
    x: float
    y: float
    magnitude: float
    direction: float = -90.0
    node_id: Optional[int] = None
    beam_id: Optional[int] = None
    position: float = 0.5
    id: int = 0
    name: str = ""
    is_relative_to_beam: bool = False

@dataclass
class Results:
    U: Optional[np.ndarray] = None
    # Per-beam sampled internal forces
    N: dict = field(default_factory=dict)  # beam_id -> list[float]
    V: dict = field(default_factory=dict)
    M: dict = field(default_factory=dict)


class FEMSystem:
    def __init__(self):
        self.nodes: List[Node] = []
        self.beams: List[Beam] = []
        self.loads: List[Load] = []
        self.results = Results()
        self._next_load_id = 1

    # ---------- Node/Beam helpers ----------
    def add_node(self, x: float, y: float) -> int:
        nid = len(self.nodes) + 1
        self.nodes.append(Node(x, y, nid))
        return nid

    def add_load(self, load_type: str, x: float, y: float, magnitude: float, **kwargs) -> int:
        load_id = self._next_load_id
        self._next_load_id += 1
        load = Load(load_type, x, y, magnitude, id=load_id, **kwargs)
        self.loads.append(load)
        return load_id

    def add_beam(self, start_id: int, end_id: int) -> int:
        # Check if beam already exists
        for beam in self.beams:
            if (beam.start == start_id and beam.end == end_id) or (beam.start == end_id and beam.end == start_id):
                return beam.id
                
        bid = len(self.beams) + 1
        self.beams.append(Beam(start=start_id, end=end_id, id=bid))
        return bid

    def find_nearest_node(self, x: float, y: float, max_distance: float = 20.0) -> Optional[Node]:
        if not self.nodes:
            return None
        p = np.array([x, y])
        best_node = None
        best_dist = float('inf')
        
        for node in self.nodes:
            dist = np.linalg.norm(p - np.array([node.x, node.y]))
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best_node = node
                
        return best_node

    def add_support(self, node_id: int, support_type: str):
        """Add support to existing node"""
        if node_id < 1 or node_id > len(self.nodes):
            return
            
        node = self.nodes[node_id - 1]
        node.support_type = support_type
        
        # Set constraints based on support type
        if support_type == "pinned":
            node.constraints = (True, True, False)  # Fixed in x and y, free rotation
        elif support_type == "roller":
            node.constraints = (False, True, False)  # Fixed in y only, free in x and rotation
        else:  # none
            node.constraints = (False, False, False)
            
    def toggle_support_type(self, node_id: int):
        if 1 <= node_id <= len(self.nodes):
            node = self.nodes[node_id - 1]
            if node.support_type == "pinned":
                node.support_type = "roller"
                node.constraints = (False, True, False)
            elif node.support_type == "roller":
                node.support_type = "none"
                node.constraints = (False, False, False)
            else:  # none
                node.support_type = "pinned"
                node.constraints = (True, True, False)
            return True
        return False
    

    # ---------- FEM assembly ----------
    def assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.nodes)
        dof = 3 * n
        K = np.zeros((dof, dof), dtype=float)
        F = np.zeros(dof, dtype=float)

        # Apply loads to nodes
        for L in self.loads:
            if L.node_id is not None:
                # Load is directly applied to a node
                i = (L.node_id - 1) * 3
            else:
                # Find nearest node
                node = self.find_nearest_node(L.x, L.y)
                if node is None:
                    continue
                i = (node.id - 1) * 3
                
            if L.type == 'force':
                ang = math.radians(L.direction)
                F[i + 0] += L.magnitude * math.cos(ang)
                F[i + 1] += L.magnitude * math.sin(ang)
            elif L.type == 'moment':
                F[i + 2] += L.magnitude

        # Element stiffness assembly (2D frame element)
        for b in self.beams:
            n1 = self.nodes[b.start - 1]
            n2 = self.nodes[b.end - 1]
            L = math.hypot(n2.x - n1.x, n2.y - n1.y)
            if L <= 1e-9:
                continue
            cx = (n2.x - n1.x) / L
            cy = (n2.y - n1.y) / L

            # Local stiffness (6x6): axial + bending
            E, A, I = b.E, b.A, b.I
            k_ax = E * A / L
            k_b = E * I / (L**3)
            k_local = np.array([
                [ k_ax,      0,        0, -k_ax,      0,        0],
                [    0,  12*k_b,  6*L*k_b,     0, -12*k_b,  6*L*k_b],
                [    0, 6*L*k_b, 4*L*L*k_b,     0, -6*L*k_b, 2*L*L*k_b],
                [-k_ax,      0,        0,  k_ax,      0,        0],
                [    0, -12*k_b, -6*L*k_b,     0,  12*k_b, -6*L*k_b],
                [    0, 6*L*k_b, 2*L*L*k_b,     0, -6*L*k_b, 4*L*L*k_b],
            ], dtype=float)

            # Transformation matrix T (from local to global)
            T = np.zeros((6, 6), dtype=float)
            R = np.array([[cx, cy, 0], [-cy, cx, 0], [0, 0, 1]], dtype=float)
            T[:3, :3] = R
            T[3:, 3:] = R

            k_global = T.T @ k_local @ T

            # DOF indices
            i = (n1.id - 1) * 3
            j = (n2.id - 1) * 3
            idx = [i, i+1, i+2, j, j+1, j+2]
            for a in range(6):
                for c in range(6):
                    K[idx[a], idx[c]] += k_global[a, c]

        # Apply constraints (penalty method)
        for node in self.nodes:
            base = (node.id - 1) * 3
            for d, fixed in enumerate(node.constraints):
                if fixed:
                    K[base + d, :] = 0
                    K[:, base + d] = 0
                    K[base + d, base + d] = 1.0
                    F[base + d] = 0.0

        return K, F
    def find_nearest_beam(self, x: float, y: float, max_distance: float = 20.0) -> Optional[Beam]:
        if not self.beams:
            return None
            
        best_beam = None
        best_dist = float('inf')
        
        for beam in self.beams:
            n1 = self.nodes[beam.start - 1]
            n2 = self.nodes[beam.end - 1]
            dist = self._point_to_segment_dist(x, y, n1.x, n1.y, n2.x, n2.y)
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best_beam = beam
                
        return best_beam

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        L2 = vx*vx + vy*vy
        if L2 == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
        projx, projy = x1 + t * vx, y1 + t * vy
        return math.hypot(px - projx, py - projy)

    def solve(self):
        try:
            K, F = self.assemble()
            # Check if system is constrained
            if not any(any(node.constraints) for node in self.nodes):
                raise ValueError("Structure is not properly constrained. Add supports.")
                
            # Regularization for singular systems
            if np.linalg.matrix_rank(K) < K.shape[0]:
                K = K + np.eye(K.shape[0]) * 1e-9
            try:
                U = np.linalg.solve(K, F)
            except np.linalg.LinAlgError:
                U, *_ = np.linalg.lstsq(K, F, rcond=None)
            self.results.U = U
            self._compute_internal_forces()
            return True
        except Exception as e:
            QMessageBox.critical(None, "Analysis Error", f"Analysis failed: {str(e)}")
            return False

    def _compute_internal_forces(self):
        self.results.N.clear(); self.results.V.clear(); self.results.M.clear()
        U = self.results.U
        if U is None:
            return
        for b in self.beams:
            n1 = self.nodes[b.start - 1]
            n2 = self.nodes[b.end - 1]
            L = math.hypot(n2.x - n1.x, n2.y - n1.y)
            if L <= 1e-9:
                self.results.N[b.id] = []
                self.results.V[b.id] = []
                self.results.M[b.id] = []
                continue
            cx = (n2.x - n1.x) / L
            cy = (n2.y - n1.y) / L

            # DOF vector in global
            i = (n1.id - 1) * 3
            j = (n2.id - 1) * 3
            ug = np.array([U[i], U[i+1], U[i+2], U[j], U[j+1], U[j+2]])

            # Transform to local
            R = np.array([[cx, cy, 0], [-cy, cx, 0], [0, 0, 1]], dtype=float)
            T = np.zeros((6, 6), dtype=float)
            T[:3, :3] = R
            T[3:, 3:] = R
            ul = T @ ug

            # Extract local dofs
            u1, v1, th1, u2, v2, th2 = ul
            E, A, I = b.E, b.A, b.I

            N_vals = []
            V_vals = []
            M_vals = []
            for s in np.linspace(0, L, DIAGRAM_SAMPLES):
                xi = s / L
                # Axial: linear interpolation -> constant N
                N = E * A / L * (u2 - u1)

                # Bending calculations
                k_b = E * I / (L**3)
                Kb = np.array([
                    [ 12*k_b,  6*L*k_b, -12*k_b,  6*L*k_b],
                    [  6*L*k_b, 4*L*L*k_b, -6*L*k_b, 2*L*L*k_b],
                    [-12*k_b, -6*L*k_b, 12*k_b, -6*L*k_b],
                    [  6*L*k_b, 2*L*L*k_b, -6*L*k_b, 4*L*L*k_b],
                ])
                vb = np.array([v1, th1, v2, th2])
                fb = Kb @ vb
                V1, M1, V2, M2 = fb
                V = (1 - xi) * V1 + xi * (-V2)
                M = (1 - xi) * M1 + xi * M2

                N_vals.append(N)
                V_vals.append(V)
                M_vals.append(M)

            self.results.N[b.id] = N_vals
            self.results.V[b.id] = V_vals
            self.results.M[b.id] = M_vals


# ------------------------------------------------------------
# UI widgets
# ------------------------------------------------------------
class DiagramWidget(QWidget):
    def __init__(self, fem: FEMSystem):
        super().__init__()
        self.fem = fem
        self.current_beam_id: Optional[int] = None
        self.setMinimumHeight(220)

    def set_beam(self, bid: Optional[int]):
        self.current_beam_id = bid
        self.update()

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(245, 245, 245))
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.current_beam_id is None:
            painter.setPen(COLORS['text'])
            painter.drawText(self.rect(), Qt.AlignCenter, "Select a beam to view internal forces")
            return

        W = self.width(); H = self.height()
        left = 50; right = W - 20; top = 10; bottom = H - 10
        section_height = (H - 20) // 3
        
        diagrams = [
            ("Axial Force (N)", COLORS['diagram_axial'], self.fem.results.N.get(self.current_beam_id, [])),
            ("Shear Force (V)", COLORS['diagram_shear'], self.fem.results.V.get(self.current_beam_id, [])),
            ("Bending Moment (M)", COLORS['diagram_moment'], self.fem.results.M.get(self.current_beam_id, []))
        ]
        
        for idx, (label, color, values) in enumerate(diagrams):
            rtop = top + idx * section_height
            rbot = top + (idx + 1) * section_height - 5
            midy = (rtop + rbot) // 2
            
            # Draw section label
            painter.setPen(QPen(COLORS['text'], 1))
            painter.drawText(int(left + 4), int(rtop + 16), label)
            
            # Draw zero line
            painter.setPen(QPen(COLORS['grid'], 1, Qt.DashLine))
            painter.drawLine(int(left), int(midy), int(right), int(midy))
            
            if len(values) < 2:
                painter.setPen(QPen(COLORS['text'], 1))
                painter.drawText(int(left), int(midy), "No data")
                continue
                
            # Draw diagram
            vmax = max(1e-9, max(abs(min(values)), abs(max(values))))
            scale = (rbot - rtop) * 0.4 / vmax
            step = (right - left) / (len(values) - 1)
            
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            path_points = []
            for i, val in enumerate(values):
                x = left + i * step
                y = midy - val * scale
                path_points.append(QPointF(x, y))
            
            for i in range(len(path_points) - 1):
                painter.drawLine(path_points[i], path_points[i + 1])
            
            # Fill area for moment diagram (last one)
            if idx == 2 and len(values) > 1:
                fill_color = QColor(color)
                fill_color.setAlpha(40)
                painter.setBrush(fill_color)
                painter.setPen(Qt.NoPen)
                polygon_points = path_points + [QPointF(right, midy), QPointF(left, midy)]
                painter.drawPolygon(*polygon_points)


class Canvas(QWidget):
    def __init__(self, fem: FEMSystem, diagram: DiagramWidget, list_widget: QListWidget, prop_table: QTableWidget):
        super().__init__()
        self.fem = fem
        self.diag = diagram
        self.list_widget = list_widget
        self.prop_table = prop_table
        self.list_widget.itemClicked.connect(self._on_list_item_clicked)
        
        self.selected_load_id: Optional[int] = None
        self.setMouseTracking(True)
        self.mode = 'select'  # 'select', 'node', 'add_beam', 'add_force', 'add_moment', 'support_pinned', 'support_roller', 'pan'
        self.preview_start: Optional[Tuple[float, float]] = None
        self.preview_node: Optional[Tuple[float, float]] = None
        self.selected_beam_id: Optional[int] = None
        self.selected_node_id: Optional[int] = None

        # View transform
        self.scale = 1.0
        self.offset = np.array([0.0, 0.0])
        self.panning = False
        self.last_mouse = None
        
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- Transform helpers ----------
    def world_to_screen(self, x: float, y: float) -> QPointF:
        X = (x + self.offset[0]) * self.scale
        Y = (y + self.offset[1]) * self.scale
        return QPointF(X, Y)
    def screen_to_world(self, px: float, py: float) -> Tuple[float, float]:
        x = px / self.scale - self.offset[0]
        y = py / self.scale - self.offset[1]
        return x, y

    # ---------- Painting ----------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), COLORS['background'])
        p.setRenderHint(QPainter.Antialiasing)

        # Grid
        self._draw_grid(p)

        # Beams
        for b in self.fem.beams:
            self._draw_beam(p, b)

        # Nodes
        for n in self.fem.nodes:
            self._draw_node(p, n)

        # Loads
        for L in self.fem.loads:
            self._draw_load(p, L)

        # Preview elements
        self._draw_preview(p)

        # Selection highlight
        self._draw_selection(p)

    def _draw_grid(self, p: QPainter):
        p.setPen(QPen(COLORS['grid'], 1))
        step = GRID_STEP * self.scale
        if step >= 8:
            startx = - (self.offset[0] * self.scale) % step
            starty = - (self.offset[1] * self.scale) % step
            x = startx
            while x < self.width():
                p.drawLine(int(x), 0, int(x), self.height())
                x += step
            y = starty
            while y < self.height():
                p.drawLine(0, int(y), self.width(), int(y))
                y += step

    def _draw_beam(self, p: QPainter, b: Beam):
        n1 = self.fem.nodes[b.start - 1]
        n2 = self.fem.nodes[b.end - 1]
        p1 = self.world_to_screen(n1.x, n1.y)
        p2 = self.world_to_screen(n2.x, n2.y)
        
        if b.id == self.selected_beam_id:
            p.setPen(QPen(COLORS['beam_selected'], 4))
        else:
            p.setPen(QPen(COLORS['beam'], 3))
            
        p.drawLine(p1, p2)

    def _draw_node(self, p: QPainter, n: Node):
        pos = self.world_to_screen(n.x, n.y)
        
        # Draw node
        p.setPen(QPen(COLORS['node'], 1))
        p.setBrush(Qt.black if n.id != self.selected_node_id else COLORS['selected'])
        p.drawEllipse(pos, 4, 4)
        
        # Draw supports
        if n.support_type == "pinned":
            p.setBrush(COLORS['support_pinned'])
            p.setPen(Qt.NoPen)
            points = [
                QPointF(pos.x() - 10, pos.y() + 10),
                QPointF(pos.x() + 10, pos.y() + 10), 
                QPointF(pos.x(), pos.y() + 20)
            ]
            p.drawPolygon(*points)
        elif n.support_type == "roller":
            p.setBrush(COLORS['support_roller'])
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(pos.x(), pos.y() + 10), 7, 7)

    def _draw_load(self, p: QPainter, L: Load):
        pos = self.world_to_screen(L.x, L.y)
        if L.type == 'force':
            p.setPen(QPen(COLORS['force'], 3))
            ang = math.radians(L.direction)
            # Use logarithmic scaling for arrow length
            base_length = 20  # base length for 1000N force
            if L.magnitude > 0:
                log_length = base_length * math.log10(L.magnitude / 1000 + 1) + 10
                length = min(80, max(15, log_length))  # clamp between 15-80 pixels
            else:
                length = 15
            end_pos = QPointF(
                pos.x() + length * math.cos(ang),
                pos.y() + length * math.sin(ang)
            )
            p.drawLine(pos, end_pos)
            self._draw_arrow(p, end_pos, ang)
            
            # Force magnitude label - convert to integer coordinates
            p.setPen(COLORS['text'])
            p.drawText(int(end_pos.x() + 5), int(end_pos.y()), f"{L.magnitude:.0f} N")
        else:  # moment
            p.setPen(QPen(COLORS['moment'], 2))
            # Use logarithmic scaling for moment radius
            base_radius = 8  # base radius for 1000N·m moment
            if L.magnitude > 0:
                log_radius = base_radius * math.log10(L.magnitude / 1000 + 1) + 5
                r = min(25, max(8, log_radius))  # clamp between 8-25 pixels
            else:
                r = 8
            p.drawEllipse(pos, r, r)
            
            # Moment direction indicator
            p.drawLine(
                int(pos.x() - r), int(pos.y()), 
                int(pos.x() + r), int(pos.y())
            )
            p.drawLine(
                int(pos.x()), int(pos.y() - r), 
                int(pos.x()), int(pos.y() + r)
            )
            
            # Moment magnitude label - convert to integer coordinates
            p.setPen(COLORS['text'])
            p.drawText(int(pos.x() + r + 5), int(pos.y()), f"{L.magnitude:.0f} N·m")

    def _draw_preview(self, p: QPainter):
        if self.last_mouse is None:
            return
            
        xw, yw = self.screen_to_world(*self.last_mouse)
            
        if self.mode == 'add_beam' and self.preview_start is not None:
            p.setPen(QPen(COLORS['preview'], 2, Qt.DashLine))
            # Extract just x,y from preview_start (ignore node_id)
            start_x, start_y, _ = self.preview_start
            start_pos = self.world_to_screen(start_x, start_y)
            end_pos = self.world_to_screen(xw, yw)
            p.drawLine(start_pos, end_pos)
        elif self.mode == 'node':
            p.setPen(QPen(COLORS['preview'], 2))
            p.setBrush(Qt.NoBrush)
            preview_pos = self.world_to_screen(xw, yw)
            p.drawEllipse(preview_pos, 6, 6)

    def _draw_selection(self, p: QPainter):
    # Highlight selected beam
        if self.selected_beam_id:
            b = next((bb for bb in self.fem.beams if bb.id == self.selected_beam_id), None)
            if b:
                p.setPen(QPen(COLORS['selected'], 4))
                n1 = self.fem.nodes[b.start - 1]
                n2 = self.fem.nodes[b.end - 1]
                p1 = self.world_to_screen(n1.x, n1.y)
                p2 = self.world_to_screen(n2.x, n2.y)
                p.drawLine(p1, p2)
            
            # Highlight selected node
        if self.selected_node_id:
            node = next((n for n in self.fem.nodes if n.id == self.selected_node_id), None)
            if node:
                pos = self.world_to_screen(node.x, node.y)
                p.setPen(QPen(COLORS['selected'], 2))
                p.setBrush(COLORS['selected'])
                p.drawEllipse(pos, 6, 6)
        
        # Highlight selected load
        if self.selected_load_id:
            load = next((l for l in self.fem.loads if l.id == self.selected_load_id), None)
            if load:
                pos = self.world_to_screen(load.x, load.y)
                if load.type == 'force':
                    # Draw selection circle around force
                    p.setPen(QPen(COLORS['selected'], 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(pos, 15, 15)
                else:  # moment
                     # Draw selection circle around moment
                    p.setPen(QPen(COLORS['selected'], 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(pos, 20, 20)

    def _draw_arrow(self, p: QPainter, tip: QPointF, angle_rad: float):
        a1 = angle_rad + math.radians(150)
        a2 = angle_rad - math.radians(150)
        s = 10
        p.setBrush(COLORS['force'])
        points = [
            tip,
            QPointF(tip.x() + s * math.cos(a1), tip.y() + s * math.sin(a1)),
            QPointF(tip.x() + s * math.cos(a2), tip.y() + s * math.sin(a2))
        ]
        p.drawPolygon(*points)

    # ---------- Mouse & keys ----------
    def mousePressEvent(self, e):
        self.last_mouse = (e.x(), e.y())
        xw, yw = self.screen_to_world(e.x(), e.y())
        
        if e.button() == Qt.MiddleButton:
            self.panning = True
            return
            
        if self.mode == 'node':
            # Auto-snap to existing nodes
            existing_node = self.fem.find_nearest_node(xw, yw)
            if existing_node:
                # Don't create new node if close to existing one
                self.selected_node_id = existing_node.id
                self._populate_properties(existing_node)
            else:
                nid = self.fem.add_node(xw, yw)
                node = self.fem.nodes[-1]
                node.name = f"Node {nid}"
                # Select the newly created node
                self.selected_node_id = nid
                self.selected_beam_id = None
                self.selected_load_id = None
                self._populate_properties(node)
            
            self._refresh_lists()
            self._highlight_list()
            self.update()
            
        elif self.mode == 'add_beam':
            if self.preview_start is None:
                # Find or create start node
                node = self.fem.find_nearest_node(xw, yw)
                if node:
                    self.preview_start = (node.x, node.y, node.id)
                else:
                    nid = self.fem.add_node(xw, yw)
                    node = self.fem.nodes[-1]
                    node.name = f"Node {nid}"
                    self.preview_start = (xw, yw, nid)
            else:
                # Find or create end node and create beam
                node = self.fem.find_nearest_node(xw, yw)
                if node:
                    end_node_id = node.id
                else:
                    nid = self.fem.add_node(xw, yw)
                    node = self.fem.nodes[-1]
                    node.name = f"Node {nid}"
                    end_node_id = nid
                    
                # Get start node ID from preview
                start_x, start_y, start_node_id = self.preview_start
                
                if start_node_id != end_node_id:
                    bid = self.fem.add_beam(start_node_id, end_node_id)
                    beam = self.fem.beams[-1]
                    beam.name = f"Beam {bid}"
                    # Select the newly created beam
                    self.selected_beam_id = bid
                    self.selected_node_id = None
                    self.selected_load_id = None
                    self._populate_properties(beam)
                    
                self.preview_start = None
            
            self._refresh_lists()
            self._highlight_list()
            self.update()
            
        elif self.mode in ['support_pinned', 'support_roller']:
            # Auto-create node if none exists nearby
            node = self.fem.find_nearest_node(xw, yw)
            if not node:
                nid = self.fem.add_node(xw, yw)
                node = self.fem.nodes[-1]
                node.name = f"Node {nid}"
            
            support_type = self.mode.replace('support_', '')
            self.fem.add_support(node.id, support_type)
            
            # Select the node with support
            self.selected_node_id = node.id
            self.selected_beam_id = None
            self.selected_load_id = None
            self._populate_properties(node)
            
            self._refresh_lists()
            self._highlight_list()
            self.update()
            
        elif self.mode == 'add_force':
            # Find nearest node and beam
            node = self.fem.find_nearest_node(xw, yw)
            beam = self.fem.find_nearest_beam(xw, yw)
            
            if not node:
                # Create new node at the clicked position
                nid = self.fem.add_node(xw, yw)
                node = self.fem.nodes[-1]
                node.name = f"Node {nid}"
                # After creating node, find the nearest beam again (might be the same or different)
                beam = self.fem.find_nearest_beam(xw, yw)
            
            # Calculate beam-relative angle if beam is nearby and close to the node
            beam_angle = 0
            is_relative = False
            beam_id = None
            
            if beam:
                # Check if the node is actually on the beam (within tolerance)
                n1 = self.fem.nodes[beam.start - 1]
                n2 = self.fem.nodes[beam.end - 1]
                dist_to_beam, pos_along_beam = self._point_to_segment_dist_and_pos(node.x, node.y, n1.x, n1.y, n2.x, n2.y)
                
                # If node is close enough to the beam, attach the force to the beam
                if dist_to_beam < 5.0:  # 5 unit tolerance
                    beam_angle = math.degrees(math.atan2(n2.y - n1.y, n2.x - n1.x))
                    is_relative = True
                    beam_id = beam.id
                    # Move node to be exactly on the beam
                    node.x = n1.x + pos_along_beam * (n2.x - n1.x)
                    node.y = n1.y + pos_along_beam * (n2.y - n1.y)
            
            load_id = self.fem.add_load('force', node.x, node.y, 1000.0, 
                                      direction=-90.0, node_id=node.id,
                                      beam_id=beam_id,
                                      is_relative_to_beam=is_relative)
            
            load = next(l for l in self.fem.loads if l.id == load_id)
            load.name = f"Force {load_id}"
            
            # Select the newly created force
            self.selected_load_id = load_id
            self.selected_beam_id = None
            self.selected_node_id = None
            self._populate_properties(load)
            
            self._refresh_lists()
            self._highlight_list()
            self.update()
            
        elif self.mode == 'add_moment':
            # Find nearest node and beam
            node = self.fem.find_nearest_node(xw, yw)
            beam = self.fem.find_nearest_beam(xw, yw)
            
            if not node:
                # Create new node at the clicked position
                nid = self.fem.add_node(xw, yw)
                node = self.fem.nodes[-1]
                node.name = f"Node {nid}"
                # After creating node, find the nearest beam again
                beam = self.fem.find_nearest_beam(xw, yw)
            
            beam_id = None
            if beam:
                # Check if the node is actually on the beam
                n1 = self.fem.nodes[beam.start - 1]
                n2 = self.fem.nodes[beam.end - 1]
                dist_to_beam, pos_along_beam = self._point_to_segment_dist_and_pos(node.x, node.y, n1.x, n1.y, n2.x, n2.y)
                
                # If node is close enough to the beam, attach the moment to the beam
                if dist_to_beam < 5.0:  # 5 unit tolerance
                    beam_id = beam.id
                    # Move node to be exactly on the beam
                    node.x = n1.x + pos_along_beam * (n2.x - n1.x)
                    node.y = n1.y + pos_along_beam * (n2.y - n1.y)
            
            load_id = self.fem.add_load('moment', node.x, node.y, 1000.0, 
                                        node_id=node.id,
                                        beam_id=beam_id)
    
            load = next(l for l in self.fem.loads if l.id == load_id)
            load.name = f"Moment {load_id}"
            
            # Select the newly created moment
            self.selected_load_id = load_id
            self.selected_beam_id = None
            self.selected_node_id = None
            self._populate_properties(load)
            
            self._refresh_lists()
            self._highlight_list()
            self.update()
        else:  # select
            self._select_at(xw, yw)

    def mouseMoveEvent(self, e):
        if self.panning and e.buttons() & Qt.MiddleButton:
            if self.last_mouse:
                dx = (e.x() - self.last_mouse[0]) / self.scale
                dy = (e.y() - self.last_mouse[1]) / self.scale
                self.offset[0] -= dx
                self.offset[1] -= dy
        self.last_mouse = (e.x(), e.y())
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.panning = False

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        factor = 1.1 if delta > 0 else 1/1.1
        pos = np.array([e.x(), e.y()], dtype=float)
        before = np.array(self.screen_to_world(*pos))
        self.scale *= factor
        after = np.array(self.screen_to_world(*pos))
        self.offset += (after - before)
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_N:
            self.mode = 'node'
        elif e.key() == Qt.Key_B:
            self.mode = 'add_beam'
        elif e.key() == Qt.Key_F:
            self.mode = 'add_force'
        elif e.key() == Qt.Key_M:
            self.mode = 'add_moment'
        elif e.key() == Qt.Key_Escape:
            self.mode = 'select'
            self.preview_start = None
        elif e.key() == Qt.Key_S:
            self._solve()
        elif e.key() == Qt.Key_Delete and self.selected_beam_id:
            self._delete_selected()
        self.update()

    # ---------- Helpers ----------
    def _solve(self):
        if self.fem.solve():
            self.diag.set_beam(self.selected_beam_id)
            self.update()

    def _delete_selected(self):
        if self.selected_beam_id:
            self.fem.beams = [b for b in self.fem.beams if b.id != self.selected_beam_id]
            self.selected_beam_id = None
            self._refresh_lists()
            self._populate_properties(None)
            self.diag.set_beam(None)
            self.update()

    def _select_at(self, x: float, y: float):
        # Reset selection
        self.selected_beam_id = None
        self.selected_node_id = None
        self.selected_load_id = None
        
        # Try to select a load first (they're smaller targets)
        best_load = None
        best_load_d = 20.0 / self.scale
        for load in self.fem.loads:
            screen_pos = self.world_to_screen(load.x, load.y)
            world_click = self.screen_to_world(screen_pos.x(), screen_pos.y())
            d = math.hypot(x - load.x, y - load.y)
            if d < best_load_d:
                best_load_d = d
                best_load = load
        
        if best_load:
            self.selected_load_id = best_load.id
            self._populate_properties(best_load)
        else:
            # Try to select a beam
            best_beam = None
            best_d = 12.0 / self.scale
            for b in self.fem.beams:
                n1 = self.fem.nodes[b.start - 1]
                n2 = self.fem.nodes[b.end - 1]
                d = self._point_to_segment_dist(x, y, n1.x, n1.y, n2.x, n2.y)
                if d < best_d:
                    best_d = d
                    best_beam = b
                    
            if best_beam:
                self.selected_beam_id = best_beam.id
                self._populate_properties(best_beam)
            else:
                # Try to select a node
                node = self.fem.find_nearest_node(x, y, max_distance=15.0/self.scale)
                if node:
                    self.selected_node_id = node.id
                    self._populate_properties(node)
                else:
                    self._populate_properties(None)
        
        self._highlight_list()
        self.diag.set_beam(self.selected_beam_id)
        self.update()



    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        L2 = vx*vx + vy*vy
        if L2 == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
        projx, projy = x1 + t * vx, y1 + t * vy
        return math.hypot(px - projx, py - projy)

    def _populate_properties(self, item):
        # Safely disconnect without error if not connected
        try:
            self.prop_table.itemChanged.disconnect(self._on_prop_changed)
        except TypeError:
            pass  # Not connected, which is fine
        
        self.prop_table.clear()
        self.prop_table.setColumnCount(2)
        self.prop_table.setHorizontalHeaderLabels(["Property", "Value"])
        
        if item is None:
            self.prop_table.setRowCount(0)
            # Reconnect after clearing
            self.prop_table.itemChanged.connect(self._on_prop_changed)
            return
            
        props = []
        
        if isinstance(item, Beam):
            props = [
                ("Name", item.name),
                ("ID", item.id),
                ("Start Node", item.start),
                ("End Node", item.end),
                ("Young's Modulus (E)", item.E),
                ("Moment of Inertia (I)", item.I),
                ("Area (A)", item.A),
                ("Material", item.material)
            ]
        elif isinstance(item, Node):
            props = [
                ("Name", item.name),
                ("ID", item.id),
                ("X Position", item.x),
                ("Y Position", item.y),
                ("Support Type", item.support_type)
            ]
        elif isinstance(item, Load):
            angle_display = f"{item.direction}° ({'Beam' if item.is_relative_to_beam else 'Global'})"
            props = [
                ("Name", item.name),
                ("ID", item.id),
                ("Type", item.type),
                ("Magnitude", item.magnitude),
                ("Direction", angle_display),
                ("Reference", "Beam Relative" if item.is_relative_to_beam else "Global"),
                ("Node ID", item.node_id or ""),
                ("Beam ID", item.beam_id or "")
            ]
        
        self.prop_table.setRowCount(len(props))
        
        for r, (k, v) in enumerate(props):
            self.prop_table.setItem(r, 0, QTableWidgetItem(str(k)))
            item_widget = QTableWidgetItem(str(v))
            
            # Make certain fields editable
            if k in ["Name", "Young's Modulus (E)", "Moment of Inertia (I)", "Area (A)", 
                    "Material", "X Position", "Y Position", "Magnitude", "Direction"]:
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
            else:
                item_widget.setFlags(item_widget.flags() & ~Qt.ItemIsEditable)
                
            # Special handling for support type and load reference
            if k == "Support Type":
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
            elif k == "Reference":
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
                
            self.prop_table.setItem(r, 1, item_widget)
            
        # Reconnect after populating
        self.prop_table.itemChanged.connect(self._on_prop_changed)
        for r, (k, v) in enumerate(props):
            self.prop_table.setItem(r, 0, QTableWidgetItem(str(k)))
            item_widget = QTableWidgetItem(str(v))
            
            # Make certain fields editable
            if k in ["Name", "Young's Modulus (E)", "Moment of Inertia (I)", "Area (A)", 
                    "Material", "X Position", "Y Position", "Magnitude", "Direction"]:
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
            else:
                item_widget.setFlags(item_widget.flags() & ~Qt.ItemIsEditable)
                
            # Special handling for support type and load reference
            if k == "Support Type":
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
            elif k == "Reference":
                item_widget.setFlags(item_widget.flags() | Qt.ItemIsEditable)
                
            self.prop_table.setItem(r, 1, item_widget)
            
        self.prop_table.itemChanged.connect(self._on_prop_changed)

    def _on_prop_changed(self, item: QTableWidgetItem):
        if item.row() < 0:
            return
            
        key = self.prop_table.item(item.row(), 0).text()
        val = item.text()
        
        try:
            if self.selected_beam_id:
                beam = next(b for b in self.fem.beams if b.id == self.selected_beam_id)
                if key == "Name":
                    beam.name = val
                elif key == "Young's Modulus (E)":
                    beam.E = float(val)
                elif key == "Moment of Inertia (I)":
                    beam.I = float(val)
                elif key == "Area (A)":
                    beam.A = float(val)
                elif key == "Material":
                    beam.material = val
                    
            elif self.selected_node_id:
                node = next(n for n in self.fem.nodes if n.id == self.selected_node_id)
                if key == "Name":
                    node.name = val
                elif key == "X Position":
                    node.x = float(val)
                elif key == "Y Position":
                    node.y = float(val)
                elif key == "Support Type":
                    if val.lower() in ["pinned", "roller", "none"]:
                        self.fem.add_support(node.id, val.lower())
                        
            elif self.selected_load_id:
                load = next(l for l in self.fem.loads if l.id == self.selected_load_id)
                if key == "Name":
                    load.name = val
                elif key == "Magnitude":
                    load.magnitude = float(val)
                elif key == "Direction":
                    # Parse angle value
                    if "°" in val:
                        val = val.split("°")[0]
                    load.direction = float(val)
                elif key == "Reference":
                    load.is_relative_to_beam = ("beam" in val.lower())
                    
        except ValueError:
            pass  # Ignore conversion errors
            
        self.update()


    def _on_list_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if not data:
            return
            
        item_type, item_id = data
        
        if item_type == "beam":
            self.selected_beam_id = item_id
            self.selected_node_id = None
            self.selected_load_id = None
            beam = next((b for b in self.fem.beams if b.id == item_id), None)
            self._populate_properties(beam)
            
        elif item_type == "node":
            self.selected_node_id = item_id
            self.selected_beam_id = None
            self.selected_load_id = None
            node = next((n for n in self.fem.nodes if n.id == item_id), None)
            self._populate_properties(node)
            
        elif item_type == "load":
            self.selected_load_id = item_id
            self.selected_beam_id = None
            self.selected_node_id = None
            load = next((l for l in self.fem.loads if l.id == item_id), None)
            self._populate_properties(load)
        
        self._highlight_list()
        self.diag.set_beam(self.selected_beam_id)
        self.update()

    def _refresh_lists(self):
        self.list_widget.clear()
        
        # Nodes section
        node_item = QListWidgetItem("NODES")
        node_item.setBackground(QColor(200, 200, 200))
        node_item.setFlags(node_item.flags() & ~Qt.ItemIsSelectable)
        self.list_widget.addItem(node_item)
        
        for node in self.fem.nodes:
            support_info = f" - {node.support_type}" if node.support_type != "none" else ""
            it = QListWidgetItem(f"  {node.name} ({node.x:.1f}, {node.y:.1f}){support_info}")
            it.setData(Qt.UserRole, ("node", node.id))
            self.list_widget.addItem(it)
        
        # Beams section with child loads
        beam_item = QListWidgetItem("BEAMS")
        beam_item.setBackground(QColor(200, 200, 200))
        beam_item.setFlags(beam_item.flags() & ~Qt.ItemIsSelectable)
        self.list_widget.addItem(beam_item)
        
        for beam in self.fem.beams:
            # Beam item
            it = QListWidgetItem(f"  {beam.name}: Node {beam.start} → Node {beam.end}")
            it.setData(Qt.UserRole, ("beam", beam.id))
            self.list_widget.addItem(it)
            
            # Child loads for this beam
            beam_loads = [l for l in self.fem.loads if l.beam_id == beam.id]
            for load in beam_loads:
                if load.type == 'force':
                    ref = "beam" if load.is_relative_to_beam else "global"
                    child_it = QListWidgetItem(f"    └─ {load.name}: {load.magnitude:.0f}N @ {load.direction}° ({ref})")
                else:  # moment
                    child_it = QListWidgetItem(f"    └─ {load.name}: {load.magnitude:.0f}N·m")
                child_it.setData(Qt.UserRole, ("load", load.id))
                self.list_widget.addItem(child_it)
        
        # Node loads section (loads attached directly to nodes)
        node_loads_item = QListWidgetItem("NODE LOADS")
        node_loads_item.setBackground(QColor(200, 200, 200))
        node_loads_item.setFlags(node_loads_item.flags() & ~Qt.ItemIsSelectable)
        self.list_widget.addItem(node_loads_item)
        
        node_loads = [l for l in self.fem.loads if l.node_id and not l.beam_id]
        for load in node_loads:
            if load.type == 'force':
                ref = "beam" if load.is_relative_to_beam else "global"
                it = QListWidgetItem(f"  {load.name}: {load.magnitude:.0f}N @ {load.direction}° ({ref})")
            else:  # moment
                it = QListWidgetItem(f"  {load.name}: {load.magnitude:.0f}N·m")
            it.setData(Qt.UserRole, ("load", load.id))
            self.list_widget.addItem(it)
    def _highlight_list(self):
        self.list_widget.clearSelection()
        
        # Find and select the corresponding list item
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if not data:
                continue
                
            item_type, item_id = data
            
            if (item_type == "beam" and item_id == self.selected_beam_id or
                item_type == "node" and item_id == self.selected_node_id or
                item_type == "load" and item_id == self.selected_load_id):
                self.list_widget.setCurrentRow(i)
                item.setSelected(True)
            break

    def _point_to_segment_dist_and_pos(self, px, py, x1, y1, x2, y2):
        """Calculate distance to segment and position along segment (0-1)"""
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        L2 = vx*vx + vy*vy
        if L2 == 0:
            return math.hypot(px - x1, py - y1), 0.0
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
        projx, projy = x1 + t * vx, y1 + t * vy
        return math.hypot(px - projx, py - projy), t


# ------------------------------------------------------------
# Main window
# ------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Analysis - Structural Engineering Tool")
        self.resize(1400, 900)

        # Create model
        self.fem = FEMSystem()

        # Create UI components
        self._create_docks()
        self._create_central_widget()
        self._create_toolbar()
        self._create_menu()
        self._create_status_bar()

    def _create_docks(self):
        # Elements list dock
        self.list_dock = QDockWidget("Beam List", self)
        self.list_widget = QListWidget()
        self.list_dock.setWidget(self.list_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.list_dock)

        # Properties dock
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_table = QTableWidget(0, 2)
        self.prop_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.prop_dock.setWidget(self.prop_table)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)

        # Diagrams dock
        self.diagram_dock = QDockWidget("Internal Force Diagrams", self)
        self.diagram = DiagramWidget(self.fem)
        self.diagram_dock.setWidget(self.diagram)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.diagram_dock)

    def _create_central_widget(self):
        # Central canvas
        self.canvas = Canvas(self.fem, self.diagram, self.list_widget, self.prop_table)
        self.setCentralWidget(self.canvas)

    def _create_toolbar(self):
        toolbar = QToolBar("Tools")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Tool actions
        tools = [
            ("Select", 'select', "Select elements", Qt.Key_Escape),
            ("Node", 'node', "Place nodes", Qt.Key_N),
            ("Beam", 'add_beam', "Draw beams between nodes", Qt.Key_B),
            ("Pinned Support", 'support_pinned', "Add pinned support", None),
            ("Roller Support", 'support_roller', "Add roller support", None),
            ("Force", 'add_force', "Add force load", Qt.Key_F),
            ("Moment", 'add_moment', "Add moment load", Qt.Key_M),
            ("Pan", 'pan', "Pan view", None),
        ]

        self.tool_actions = {}
        for name, mode, tip, shortcut in tools:
            action = QAction(name, self)
            action.setToolTip(tip)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(lambda checked, m=mode: self._set_mode(m))
            toolbar.addAction(action)
            self.tool_actions[mode] = action

        toolbar.addSeparator()
        
        # Solve action
        solve_action = QAction("Solve", self)
        solve_action.setToolTip("Run analysis (S)")
        solve_action.setShortcut(Qt.Key_S)
        solve_action.triggered.connect(self._solve)
        toolbar.addAction(solve_action)

    def _create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self._new, "Ctrl+N")
        file_menu.addAction("Open...", self._open, "Ctrl+O")
        file_menu.addAction("Save...", self._save, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Delete Selected", self._delete_selected, "Del")
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Instructions", self._show_instructions)
        help_menu.addAction("About", self._show_about)

    def _create_status_bar(self):
        self.status = self.statusBar()
        self.status.showMessage("Ready - Use toolbar or shortcuts: N=Node, B=Beam, F=Force, M=Moment, S=Solve, ESC=Select")

    def _set_mode(self, mode: str):
        self.canvas.mode = mode
        # Update toolbar button states
        for m, action in self.tool_actions.items():
            action.setChecked(m == mode)
        self.status.showMessage(f"Mode: {mode}")

    def _solve(self):
        self.canvas._solve()

    def _delete_selected(self):
        self.canvas._delete_selected()

    def _new(self):
        reply = QMessageBox.question(self, "New Project", 
                                   "Create new project? Unsaved changes will be lost.",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.fem.nodes.clear()
            self.fem.beams.clear() 
            self.fem.loads.clear()
            self.fem.results = Results()
            self.canvas.selected_beam_id = None
            self.canvas.selected_node_id = None
            self.canvas.update()
            self.canvas._refresh_lists()
            self.diagram.set_beam(None)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Beam Project (*.json)")
        if not path:
            return
        data = {
            'nodes': [n.__dict__ for n in self.fem.nodes],
            'beams': [b.__dict__ for b in self.fem.beams],
            'loads': [l.__dict__ for l in self.fem.loads],
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.status.showMessage(f"Project saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Beam Project (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.fem.nodes = [Node(**n) for n in data.get('nodes', [])]
            self.fem.beams = [Beam(**b) for b in data.get('beams', [])]
            self.fem.loads = [Load(**l) for l in data.get('loads', [])]
            self.canvas._refresh_lists()
            self.canvas.selected_beam_id = None
            self.canvas.selected_node_id = None
            self.canvas.update()
            self.diagram.set_beam(None)
            self.status.showMessage(f"Project loaded: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))

    def _show_instructions(self):
        instructions = """
        Beam Analysis Software - Instructions
        
        WORKFLOW:
        1. Place nodes using the Node tool (N)
        2. Connect nodes with beams using Beam tool (B)  
        3. Add supports (pinned or roller) at support points
        4. Apply loads (forces or moments)
        5. Run analysis with Solve (S)
        6. Review internal force diagrams
        
        TOOLS:
        - Select (ESC): Select and edit elements
        - Node (N): Place connection points
        - Beam (B): Draw beams between nodes
        - Supports: Add pinned or roller constraints
        - Force (F): Apply force loads
        - Moment (M): Apply moment loads
        - Pan: Drag view with middle mouse button
        
        VIEW:
        - Zoom: Mouse wheel
        - Pan: Middle mouse drag or Pan tool
        - Grid: Shows for reference
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Instructions")
        msg.setText(instructions)
        msg.exec_()

    def _show_about(self):
        QMessageBox.about(self, "About Beam Analysis", 
                         "Beam Analysis Software\n\n"
                         "A structural engineering tool for analyzing beam structures.\n"
                         "Supports static analysis of 2D frame structures with various "
                         "loads and boundary conditions.")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Beam Analysis")
    app.setApplicationVersion("1.0")
    
    w = MainWindow()
    w.show()
    return app.exec_()

if __name__ == '__main__':
    main()
