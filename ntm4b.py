import pygame
import numpy as np
import math
import sys
import os
from scipy.linalg import solve
from pygame.locals import *

# Constants and Colors
COLORS = {
    'background': (240, 240, 240),
    'beam': (80, 80, 80),
    'force': (255, 0, 0),
    'moment': (0, 0, 255),
    'support_pinned': (0, 100, 0),
    'support_roller': (0, 0, 100),
    'selected': (255, 255, 0),
    'text': (60, 60, 60),
    'deformed': (255, 150, 150),
    'diagram_N': (200, 0, 0),
    'diagram_V': (0, 200, 0),
    'diagram_M': (0, 0, 200),
    'dialog_bg': (245, 245, 245),
    'panel': (160, 160, 160),
    'button': (220, 220, 220),
    'button_hover': (200, 200, 255),
    'grid': (200, 200, 200),
    'grid_highlight': (180, 180, 180),
    'element_list': (220, 220, 220),
    'list_header': (80, 80, 80),
    'list_item': (240, 240, 240),
    'gradient_top': (30, 60, 120),
    'gradient_bottom': (80, 130, 200),
    'tree_bg': (40, 40, 60),
    'tree_item': (70, 70, 100),
    'panel': (50, 50, 50),  # Dark gray background
    'tool_text': (255, 255, 255),  # White text
    'tool_selected': (0, 100, 200),  # Blue selection
}

UI_PANEL_WIDTH = 80
PROP_PANEL_WIDTH = 300
DIAGRAM_HEIGHT = 250

class ResolutionDialog:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Select Resolution")
        self.clock = pygame.time.Clock()
        self.resolutions = [
            (800, 600), (1280, 720),
            (1600, 900), (1920, 1080)
        ]
        self.selected_res = None
        self.button_width = 200
        self.button_height = 40
        self.button_spacing = 15

    def draw(self):
        self.screen.fill(COLORS['dialog_bg'])
        title_font = pygame.font.SysFont('Arial', 28, bold=True)
        title_text = title_font.render("Select Resolution", True, COLORS['text'])
        title_rect = title_text.get_rect(center=(200, 40))
        self.screen.blit(title_text, title_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        button_font = pygame.font.SysFont('Arial', 22)
        
        for i, res in enumerate(self.resolutions):
            btn_x = (400 - self.button_width) // 2
            btn_y = 80 + i * (self.button_height + self.button_spacing)
            btn_rect = pygame.Rect(btn_x, btn_y, self.button_width, self.button_height)
            
            bg_color = COLORS['button_hover'] if btn_rect.collidepoint(mouse_pos) else COLORS['button']
            pygame.draw.rect(self.screen, bg_color, btn_rect, border_radius=5)
            
            res_text = button_font.render(f"{res[0]} × {res[1]}", True, COLORS['text'])
            text_rect = res_text.get_rect(center=btn_rect.center)
            self.screen.blit(res_text, text_rect)

    def run(self):
        while not self.selected_res:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    for i, res in enumerate(self.resolutions):
                        btn_x = (400 - self.button_width) // 2
                        btn_y = 80 + i * (self.button_height + self.button_spacing)
                        btn_rect = pygame.Rect(btn_x, btn_y, self.button_width, self.button_height)
                        if btn_rect.collidepoint(mouse_pos):
                            self.selected_res = res
                            
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()
        return self.selected_res
    
class TreeNode:
    def __init__(self, name, parent=None, data_type=None, expanded=True):  # Fixed constructor
        self.name = name
        self.parent = parent
        self.children = []
        self.expanded = expanded
        self.data_type = data_type  # Now properly initialized
        self.icon = None

class StructureTree:
    def __init__(self, editor):
        self.editor = editor
        self.root = TreeNode("Construction")
        self.build_tree()
        self.scroll = 0
        self.item_height = 25
        self.indent = 20
        
    def build_tree(self):
        geometric = TreeNode("Geometric Parameters", self.root)
        geometric.children = [
            TreeNode("Points", parent=geometric, data_type="points"),  # Fixed parameters
            TreeNode("Coordinate Systems", parent=geometric, data_type="coord_sys")
        ]
        
        elements = TreeNode("Elements", self.root)
        elements.children = [
            TreeNode("Beams", parent=elements, data_type="beams"),
            TreeNode("Forces", parent=elements, data_type="forces"),
            TreeNode("Moments", parent=elements, data_type="moments"),
            TreeNode("Supports", parent=elements, data_type="supports")
        ]
        
        self.root.children = [geometric, elements]

    def draw(self, surface):
        width = PROP_PANEL_WIDTH
        height = self.editor.screen_size[1]
        
        # Draw gradient background
        self.draw_gradient(surface, (0, 0, width, height), 
                         COLORS['gradient_top'], COLORS['gradient_bottom'])
        
        y = 10 - self.scroll
        self.draw_node(self.root, surface, 10, y, 0)
        
    def draw_gradient(self, surface, rect, top_color, bottom_color):
        """Draw vertical gradient"""
        # Convert tuple to Rect if needed
        if not isinstance(rect, pygame.Rect):
            rect = pygame.Rect(rect)
            
        for y in range(rect.top, rect.bottom):
            ratio = (y - rect.top) / rect.height
            r = top_color[0] + (bottom_color[0] - top_color[0]) * ratio
            g = top_color[1] + (bottom_color[1] - top_color[1]) * ratio
            b = top_color[2] + (bottom_color[2] - top_color[2]) * ratio
            pygame.draw.line(surface, (int(r), int(g), int(b)), 
                           (rect.left, y), (rect.right, y))

    def draw_node(self, node, surface, x, y, level):
        rect = pygame.Rect(x + level*self.indent, y, 
                         PROP_PANEL_WIDTH - x - level*self.indent, self.item_height)
        
        # Highlight if containing current element
        is_selected = (self.editor.selected_element and 
                      self.element_matches_node(self.editor.selected_element, node))
        
        bg_color = COLORS['selected'] if is_selected else COLORS['tree_item']
        pygame.draw.rect(surface, bg_color, rect.inflate(-2, 0), border_radius=3)
        
        # Draw expand/collapse toggle
        if node.children:
            toggle = "+" if not node.expanded else "-"
            toggle_surf = self.editor.font.render(toggle, True, (255,255,255))
            surface.blit(toggle_surf, (rect.left + 5, y + 5))
        
        # Draw icon and text
        text_x = rect.left + (25 if node.children else 10)
        text_surf = self.editor.font.render(node.name, True, (255,255,255))
        surface.blit(text_surf, (text_x, y + 5))
        
        y += self.item_height
        if node.expanded:
            for child in node.children:
                y = self.draw_node(child, surface, x, y, level + 1)
        return y

    def element_matches_node(self, element, node):
        if node.data_type == "points" and isinstance(element, Point):
            return True
        if node.data_type == "beams" and isinstance(element, Beam):
            return True
        if node.data_type == "forces" and isinstance(element, Force):
            return True
        if node.data_type == "supports" and isinstance(element, Support):
            return True
        return False

class Point:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.name = f"Point_{id(self)}"

class Beam:
    def __init__(self, start_point, end_point, shape_func="lambda x: 0"):
        self.start = start_point
        self.end = end_point
        self.shape_func = shape_func
        self.name = f"Beam_{id(self)}"
        self.profile = "Rectangle"
        self.material = "Steel"

class PropertyPanel:
    def __init__(self, editor):
        self.editor = editor
        self.font = pygame.font.SysFont('Arial', 16)
        
    def draw(self, surface, rect):
        pygame.draw.rect(surface, COLORS['dialog_bg'], rect)
        if not self.editor.selected_element:
            return
            
        y = rect.top + 10
        element = self.editor.selected_element
        
        if isinstance(element, Point):
            self.draw_property(surface, "X:", f"{element.x:.2f} m", rect.left+10, y)
            y += 25
            self.draw_property(surface, "Y:", f"{element.y:.2f} m", rect.left+10, y)
            y += 25
            self.draw_property(surface, "Z:", f"{element.z:.2f} m", rect.left+10, y)
            
        elif isinstance(element, Beam):
            self.draw_property(surface, "Start:", element.start.name, rect.left+10, y)
            y += 25
            self.draw_property(surface, "End:", element.end.name, rect.left+10, y)
            y += 25
            self.draw_property(surface, "Length:", f"{self.calc_length(element):.2f} m", rect.left+10, y)
            y += 25
            self.draw_editable_field(surface, "Shape Function:", element.shape_func, rect.left+10, y)
            
        elif isinstance(element, Support):
            self.draw_property(surface, "Type:", element.type, rect.left+10, y)
            y += 25
            self.draw_property(surface, "Location:", f"{element.x:.2f} m", rect.left+10, y)

    def draw_property(self, surface, label, value, x, y):
        label_surf = self.font.render(label, True, COLORS['text'])
        value_surf = self.font.render(value, True, (0,0,0))
        surface.blit(label_surf, (x, y))
        surface.blit(value_surf, (x + 100, y))

    def draw_editable_field(self, surface, label, value, x, y):
        btn_rect = pygame.Rect(x + 100, y, 150, 25)
        pygame.draw.rect(surface, (255,255,255), btn_rect, 0)
        pygame.draw.rect(surface, (0,0,0), btn_rect, 1)
        
        # Add actual text input handling
        if self.editor.selected_element == element:
            text = self.font.render(str(value), True, (0,0,0))
            surface.blit(text, (x + 105, y + 5))

    def handle_input(self, event):
        if isinstance(self.editor.selected_element, Beam):
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    try:
                        self.editor.selected_element.shape_func = self.current_text
                        self.editor.update_geometry()
                    except:
                        print("Invalid function")
                else:
                    # Handle text input
                    pass


class BeamElement:
    def __init__(self, node1, node2, E=2e11, I=1e-6, A=1e-4):
        self.nodes = [node1, node2]
        self.E = E
        self.I = I
        self.A = A
        self.L = abs(node2.x - node1.x)
        self.k = self.calculate_stiffness_matrix()
        

    def calculate_stiffness_matrix(self):
        L = self.L
        E, A, I = self.E, self.A, self.I
        return np.array([
            [E*A/L, 0, 0, -E*A/L, 0, 0],
            [0, 12*E*I/L**3, 6*E*I/L**2, 0, -12*E*I/L**3, 6*E*I/L**2],
            [0, 6*E*I/L**2, 4*E*I/L, 0, -6*E*I/L**2, 2*E*I/L],
            [-E*A/L, 0, 0, E*A/L, 0, 0],
            [0, -12*E*I/L**3, -6*E*I/L**2, 0, 12*E*I/L**3, -6*E*I/L**2],
            [0, 6*E*I/L**2, 2*E*I/L, 0, -6*E*I/L**2, 4*E*I/L]
        ])

class Node:
    def __init__(self, x, y=0):
        self.x = x
        self.y = y
        self.dofs = np.zeros(3)
        self.constrained = [False, False, False]

class FEModel:
    def __init__(self, length=5.0, num_elements=20, shape_func=lambda x: 0):
        self.length = length
        self.num_elements = num_elements
        self.shape_func = shape_func
        self.nodes = []
        self.elements = []
        self.K = None
        self.F = None
        self.U = None
        self.create_mesh()

    def create_mesh(self):
        dx = self.length / self.num_elements
        self.nodes = [Node(i*dx, self.shape_func(i*dx)) 
                     for i in range(self.num_elements+1)]
        self.elements = [BeamElement(self.nodes[i], self.nodes[i+1])
                        for i in range(self.num_elements)]

    def assemble_global_stiffness(self):
        total_dof = 3 * len(self.nodes)
        self.K = np.zeros((total_dof, total_dof))
        for elem in self.elements:
            for i in range(2):
                n1 = elem.nodes[i]
                idx1 = self.nodes.index(n1)*3
                for j in range(2):
                    n2 = elem.nodes[j]
                    idx2 = self.nodes.index(n2)*3
                    self.K[idx1:idx1+3, idx2:idx2+3] += elem.k[3*i:3*(i+1), 3*j:3*(j+1)]

    def apply_boundary_conditions(self):
        for node in self.nodes:
            if any(node.constrained):
                dofs = [i for i, val in enumerate(node.constrained) if val]
                global_dof = self.nodes.index(node)*3 + np.array(dofs)
                for dof in global_dof:
                    self.K[dof,:] = 0
                    self.K[:,dof] = 0
                    self.K[dof,dof] = 1e20

    def solve(self):
        self.assemble_global_stiffness()
        self.apply_boundary_conditions()
        
        # Improved regularization
        np.fill_diagonal(self.K, self.K.diagonal() + np.max(self.K) * 1e-6)
        
        try:
            self.U = solve(self.K, self.F, assume_a='pos')
        except np.linalg.LinAlgError as e:
            print(f"Solution failed: {str(e)}")
            print("Check boundary conditions and supports")
            self.U = np.zeros_like(self.F)

    def calculate_internal_forces(self):
        N,V,M = [],[],[]
        for elem in self.elements:
            n1_idx = self.nodes.index(elem.nodes[0])
            u = self.U[n1_idx*3:n1_idx*3+6]
            f_local = elem.k @ u
            N.append(f_local[0])
            V.append(f_local[1])
            M.append(f_local[2])
        return N, V, M

class StructuralElement:
    def __init__(self, x_pos):
        self.x = x_pos
        self.selected = False

class Force(StructuralElement):
    def __init__(self, x_pos, magnitude=100, angle=90):
        super().__init__(x_pos)
        self.magnitude = magnitude
        self.angle = angle

    def get_components(self):
        rad = math.radians(self.angle)
        return (self.magnitude * math.cos(rad), self.magnitude * math.sin(rad))

class Support(StructuralElement):
    def __init__(self, x_pos, support_type='pinned'):
        super().__init__(x_pos)
        self.type = support_type

class BeamEditor:
    def __init__(self, screen_size):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size, FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        self.tool_icons = self.create_tool_icons()
        self.screen_size = screen_size
        self.model = FEModel(shape_func=lambda x: 0.1 * math.sin(x))
        self.scale = 100  # pixels/meter
        self.offset_x = 400
        self.beam_y = screen_size[1] // 2
        self.elements = []
        self.selected_element = None
        self.support_type = 'pinned'
        self.show_deformed = False
        self.deformation_scale = 5000
        self.file_dialog_active = False
        self.current_file = ""
        self.mode = 'select'  # Add this line
        self.init_ui()
        self.init_defaults()
        self.update_FEM()
        
        self.grid_size = 50  # pixels between grid lines
        self.element_list_scroll = 0
        self.element_icons = {
            'beam': pygame.Surface((20, 20)),
            'force': pygame.Surface((20, 20)),
            'moment': pygame.Surface((20, 20)),
            'support': pygame.Surface((20, 20))
        }
        self.init_element_icons()
        self.points = [Point(0, 0), Point(5, 0)]
        self.beams = [Beam(self.points[0], self.points[1])]
        self.structure_tree = StructureTree(self)
        self.property_panel = PropertyPanel(self)
        

    def init_ui(self):
        self.tools = [
            ('select', 'S', (30, 30)),
            ('force', 'F', (30, 80)),
            ('moment', 'M', (30, 130)),
            ('support', 'P', (30, 180)),
            ('deformed', 'D', (30, 230)),
            ('save', 'S', (30, 280)),
            ('open', 'O', (30, 330))
        ]

    def init_defaults(self):
        pinned = Support(0, 'pinned')
        roller = Support(self.model.length, 'roller')
        self.elements.extend([pinned, roller])
        self.apply_support_constraints(pinned)
        self.apply_support_constraints(roller)

    def init_element_icons(self):
        # Create simple icons for element types
        beam_icon = self.element_icons['beam']
        pygame.draw.line(beam_icon, COLORS['beam'], (0, 10), (20, 10), 3)
        
        force_icon = self.element_icons['force']
        pygame.draw.line(force_icon, COLORS['force'], (10, 0), (10, 20), 3)
        pygame.draw.polygon(force_icon, COLORS['force'], [(10,3), (7,10), (13,10)])
        
        moment_icon = self.element_icons['moment']
        pygame.draw.circle(moment_icon, COLORS['moment'], (10, 10), 8, 2)
        
        support_icon = self.element_icons['support']
        pygame.draw.polygon(support_icon, COLORS['support_pinned'], 
                          [(0,20), (10,0), (20,20)])
        
    def draw_structure_panel(self):
        panel_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH, 0, 
                               PROP_PANEL_WIDTH, self.screen_size[1])
        self.structure_tree.draw(self.screen)
        
        # Property panel at top
        prop_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH, 0,
                              PROP_PANEL_WIDTH, 200)
        self.property_panel.draw(self.screen, prop_rect)

    def handle_tree_click(self, pos):
        # Convert click position to tree item
        rel_y = pos[1] + self.structure_tree.scroll - 10
        item_index = rel_y // self.structure_tree.item_height
        node = self.find_clicked_node(self.structure_tree.root, item_index)
        
        if node:
            if node.data_type:
                self.select_first_element_of_type(node.data_type)
            elif node.children:
                node.expanded = not node.expanded

    def select_first_element_of_type(self, data_type):
        if data_type == "points":
            self.selected_element = self.points[0] if self.points else None
        elif data_type == "beams":
            self.selected_element = self.beams[0] if self.beams else None
    def draw_grid(self):
        width, height = self.screen_size
        offset_x = self.offset_x % self.grid_size
        offset_y = self.beam_y % self.grid_size
        
        # Draw vertical grid lines
        for x in range(-self.grid_size, width + self.grid_size, self.grid_size):
            px = x - offset_x
            color = COLORS['grid_highlight'] if x % (self.grid_size*5) == 0 else COLORS['grid']
            pygame.draw.line(self.screen, color, (px, 0), (px, height), 1)
            
        # Draw horizontal grid lines
        for y in range(-self.grid_size, height + self.grid_size, self.grid_size):
            py = y - offset_y + self.beam_y
            color = COLORS['grid_highlight'] if y % (self.grid_size*5) == 0 else COLORS['grid']
            pygame.draw.line(self.screen, color, (0, py), (width, py), 1)
            
        # Draw origin marker
        origin_x = self.offset_x
        origin_y = self.beam_y
        pygame.draw.circle(self.screen, COLORS['text'], (origin_x, origin_y), 5)
        font = pygame.font.SysFont('Arial', 14)
        text = font.render("(0,0)", True, COLORS['text'])
        self.screen.blit(text, (origin_x + 10, origin_y - 10))

    def show_context_menu(self, pos):
        menu_items = [
            ("Add Point", lambda: self.create_new_point(pos)),
            ("Add Beam", self.start_beam_creation),
            ("Add Force", lambda: self.set_creation_mode('force'))
        ]
    def create_new_point(self, screen_pos):
        x = (screen_pos[0] - self.offset_x) / self.scale
        y = (self.beam_y - screen_pos[1]) / self.scale
        self.points.append(Point(x, y))
        self.update_geometry()
    def draw_element_list(self):
        panel_width = PROP_PANEL_WIDTH - 20
        panel_height = self.screen_size[1] - 150  # Leave space for properties
        
        # Main panel
        pygame.draw.rect(self.screen, COLORS['element_list'], 
                       (self.screen_size[0]-PROP_PANEL_WIDTH, 40, 
                        PROP_PANEL_WIDTH, panel_height))
        
        # Scrollable content area
        content_height = len(self.elements) * 30
        visible_height = panel_height - 20
        
        # Only draw visible items
        start_index = max(0, self.element_list_scroll // 30)
        end_index = min(len(self.elements), start_index + (visible_height // 30) + 1)
        
        y = 50 - (self.element_list_scroll % 30)
        for i in range(start_index, end_index):
            elem = self.elements[i]
            item_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH+10, y, 
                                  panel_width-20, 25)
            
            # Selection background
            if elem == self.selected_element:
                pygame.draw.rect(self.screen, COLORS['selected'], item_rect, border_radius=3)
            
            # Element text
            elem_type = 'Support' if isinstance(elem, Support) else 'Force'
            text = f"{elem_type} @ {elem.x:.2f}m"
            if isinstance(elem, Support):
                text += f" ({elem.type})"
            text_surf = self.font.render(text, True, COLORS['text'])
            self.screen.blit(text_surf, (item_rect.x+5, item_rect.y+5))
            
            y += 30



    def update_FEM(self):
        self.model.F = np.zeros(3*len(self.model.nodes))
        for elem in self.elements:
            if isinstance(elem, Force):
                node_idx = self.find_nearest_node(elem.x)
                fx, fz = elem.get_components()
                self.model.F[node_idx*3] += fx
                self.model.F[node_idx*3+1] += fz
        self.model.solve()

    def find_nearest_node(self, x):
        return min(range(len(self.model.nodes)), key=lambda i: abs(self.model.nodes[i].x - x))

    def screen_to_beam_x(self, screen_x):
        return (screen_x - self.offset_x) / self.scale

    def get_beam_position(self, x):
        screen_x = self.offset_x + x * self.scale
        screen_y = self.beam_y + self.model.shape_func(x) * 50
        return (screen_x, screen_y)

    def draw_beam(self):
        for elem in self.model.elements:
            start = self.get_beam_position(elem.nodes[0].x)
            end = self.get_beam_position(elem.nodes[1].x)
            pygame.draw.line(self.screen, COLORS['beam'], start, end, 6)
            
            if self.show_deformed:
                dx1 = self.deformation_scale * elem.nodes[0].dofs[0]
                dy1 = self.deformation_scale * elem.nodes[0].dofs[1]
                dx2 = self.deformation_scale * elem.nodes[1].dofs[0]
                dy2 = self.deformation_scale * elem.nodes[1].dofs[1]
                pygame.draw.line(self.screen, COLORS['deformed'], 
                               (start[0]+dx1, start[1]+dy1),
                               (end[0]+dx2, end[1]+dy2), 3)

    def draw_diagrams(self):
        N, V, M = self.model.calculate_internal_forces()
        diagram_rect = pygame.Rect(UI_PANEL_WIDTH, self.screen_size[1]-DIAGRAM_HEIGHT,
                                 self.screen_size[0]-UI_PANEL_WIDTH-PROP_PANEL_WIDTH, DIAGRAM_HEIGHT)
        pygame.draw.rect(self.screen, (255,255,255), diagram_rect)
        
        # Normal force diagram
        max_N = max(abs(n) for n in N) or 1
        points = [(diagram_rect.left + 50 + (i/len(N))*(diagram_rect.width-100),
                 diagram_rect.top + DIAGRAM_HEIGHT//6 - (n/max_N)*(DIAGRAM_HEIGHT//6 - 10))
                for i,n in enumerate(N)]
        pygame.draw.lines(self.screen, COLORS['diagram_N'], False, points, 2)
        
        # Shear force diagram
        max_V = max(abs(v) for v in V) or 1
        points = [(diagram_rect.left + 50 + (i/len(V))*(diagram_rect.width-100),
                 diagram_rect.top + DIAGRAM_HEIGHT//2 - (v/max_V)*(DIAGRAM_HEIGHT//6 - 10))
                for i,v in enumerate(V)]
        pygame.draw.lines(self.screen, COLORS['diagram_V'], False, points, 2)
        
        # Bending moment diagram
        max_M = max(abs(m) for m in M) or 1
        points = [(diagram_rect.left + 50 + (i/len(M))*(diagram_rect.width-100),
                 diagram_rect.top + 5*DIAGRAM_HEIGHT//6 - (m/max_M)*(DIAGRAM_HEIGHT//6 - 10))
                for i,m in enumerate(M)]
        pygame.draw.lines(self.screen, COLORS['diagram_M'], False, points, 2)

    def create_tool_icons(self):
        icons = {
            'select': self.create_icon(COLORS['selected']),
            'force': self.create_icon(COLORS['force']),
            'moment': self.create_icon(COLORS['moment']),
            'support': self.create_icon(COLORS['support_pinned']),
            'deformed': self.create_icon(COLORS['deformed']),
            'save': self.create_icon(COLORS['button']),
            'open': self.create_icon(COLORS['button'])
        }
        return icons

    def create_icon(self, color):
        icon = pygame.Surface((30, 30))
        pygame.draw.circle(icon, color, (15, 15), 12)
        return icon

    def draw_toolbar(self):
        # Draw solid background
        pygame.draw.rect(self.screen, COLORS['panel'], 
                       (0, 0, UI_PANEL_WIDTH, self.screen_size[1]))
        
        font = pygame.font.SysFont('Arial', 16, bold=True)
        tools = [
            ("[S] Select", 40),
            ("[F] Force", 80),
            ("[P] Support", 120),
            ("[D] Deformed", 160),
            ("[Ctrl+S] Save", 200),
            ("[Ctrl+O] Open", 240)
        ]

        for text, y_pos in tools:
            # Draw text with white color
            text_surf = font.render(text, True, COLORS['tool_text'])
            self.screen.blit(text_surf, (10, y_pos))
            
            # Highlight selected tool
            if self.mode == text.split()[0].lower().strip('[]'):
                pygame.draw.rect(self.screen, COLORS['tool_selected'],
                               (5, y_pos-5, UI_PANEL_WIDTH-10, 25))

    def draw_tooltip(self, text, mouse_pos):
        font = pygame.font.SysFont('Arial', 18)
        text_surf = font.render(text, True, COLORS['text'])
        bg_rect = text_surf.get_rect().inflate(10, 5)
        bg_rect.topleft = (mouse_pos[0] + 15, mouse_pos[1] + 15)
        
        if bg_rect.right > self.screen_size[0]:
            bg_rect.right = mouse_pos[0] - 5
        if bg_rect.bottom > self.screen_size[1]:
            bg_rect.bottom = mouse_pos[1] - 5
            
        pygame.draw.rect(self.screen, COLORS['selected'], bg_rect, border_radius=3)
        self.screen.blit(text_surf, bg_rect.move(5, 5))

    def draw_properties_panel(self):
        panel_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH, 
                               self.screen_size[1]-150, 
                               PROP_PANEL_WIDTH, 150)
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect)
        
        if self.selected_element:
            font = pygame.font.SysFont('Arial', 14)
            y = panel_rect.top + 10
            
            if isinstance(self.selected_element, Force):
                self.draw_property("Type:", "Force", panel_rect.left+10, y)
                self.draw_property("Position:", f"{self.selected_element.x:.2f}m", panel_rect.left+10, y+25)
                self.draw_property("Magnitude:", f"{self.selected_element.magnitude}N", panel_rect.left+10, y+50)
                self.draw_property("Angle:", f"{self.selected_element.angle}°", panel_rect.left+10, y+75)
            
            elif isinstance(self.selected_element, Support):
                self.draw_property("Type:", "Support", panel_rect.left+10, y)
                self.draw_property("Position:", f"{self.selected_element.x:.2f}m", panel_rect.left+10, y+25)
                self.draw_property("Constraint:", self.selected_element.type, panel_rect.left+10, y+50)

    def draw_property(self, label, value, x, y):
        font = pygame.font.SysFont('Arial', 14)
        label_surf = font.render(label, True, COLORS['text'])
        value_surf = font.render(value, True, (0, 0, 0))
        self.screen.blit(label_surf, (x, y))
        self.screen.blit(value_surf, (x + 80, y))

    def draw_structure_panel(self):
        # Tree panel (top 2/3 of right side)
        tree_panel_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH, 40,
                                    PROP_PANEL_WIDTH, self.screen_size[1]//2)
        pygame.draw.rect(self.screen, COLORS['tree_bg'], tree_panel_rect)
        self.structure_tree.draw(self.screen)
        
        # Properties panel (bottom 1/3)
        prop_panel_rect = pygame.Rect(self.screen_size[0]-PROP_PANEL_WIDTH,
                                    self.screen_size[1]//2 + 40,
                                    PROP_PANEL_WIDTH, self.screen_size[1]//2 - 40)
        self.property_panel.draw(self.screen, prop_panel_rect)
    def handle_right_click(self):
        if event.type == MOUSEBUTTONDOWN and event.button == 3:
            self.show_context_menu(pygame.mouse.get_pos())
    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Handle toolbar clicks
                    if mouse_pos[0] < UI_PANEL_WIDTH:
                        y = mouse_pos[1]
                        if 15 < y < 45: self.mode = 'select'
                        elif 65 < y < 95: self.mode = 'force'
                        elif 115 < y < 145: self.mode = 'support'
                        elif 165 < y < 195: self.show_deformed = not self.show_deformed
                        elif 215 < y < 245: self.save_project("autosave.ntmd")
                        elif 265 < y < 295: self.load_project("autosave.ntmd")
                    
                    # Handle main workspace clicks
                    elif UI_PANEL_WIDTH < mouse_pos[0] < self.screen_size[0]-PROP_PANEL_WIDTH:
                        beam_x = self.screen_to_beam_x(mouse_pos[0])
                        if self.mode == 'force':
                            new_force = Force(beam_x, 1000, 270)  # Downward force
                            self.elements.append(new_force)
                            self.selected_element = new_force
                            self.update_FEM()
                        elif self.mode == 'support':
                            new_support = Support(beam_x, self.support_type)
                            self.elements.append(new_support)
                            self.apply_support_constraints(new_support)
                            self.selected_element = new_support
                            self.update_FEM()

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.selected_element = None
                elif event.key == K_t and isinstance(self.selected_element, Support):
                    self.support_type = 'roller' if self.support_type == 'pinned' else 'pinned'
                    self.update_FEM()
                elif self.selected_element:
                    if isinstance(self.selected_element, Force):
                        if event.key == K_LEFT:
                            self.selected_element.angle = (self.selected_element.angle - 5) % 360
                        elif event.key == K_RIGHT:
                            self.selected_element.angle = (self.selected_element.angle + 5) % 360
                        self.update_FEM()

    def apply_support_constraints(self, support):
        node_idx = self.find_nearest_node(support.x)
        node = self.model.nodes[node_idx]
    
        if support.type == 'pinned':
            # Constrain translations (u, v)
            node.constrained = [True, True, False]
        elif support.type == 'roller':
            # Constrain vertical translation only (v)
            node.constrained = [False, True, False]
            
        print(f"Applied {support.type} constraint at node {node_idx} (x={node.x:.2f})")

    def save_project(self, filename):
        with open(filename, 'w') as f:
            f.write(f"BeamLength,{self.model.length}\n")
            for elem in self.elements:
                if isinstance(elem, Force):
                    f.write(f"Force,{elem.x},{elem.magnitude},{elem.angle}\n")
                elif isinstance(elem, Support):
                    f.write(f"Support,{elem.x},{elem.type}\n")

    def load_project(self, filename):
        self.elements = []
        self.model = FEModel()
        with open(filename, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if parts[0] == "BeamLength":
                    self.model.length = float(parts[1])
                elif parts[0] == "Force":
                    x = float(parts[1])
                    mag = float(parts[2])
                    angle = float(parts[3])
                    self.elements.append(Force(x, mag, angle))
                elif parts[0] == "Support":
                    x = float(parts[1])
                    s_type = parts[2]
                    support = Support(x, s_type)
                    self.elements.append(support)
                    self.apply_support_constraints(support)
        self.update_FEM()

    def draw_elements(self):
        for element in self.elements:
            if isinstance(element, Force):
                self.draw_force(element)
            elif isinstance(element, Support):
                self.draw_support(element)
            
            # Draw selection indicator
            if element == self.selected_element:
                x = self.offset_x + element.x * self.scale
                y = self.beam_y + self.model.shape_func(element.x) * 50
                pygame.draw.circle(self.screen, COLORS['selected'], (int(x), int(y)), 8)

    def draw_force(self, force):
        scale = self.scale
        offset_x = self.offset_x
        beam_y = self.beam_y + self.model.shape_func(force.x) * 50
        
        angle_rad = math.radians(force.angle)
        length = force.magnitude * 0.2
        end_x = force.x * scale + offset_x + length * math.cos(angle_rad)
        end_y = beam_y + length * math.sin(angle_rad)
        
        color = COLORS['selected'] if force.selected else COLORS['force']
        pygame.draw.line(self.screen, color, 
                        (force.x * scale + offset_x, beam_y),
                        (end_x, end_y), 3)
        pygame.draw.polygon(self.screen, color, [
            (end_x + 8 * math.cos(angle_rad + math.pi/2),
             end_y + 8 * math.sin(angle_rad + math.pi/2)),
            (end_x + 8 * math.cos(angle_rad - math.pi/2),
             end_y + 8 * math.sin(angle_rad - math.pi/2)),
            (end_x, end_y)
        ])

    def draw_support(self, support):
        scale = self.scale
        offset_x = self.offset_x
        beam_y = self.beam_y + self.model.shape_func(support.x) * 50
        screen_x = support.x * scale + offset_x
        
        color = COLORS['selected'] if support.selected else COLORS[f'support_{support.type}']
        
        if support.type == 'pinned':
            base_y = beam_y + 20
            pygame.draw.polygon(self.screen, color, [
                (screen_x - 15, base_y),
                (screen_x + 15, base_y),
                (screen_x, base_y + 20)
            ])
        else:
            pygame.draw.circle(self.screen, color, (screen_x, beam_y + 15), 10)
            pygame.draw.line(self.screen, color, 
                            (screen_x - 15, beam_y), 
                            (screen_x + 15, beam_y), 3)
    def update_geometry(self):
        # Rebuild FEM model based on points and beams
        pass
    def run(self):
        while True:
            # Handle events first
            self.handle_events()
            
            # Update FEM model if needed
            if any(element.selected for element in self.elements):
                self.update_FEM()
            
            # Clear screen
            self.screen.fill(COLORS['background'])
            
            # Draw grid and main beam first
            self.draw_grid()
            self.draw_beam()
            
            # Draw structural elements
            self.draw_elements()
            
            # Draw diagrams above beam but below UI panels
            self.draw_diagrams()
            
            # Draw UI panels last (on top)

            self.draw_structure_panel()
            self.draw_properties_panel()
            self.draw_toolbar()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS

            # Debug print (optional)
            if False:  # Set to True for debugging
                print(f"Mode: {self.mode} | Selected: {self.selected_element} | Elements: {len(self.elements)}")

if __name__ == "__main__":
    try:
        pygame.init()
        if len(sys.argv) > 1 and sys.argv[1].endswith('.ntmd'):
            editor = BeamEditor((1600, 900))
            editor.load_project(sys.argv[1])
            editor.run()
        else:
            res_dialog = ResolutionDialog()
            screen_size = res_dialog.run()
            editor = BeamEditor(screen_size)
            editor.run()
    except Exception as e:
        pygame.quit()
        sys.exit(f"Error: {str(e)}")
