import pygame
import math
import sys
import pickle
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Colors and Constants
COLORS = {
    'background': (240, 240, 240),
    'beam': (80, 80, 80),
    'force': (255, 0, 0),
    'moment': (0, 0, 255),
    'support_pinned': (0, 100, 0),
    'support_roller': (0, 0, 100),
    'selected': (255, 255, 0),
    'text': (0, 0, 0),
    'panel': (200, 200, 200),
    'axis': (150, 150, 150)
}

SCREEN_SIZE = (1600, 1000)
UI_PANEL_WIDTH = 80
PROP_PANEL_WIDTH = 300
DIAGRAM_HEIGHT = 250
BEAM_Y = 600

class StructuralElement:
    def __init__(self, x_pos, value=0):
        self.x = x_pos  # X position along beam (meters)
        self.value = value
        self.selected = False
        
    def draw(self, surface, scale, offset_x):
        pass

class Force(StructuralElement):
    def __init__(self, x_pos, magnitude=100, angle=90):
        super().__init__(x_pos, abs(magnitude))
        self.angle = angle  # Degrees from horizontal (0° = right, 90° = down)
        self.direction = -1 if magnitude < 0 else 1
        
    def get_components(self):
        """Return horizontal (x) and vertical (z) force components"""
        rad = math.radians(self.angle)
        fx = self.value * math.cos(rad) * self.direction
        fz = self.value * math.sin(rad) * self.direction
        return fx, fz

    def draw(self, surface, scale, offset_x):
        screen_x = int(self.x * scale + offset_x)
        length = self.value * 0.2
        color = COLORS['selected'] if self.selected else COLORS['force']
        
        # Calculate arrow direction based on angle
        angle_rad = math.radians(self.angle)
        end_x = screen_x + length * math.cos(angle_rad) * self.direction
        end_y = BEAM_Y + length * math.sin(angle_rad) * self.direction
        
        pygame.draw.line(surface, color, (screen_x, BEAM_Y), (end_x, end_y), 3)
        pygame.draw.polygon(surface, color, [
            (end_x + 5*math.cos(angle_rad + math.pi/2),
             end_y + 5*math.sin(angle_rad + math.pi/2)),
            (end_x + 5*math.cos(angle_rad - math.pi/2),
             end_y + 5*math.sin(angle_rad - math.pi/2)),
            (end_x, end_y)
        ])

class Moment(StructuralElement):
    def __init__(self, x_pos, magnitude=100):
        super().__init__(x_pos, abs(magnitude))
        self.direction = 1 if magnitude >= 0 else -1  # 1 = clockwise, -1 = CCW
        
    def draw(self, surface, scale, offset_x):
        screen_x = int(self.x * scale + offset_x)
        radius = self.value * 0.1
        color = COLORS['selected'] if self.selected else COLORS['moment']
        
        # Draw direction-dependent arrows
        start_angle = 45 if self.direction == 1 else 225
        for i in range(4):
            angle = math.radians(start_angle + 90*i)
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            pygame.draw.line(surface, color, 
                            (screen_x + dx, BEAM_Y + dy),
                            (screen_x + dx*1.5, BEAM_Y + dy*1.5), 2)

class Support(StructuralElement):
    def __init__(self, x_pos, support_type='pinned'):
        super().__init__(x_pos)
        self.type = support_type
        
    def draw(self, surface, scale, offset_x):
        screen_x = int(self.x * scale + offset_x)
        color = COLORS['selected'] if self.selected else COLORS['support_'+self.type]
        
        if self.type == 'pinned':
            pygame.draw.polygon(surface, color, [
                (screen_x - 15, BEAM_Y),
                (screen_x + 15, BEAM_Y),
                (screen_x, BEAM_Y + 20)
            ])
        else:
            pygame.draw.circle(surface, color, (screen_x, BEAM_Y + 10), 10)
            pygame.draw.line(surface, color, 
                            (screen_x - 15, BEAM_Y), 
                            (screen_x + 15, BEAM_Y), 3)

class BeamEditor:
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        
        # Structural properties
        self.beam_length = 5.0  # meters
        self.elements = []
        self.selected_element = None
        self.scale = 100        # pixels/meter
        self.offset_x = 400
        
        # UI state
        self.mode = 'select'  # Initialize mode attribute
        self.support_type = 'pinned'
        self.editing_field = None
        self.editing_value = None
        self.value_str= "-"
        # Diagram data
        self.N = [0] * 100  # Initialize with 100 points
        self.V = [0] * 100
        self.M = [0] * 100
        
        # Initialize default supports
        self.elements.append(Support(0, 'pinned'))
        self.elements.append(Support(self.beam_length, 'roller'))
        
        # Tool icons
        self.tools = [
            ('select', 'S', (30, 30)),
            ('force', 'F', (30, 80)),
            ('moment', 'M', (30, 130)),
            ('support', 'P', (30, 180))
        ]
        
        # Initialize internal forces
        self.calculate_internal_forces()

    def screen_to_beam_x(self, screen_x):
        return (screen_x - self.offset_x) / self.scale
        
    def draw_beam(self):
        start_x = self.offset_x
        end_x = self.offset_x + self.beam_length * self.scale
        pygame.draw.line(self.screen, COLORS['beam'], 
                        (start_x, BEAM_Y), 
                        (end_x, BEAM_Y), 6)
        
    def draw_coordinate_system(self):
        origin = (50, SCREEN_SIZE[1] - 100)
        length = 60
        
        # X axis
        pygame.draw.line(self.screen, COLORS['axis'], origin, (origin[0] + length, origin[1]), 2)
        pygame.draw.polygon(self.screen, COLORS['axis'], [
            (origin[0] + length + 5, origin[1] - 5),
            (origin[0] + length + 5, origin[1] + 5),
            (origin[0] + length + 15, origin[1])
        ])
        
        # Z axis
        pygame.draw.line(self.screen, COLORS['axis'], origin, (origin[0], origin[1] + length), 2)
        pygame.draw.polygon(self.screen, COLORS['axis'], [
            (origin[0] - 5, origin[1] + length + 5),
            (origin[0] + 5, origin[1] + length + 5),
            (origin[0], origin[1] + length + 15)
        ])
        
        font = pygame.font.SysFont('Arial', 14)
        self.screen.blit(font.render('X', True, COLORS['axis']), (origin[0] + length + 20, origin[1] - 10))
        self.screen.blit(font.render('Z', True, COLORS['axis']), (origin[0] - 20, origin[1] + length + 20))
        
    def draw_elements(self):
        for element in self.elements:
            element.draw(self.screen, self.scale, self.offset_x)
            
    def draw_toolbar(self):
        pygame.draw.rect(self.screen, COLORS['panel'], (0, 0, UI_PANEL_WIDTH, SCREEN_SIZE[1]))
        font = pygame.font.SysFont('Arial', 20)
        for i, (mode, key, pos) in enumerate(self.tools):
            color = COLORS['selected'] if mode == self.mode else (220, 220, 220)
            pygame.draw.rect(self.screen, color, (10, pos[1], 60, 40))
            self.screen.blit(font.render(key, True, COLORS['text']), (pos[0], pos[1] + 10))
            
    def draw_properties_panel(self):
        panel_rect = pygame.Rect(SCREEN_SIZE[0]-PROP_PANEL_WIDTH, 0, PROP_PANEL_WIDTH, SCREEN_SIZE[1])
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect)
        font = pygame.font.SysFont('Arial', 20)
        y = 20
        
        # Element list
        self.screen.blit(font.render("Elements:", True, COLORS['text']), (SCREEN_SIZE[0]-280, y))
        y += 30
        for element in self.elements:
            if isinstance(element, Force):
                text = f"Force: {element.value * element.direction:.1f}N {'↓' if element.direction == -1 else '↑'}"
            elif isinstance(element, Moment):
                text = f"Moment: {element.value * element.direction:.1f}Nm {'CW' if element.direction == 1 else 'CCW'}"
            elif isinstance(element, Support):
                text = f"Support: {'Pinned' if element.type == 'pinned' else 'Roller'}"
            self.screen.blit(font.render(text, True, COLORS['text']), (SCREEN_SIZE[0]-280, y))
            y += 30

        # Selected element properties
        if self.selected_element:
            y += 20
            self.screen.blit(font.render("Selected Element:", True, COLORS['text']), (SCREEN_SIZE[0]-280, y))
            y += 40
            
            # Position
            self.screen.blit(font.render(f"X: {self.selected_element.x:.2f}m", True, COLORS['text']), (SCREEN_SIZE[0]-280, y))
            y += 30
            
            # Value editor
            value_text = f"Value: {self.value_str}" if self.editing_value else f"Value: {self.selected_element.value * abs(self.selected_element.direction):.1f}"
            self.screen.blit(font.render(value_text, True, COLORS['text']), (SCREEN_SIZE[0]-280, y))
            y += 30
            
            # Direction control
            if isinstance(self.selected_element, (Force, Moment)):
                btn_rect = pygame.Rect(SCREEN_SIZE[0]-140, y, 120, 30)
                pygame.draw.rect(self.screen, (180, 180, 255), btn_rect)
                if isinstance(self.selected_element, Force):
                    dir_text = "↓ Down" if self.selected_element.direction == -1 else "↑ Up"
                else:
                    dir_text = "↻ CW" if self.selected_element.direction == 1 else "↺ CCW"
                self.screen.blit(font.render(dir_text, True, COLORS['text']), (SCREEN_SIZE[0]-135, y+5))
                y += 40

    def draw_diagrams(self):
        diagram_rect = pygame.Rect(UI_PANEL_WIDTH, SCREEN_SIZE[1]-DIAGRAM_HEIGHT, 
                                 SCREEN_SIZE[0]-UI_PANEL_WIDTH-PROP_PANEL_WIDTH, DIAGRAM_HEIGHT)
        pygame.draw.rect(self.screen, (255, 255, 255), diagram_rect)
        
        # Draw individual diagrams
        self.draw_single_diagram(diagram_rect, self.N, COLORS['force'], "Normal Force (N)")
        self.draw_single_diagram(diagram_rect, self.V, COLORS['force'], "Shear Force (N)", 1)
        self.draw_single_diagram(diagram_rect, self.M, COLORS['moment'], "Bending Moment (Nm)", 2)
        
    def draw_single_diagram(self, area, data, color, label, index=0):
        height = DIAGRAM_HEIGHT // 3
        top = area.top + index * height
        max_val = max(abs(v) for v in data) if data else 1
        max_val = max(max_val, 1e-6)
        
        # Draw axes and curve
        pygame.draw.line(self.screen, COLORS['text'], 
                        (area.left + 50, top + height//2),
                        (area.right - 50, top + height//2), 2)
        points = []
        if data:
            points = [(area.left + 50 + (i/len(data))*(area.width-100), 
                      top + height//2 - (val/max_val)*(height//2 - 20)) 
                     for i, val in enumerate(data)]
            pygame.draw.lines(self.screen, color, False, points, 2)

        # Find and label critical points
        font = pygame.font.SysFont('Arial', 12)
        critical_points = []
        
        # Always include first and last points
        critical_points.append(0)
        critical_points.append(len(data)-1)
        
        # Find local maxima/minima
        for i in range(1, len(data)-1):
            if (data[i] > data[i-1] and data[i] > data[i+1]) or \
               (data[i] < data[i-1] and data[i] < data[i+1]):
                critical_points.append(i)
        
        # Draw labels
        for i in critical_points:
            x, y = points[i]
            value = data[i]
            text = font.render(f"{value:.1f}", True, color)
            self.screen.blit(text, (x - 15, y - 15))

        # Draw diagram label
        self.screen.blit(pygame.font.SysFont('Arial', 16).render(label, True, color), 
                        (area.left + 60, top + 10))

    def calculate_reactions(self):
        sum_Fx = 0
        sum_Fz = 0
        sum_M_about_A = 0

        supports = [s for s in self.elements if isinstance(s, Support)]
        reactions = {}

        if len(supports) >= 2:
            A = supports[0]
            B = supports[1]

            for element in self.elements:
                if isinstance(element, Force):
                    fx, fz = element.get_components()
                    sum_Fx += fx
                    sum_Fz += fz
                    sum_M_about_A += fz * (element.x - A.x)
                elif isinstance(element, Moment):
                    sum_M_about_A += element.value * element.direction

            # Calculate reactions
            try:
                R_By = -sum_M_about_A / (B.x - A.x)
            except ZeroDivisionError:
                R_By = 0
            R_Ay = -sum_Fz - R_By
            R_Ax = -sum_Fx

            reactions[A] = (R_Ax, R_Ay)
            reactions[B] = (0, R_By)

        return reactions

    def calculate_internal_forces(self):
        reactions = self.calculate_reactions()
        num_points = 100
        self.N = [0] * num_points
        self.V = [0] * num_points
        self.M = [0] * num_points
        
        for i in range(num_points):
            x = self.beam_length * i / num_points
            N = 0  # Normal force
            V = 0  # Shear force
            M = 0  # Bending moment
            
            # Add reactions
            for support, (Rx, Rz) in reactions.items():
                if support.x < x:
                    V += Rz
                    M += Rz * (x - support.x)
                N += Rx  # Horizontal reactions contribute to normal force
            
            # Add applied forces and moments
            for element in self.elements:
                if isinstance(element, Force) and element.x < x:
                    fx, fz = element.get_components()
                    N += fx
                    V += fz
                    M += fz * (x - element.x)
                elif isinstance(element, Moment) and element.x < x:
                    M += element.value * element.direction
            
            self.N[i] = N
            self.V[i] = V
            self.M[i] = M

    def save_project(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write(f"BeamLength,{self.beam_length}\n")
                for element in self.elements:
                    if isinstance(element, Force):
                        magnitude = element.value * element.direction
                        f.write(f"Force,{element.x},{magnitude},{element.angle}\n")
                    elif isinstance(element, Moment):
                        magnitude = element.value * element.direction
                        f.write(f"Moment,{element.x},{magnitude}\n")
                    elif isinstance(element, Support):
                        f.write(f"Support,{element.x},{element.type}\n")
            print(f"Project saved to {filename}")
        except Exception as e:
            print(f"Save error: {str(e)}")

    def load_project(self, filename):
        try:
            self.elements = []
            with open(filename, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(',')
                    if parts[0] == "BeamLength":
                        self.beam_length = float(parts[1])
                    elif parts[0] == "Force":
                        x = float(parts[1])
                        magnitude = float(parts[2])
                        angle = float(parts[3])
                        self.elements.append(Force(x, magnitude, angle))
                    elif parts[0] == "Moment":
                        x = float(parts[1])
                        magnitude = float(parts[2])
                        self.elements.append(Moment(x, magnitude))
                    elif parts[0] == "Support":
                        x = float(parts[1])
                        support_type = parts[2]
                        self.elements.append(Support(x, support_type))
            self.calculate_internal_forces()
            print(f"Project loaded from {filename}")
        except Exception as e:
            print(f"Load error: {str(e)}")


    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                self.handle_mouse_click(mouse_pos, event.button)
            if event.type == KEYDOWN:
                self.handle_keyboard(event)
            if event.type == pygame.TEXTINPUT:
                self.handle_text_input(event.text)

    def handle_mouse_click(self, mouse_pos, button):
        if button == 3:  # Right click
            self.delete_selected_element()
            return
        if mouse_pos[0] < UI_PANEL_WIDTH:
            for i, (mode, _, pos) in enumerate(self.tools):
                if pos[1] < mouse_pos[1] < pos[1] + 40:
                    self.mode = mode
                    return
                    
        if mouse_pos[0] > SCREEN_SIZE[0] - PROP_PANEL_WIDTH:
            if self.selected_element and isinstance(self.selected_element, (Force, Moment)):
                y = 170 + 30 * (3 + len(self.elements))
                if SCREEN_SIZE[0]-140 < mouse_pos[0] < SCREEN_SIZE[0]-20 and y < mouse_pos[1] < y+30:
                    self.selected_element.direction *= -1
                    self.calculate_internal_forces()
            return
        
        beam_x = self.screen_to_beam_x(mouse_pos[0])
        if button == 1:
            if self.mode == 'force':
                new_force = Force(beam_x, 100)
                self.elements.append(new_force)
                self.selected_element = new_force
            elif self.mode == 'moment':
                new_moment = Moment(beam_x, 50)
                self.elements.append(new_moment)
                self.selected_element = new_moment
            elif self.mode == 'support':
                new_support = Support(beam_x, self.support_type)
                self.elements.append(new_support)
                self.selected_element = new_support
            else:
                closest = None
                min_dist = float('inf')
                for element in self.elements:
                    dist = abs(element.x - beam_x)
                    if dist < min_dist and dist < 0.2:
                        min_dist = dist
                        closest = element
                self.selected_element = closest
            self.calculate_internal_forces()
        elif button == 3 and self.selected_element:
            self.elements.remove(self.selected_element)
            self.selected_element = None
            self.calculate_internal_forces()
            
    def handle_keyboard(self, event):
        if event.key == K_ESCAPE:
            self.selected_element = None
        elif event.key == K_s and (pygame.key.get_mods() & KMOD_CTRL):
            self.save_project("project.ntmd")
        elif event.key == K_o and (pygame.key.get_mods() & KMOD_CTRL):
            self.load_project("project.ntmd")
        elif event.key == K_s:
            self.mode = 'select'
        elif event.key == K_f:
            self.mode = 'force'
        elif event.key == K_m:
            self.mode = 'moment'
        elif event.key == K_p:
            self.mode = 'support'
        elif self.selected_element:
            if event.key == K_LEFT:
                self.selected_element.x = max(0, self.selected_element.x - 0.1)
                self.calculate_internal_forces()
            elif event.key == K_RIGHT:
                self.selected_element.x = min(self.beam_length, self.selected_element.x + 0.1)
                self.calculate_internal_forces()
            elif event.key == K_UP:
                if isinstance(self.selected_element, (Force, Moment)):
                    self.selected_element.value *= 1.1
                    self.calculate_internal_forces()
            elif event.key == K_DOWN:
                if isinstance(self.selected_element, (Force, Moment)):
                    self.selected_element.value *= 0.9
                    self.calculate_internal_forces()
            elif event.key == K_d:
                if isinstance(self.selected_element, (Force, Moment)):
                    self.selected_element.direction *= -1
                    self.calculate_internal_forces()
            elif event.key == K_RETURN:
                self.editing_value = False
                if self.value_str:
                    try:
                        self.selected_element.value = abs(float(self.value_str))
                        if isinstance(self.selected_element, Force):
                            self.selected_element.direction = -1 if float(self.value_str) < 0 else 1
                        elif isinstance(self.selected_element, Moment):
                            self.selected_element.direction = 1 if float(self.value_str) >= 0 else -1
                        self.calculate_internal_forces()
                    except ValueError:
                        pass
                self.value_str = ""
            elif event.key == K_BACKSPACE:
                if self.editing_value:
                    self.value_str = self.value_str[:-1]
                else:
                    if isinstance(self.selected_element, (Force, Moment)):
                        self.selected_element.value = abs(self.selected_element.value) // 10
                        self.calculate_internal_forces()

    def handle_text_input(self, text):
        if self.selected_element and (text.isdigit() or text in ['.', '-']):
            if text == '-' and self.value_str == "":
                self.value_str = "-"
            else:
                self.editing_value = True
                self.value_str += text
                try:
                    value = float(self.value_str)
                    self.selected_element.value = abs(value)
                    if isinstance(self.selected_element, Force):
                        self.selected_element.direction = -1 if value < 0 else 1
                    elif isinstance(self.selected_element, Moment):
                        self.selected_element.direction = 1 if value >= 0 else -1
                    self.calculate_internal_forces()
                except ValueError:
                    pass

    def delete_selected_element(self):
        if self.selected_element:
            self.elements.remove(self.selected_element)
            self.selected_element = None
            self.calculate_internal_forces()
            

    def run(self):
        while True:
            self.screen.fill(COLORS['background'])
            self.handle_events()
            
            self.draw_beam()
            self.draw_elements()
            self.draw_coordinate_system()
            
            self.draw_toolbar()
            self.draw_properties_panel()
            self.draw_diagrams()
            
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    editor = BeamEditor()
    editor.run()
