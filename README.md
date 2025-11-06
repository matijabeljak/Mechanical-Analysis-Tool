# Beam Editor Documentation
(OLD VERSION!!! DOCUMENTATION WRITTEN BY AI)
This documentation describes the functionality of the Pygame-based **Beam Editor** application, which allows users to simulate structural elements (forces, moments, supports) on a beam and visualize internal forces/moments. Below is a breakdown of all classes, methods, and key functions.

---

## Table of Contents
1. [Classes](#classes)
   - [StructuralElement](#structuralelement)
   - [Force](#force)
   - [Moment](#moment)
   - [Support](#support)
   - [BeamEditor](#beameditor)
2. [Key Methods](#key-methods)
   - [Reaction/Force Calculations](#reactionforce-calculations)
   - [UI Rendering](#ui-rendering)
   - [Event Handling](#event-handling)
   - [File Operations](#file-operations)
3. [Usage](#usage)

---

## Classes

### StructuralElement
Base class for all structural elements (forces, moments, supports).

#### Methods:
- **`__init__(self, x_pos, value=0)`**  
  Initializes an element with an `x` position and `value` (magnitude).
  - `x_pos` (float): Position along the beam (meters).
  - `value` (float): Magnitude of the element (default: 0).

- **`draw(self, surface, scale, offset_x)`**  
  Abstract method to draw the element on the Pygame surface.  
  Must be implemented by subclasses.

---

### Force
Subclass of `StructuralElement` representing a force applied to the beam.

#### Methods:
- **`__init__(self, x_pos, magnitude=100, angle=90)`**  
  Initializes a force with direction and angle.  
  - `magnitude` (float): Force magnitude (positive = upward, negative = downward).  
  - `angle` (float): Angle from horizontal (degrees).  

- **`get_components(self)`**  
  Returns horizontal (`fx`) and vertical (`fz`) force components.  
  - **Returns**: Tuple `(fx, fz)`.

- **`draw(...)`**  
  Draws the force as an arrow on the beam.

---

### Moment
Subclass of `StructuralElement` representing a rotational moment.

#### Methods:
- **`__init__(self, x_pos, magnitude=100)`**  
  Initializes a moment with direction.  
  - `magnitude` (float): Moment magnitude (positive = clockwise, negative = counter-clockwise).  

- **`draw(...)`**  
  Draws the moment as a circular arrow.

---

### Support
Subclass of `StructuralElement` representing pinned or roller supports.

#### Methods:
- **`__init__(self, x_pos, support_type='pinned')`**  
  Initializes a support.  
  - `support_type` (str): `'pinned'` (triangle) or `'roller'` (circle).  

- **`draw(...)`**  
  Renders the support symbol.

---

### BeamEditor
Main class managing the application loop, UI, and calculations.

#### Attributes:
- `beam_length` (float): Length of the beam (meters).  
- `elements` (list): List of all structural elements.  
- `selected_element` (StructuralElement): Currently selected element.  
- `N`, `V`, `M` (list): Internal normal, shear, and bending moment diagrams.  

---

## Key Methods

### Reaction/Force Calculations

#### `calculate_reactions()`
Calculates support reactions using equilibrium equations.  
- **Returns**: Dictionary mapping supports to `(Rx, Rz)` reaction components.  
- **Logic**:  
  - Sums forces/moments from all elements.  
  - Solves for vertical reactions using sum of moments about the first support.  

#### `calculate_internal_forces()`
Computes internal forces (`N`, `V`, `M`) along the beam.  
- **Steps**:  
  1. Calculate reactions.  
  2. For each point on the beam, sum forces/moments to the left of the point.  

---

### UI Rendering

#### `draw_beam()`
Draws the beam as a horizontal line on the screen.

#### `draw_diagrams()`
Renders the internal force/moment diagrams below the beam.

#### `draw_properties_panel()`
Displays element properties and editing tools on the right panel.

#### `draw_toolbar()`
Draws the left-side toolbar with mode selection buttons.

---

### Event Handling

#### `handle_events()`
Processes Pygame events (mouse, keyboard). Delegates to:  
- `handle_mouse_click()`: Handles element placement/deletion.  
- `handle_keyboard()`: Processes keyboard shortcuts (e.g., `Ctrl+S` to save).  
- `handle_text_input()`: Manages numeric input for element values.

---

### File Operations

#### `save_project(filename)`
Saves the current project to a `.ntmd` file.  
- **File Format**: CSV with lines like `Force,x,magnitude,angle`.

#### `load_project(filename)`
Loads a project from a `.ntmd` file.  
- **Note**: Reinitializes the beam and elements.

---

## Usage

### Running the Application
```python
if __name__ == "__main__":
    editor = BeamEditor()
    editor.run()
```

### Controls
| **Action**               | **Key/Mouse**                  |
|--------------------------|--------------------------------|
| Select mode              | Toolbar buttons (`S`, `F`, etc) |
| Place element            | Left-click on beam            |
| Delete element           | Right-click on element        |
| Adjust element position  | Left/Right arrow keys          |
| Adjust magnitude         | Up/Down arrow keys             |
| Toggle direction         | `D` key                        |
| Save project             | `Ctrl + S`                     |
| Load project             | `Ctrl + O`                     |

### Notes
- Supports must be placed at the beam ends for valid reaction calculations.  
- Diagrams update automatically when elements are modified.  
- Negative force values flip direction (e.g., `-100N` = downward force).

---

This documentation covers all core functionalities. For advanced usage, refer to the code comments or modify the calculation logic in `calculate_reactions()` and `calculate_internal_forces()`.
