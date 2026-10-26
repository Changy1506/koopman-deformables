from enum import Enum

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from PIL import Image



class EditMode(Enum):
    DRAW = 1
    MOVE = 2


class SimMode(Enum):
    SIMULATE_DYNAMIC = 1
    SIMULATE_STATIC = 2
    PAUSE = 3






class InteractiveApp:
    def __init__(self):

        # UI state
        self.edit_mode = EditMode.DRAW
        self.sim_mode = SimMode.PAUSE
        self.polyline_points = []
        self.selected_point_idx = None
        self.canvas_size = (400, 300)
        self.point_radius = 4.0
        self.selection_radius = 8.0
        self.circle_radius = 70

        # UI colors
        self.selected_color = self._color_u32(76, 175, 80)
        self.unselected_color = self._color_u32(52, 152, 219)
        self.dark_gray = self._color_u32(0x32, 0x32, 0x32, 0xFF)
        self.med_gray = self._color_u32(0x50, 0x50, 0x50, 0xFF)
        self.green = self._color_u32(0x00, 0xFF, 0x00, 0xFF)
        self.white = self._color_u32(0xFF, 0xFF, 0xFF, 0xFF)

        self.ps_cloud = None
        self.direction_point = None
        self.gravity_point = None

        self.canvas_window_pos = (0, 0)
        
        self._polyline_curve = None
        self._polyline_n_last = 0
        self.polyline_y_3d = 2.0
        self.polyline_3d_name = "ui_polyline"
        self.polyline_3d_scale = 0.01


    def _color_u32(self, r, g, b, a=255):
        return (a << 24) | (b << 16) | (g << 8) | r

    def initialize_polyscope(self):
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.look_at((2, -0.5, 2), (0.5, -0.5, 0))

        




    def draw_ui_callback(self):
        self._draw_simulation_controls()
        self._draw_canvas()
        self._handle_simulation_step()



    def _draw_simulation_controls(self):
        psim.TextUnformatted("Simulation mode:")

        self._draw_toggle_button(
            "Simulate",
            self.sim_mode,
            SimMode.SIMULATE_DYNAMIC,
            self._start_dynamic_simulation,
        )
        psim.SameLine()
        

        

        transform_changed = False

        
        if transform_changed and len(self.polyline_points) > 1:
            self.simulator.update_geometry_from_polyline(
                self.polyline_points, self.canvas_size
            )

    def _draw_toggle_button(self, label, current_value, target_value, callback):
        if current_value == target_value:
            psim.PushStyleColor(psim.ImGuiCol_Button, self.selected_color)
        else:
            psim.PushStyleColor(psim.ImGuiCol_Button, self.unselected_color)

        if psim.Button(label):
            callback()

        psim.PopStyleColor()

    def _start_dynamic_simulation(self):
        self.sim_mode = SimMode.SIMULATE_DYNAMIC
        if len(self.polyline_points) > 1:
            pass


    def _draw_canvas(self):
        psim.Spacing()
        psim.Separator()
        psim.TextUnformatted("Goal Trajectory")

        # Get canvas position
        window_pos = psim.GetCursorScreenPos()
        canvas_end = (
            window_pos[0] + self.canvas_size[0],
            window_pos[1] + self.canvas_size[1],
        )

        # Store window position for coordinate transformation (like original)
        self.canvas_window_pos = window_pos

        # Draw canvas background
        draw_list = psim.GetWindowDrawList()
        draw_list.AddRectFilled(window_pos, canvas_end, self.dark_gray)

        # Reserve space
        psim.Dummy(self.canvas_size)

        # Handle mouse interactions
        self._handle_canvas_mouse(window_pos)

        # Draw polyline
        self._draw_polyline()

    def _handle_canvas_mouse(self, window_pos):
        mouse_pos = psim.GetMousePos()
        mouse_clicked = psim.IsMouseClicked(psim.ImGuiMouseButton_Left)
        mouse_down = psim.IsMouseDown(psim.ImGuiMouseButton_Left)
        mouse_released = psim.IsMouseReleased(psim.ImGuiMouseButton_Left)

        inside_canvas = (
            window_pos[0] <= mouse_pos[0] <= window_pos[0] + self.canvas_size[0]
            and window_pos[1] <= mouse_pos[1] <= window_pos[1] + self.canvas_size[1]
        )

        if not inside_canvas:
            return

        if mouse_clicked:
            self._handle_canvas_click(mouse_pos)
        elif mouse_down and self.selected_point_idx is not None:
            relative_pos = (mouse_pos[0] - window_pos[0], mouse_pos[1] - window_pos[1])
            self.polyline_points[self.selected_point_idx] = relative_pos
            if len(self.polyline_points) > 1:
                self.simulator.update_geometry_from_polyline(
                    self.polyline_points, self.canvas_size
                )
                self._update_visualization()
        elif mouse_released:
            if self.selected_point_idx is not None:
                relative_pos = (
                    mouse_pos[0] - window_pos[0],
                    mouse_pos[1] - window_pos[1],
                )
                self.polyline_points[self.selected_point_idx] = relative_pos
                if len(self.polyline_points) > 1:
                    self.simulator.update_geometry_from_polyline(
                        self.polyline_points, self.canvas_size
                    )
                    self._update_visualization()
            self.selected_point_idx = None

    def _handle_canvas_click(self, mouse_pos):
        """Handle mouse click on canvas."""
        relative_pos = (
            mouse_pos[0] - self.canvas_window_pos[0],
            mouse_pos[1] - self.canvas_window_pos[1],
        )

        if self.edit_mode == EditMode.DRAW:
            self.polyline_points.append(relative_pos)
            self.selected_point_idx = None
        elif self.edit_mode == EditMode.MOVE:
            if len(self.polyline_points) > 0:
                points_array = np.array(self.polyline_points)
                distances = np.sum((points_array - np.array(relative_pos)) ** 2, axis=1)
                closest_idx = np.argmin(distances)

                if distances[closest_idx] < self.selection_radius**2:
                    self.selected_point_idx = closest_idx

    def _ensure_3d_polyline(self, nodes: np.ndarray, edges: np.ndarray):
        """Create or recreate the curve network if node count changed."""
        n = nodes.shape[0]

        if n < 2:
            if self._polyline_curve is not None:
                self._polyline_curve.set_enabled(False)
            self._polyline_n_last = n
            return

        need_recreate = (self._polyline_curve is None) or (n != self._polyline_n_last)

        if need_recreate:
            # Safest: remove old structure then register new one
            try:
                ps.remove_structure(self.polyline_3d_name)
            except Exception:
                pass

            self._polyline_curve = ps.register_curve_network(self.polyline_3d_name, nodes, edges)
            self._polyline_curve.set_enabled(True)

            # Optional appearance
            #self._polyline_curve.set_radius(0.002)
            

            self._polyline_n_last = n
        else:
            # Same node count -> fast path
            self._polyline_curve.set_enabled(True)
            self._polyline_curve.update_node_positions(nodes)

    def _draw_polyline(self):
        # ---------- 2D overlay drawing ----------
        dl = psim.GetWindowDrawList()

        if len(self.polyline_points) > 1:
            abs_points = [
                (pt[0] + self.canvas_window_pos[0], pt[1] + self.canvas_window_pos[1])
                for pt in self.polyline_points
            ]
            dl.AddPolyline(abs_points, len(abs_points), self.green, 0, 2.0)

        for pt in self.polyline_points:
            abs_pt = (pt[0] + self.canvas_window_pos[0], pt[1] + self.canvas_window_pos[1])
            dl.AddCircleFilled(abs_pt, self.point_radius, self.white)

        # ---------- 3D curve network (x <- 2D x, z <- 2D y, y = const) ----------
        n = len(self.polyline_points)
        if n < 2:
            self._ensure_3d_polyline(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32))
            return

        s = float(getattr(self, "polyline_3d_scale", 1.0))
        nodes = np.array([(s*p[0] - 2, self.polyline_y_3d, s*p[1] - 1.5) for p in self.polyline_points], dtype=np.float32)
        edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)

        self._ensure_3d_polyline(nodes, edges)
        




    def _handle_simulation_step(self):
        if self.sim_mode == SimMode.SIMULATE_DYNAMIC:
            if len(self.polyline_points) > 1:
                pass
                #self._update_visualization()
        

    def _update_visualization(self):
        pass

    def run(self):
        self.initialize_polyscope()
        ps.set_user_callback(self.draw_ui_callback)
        ps.show()


def main():
    app = InteractiveApp()
    app.run()


if __name__ == "__main__":
    main()
