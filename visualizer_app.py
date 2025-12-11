import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# Configure matplotlib (light theme)
matplotlib.use("TkAgg")
plt.style.use("default")

class ADDAVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADDA Ultra Visualizer (High-Fidelity)")
        self.root.geometry("1500x950")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Data State ---
        self.current_run_dir = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.int_data = None 
        self.scat_data = None
        
        # --- Visualization Settings ---
        self.pol_var = tk.StringVar(value="X")
        self.current_z_idx = tk.IntVar(value=0)
        
        # Appearance
        self.cmap_var = tk.StringVar(value="jet")
        self.interp_method = tk.StringVar(value="cubic") # none, linear, cubic
        self.show_vectors = tk.BooleanVar(value=False)
        self.vector_type = tk.StringVar(value="quiver") # quiver, stream
        self.show_contours = tk.BooleanVar(value=False)
        self.alpha_val = tk.DoubleVar(value=0.3)
        self.pt_size = tk.DoubleVar(value=2.0)
        
        self.scat_element = tk.StringVar(value="S11")
        self.log_mueller = tk.BooleanVar(value=False)
        self.convert_coords = tk.BooleanVar(value=True)  # Whether to convert beam->lab for non-surf
        self._scat_cbar = None
        self._cached_prop = {}
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Top: Source
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)
        ttk.Label(top, text="Run Directory:").pack(side="left")
        ttk.Entry(top, textvariable=self.current_run_dir, width=50).pack(side="left", padx=5)
        ttk.Button(top, text="Browse...", command=self.browse_dir).pack(side="left")
        ttk.Button(top, text="LOAD DATA", command=self.load_all_data).pack(side="left", padx=10)
        ttk.Label(top, textvariable=self.status_var, foreground="#00ffff").pack(side="left", padx=10)

        # Main Layout
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Internal/Near Field
        self.tab_int = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_int, text="Near/Internal Field Analysis")
        self._setup_int_tab()
        
        # Tab 2: Far Field
        self.tab_scat = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scat, text="Far Field Scattering")
        self._setup_scat_tab()

    def _setup_int_tab(self):
        paned = ttk.PanedWindow(self.tab_int, orient="horizontal")
        paned.pack(fill="both", expand=True)
        
        # --- Sidebar Controls ---
        ctrl = ttk.Frame(paned, width=320)
        paned.add(ctrl, weight=0)
        
        # Group 1: Data Slice
        g1 = ttk.LabelFrame(ctrl, text="Slicing & Polarization", padding=10)
        g1.pack(fill="x", padx=5, pady=5)
        
        ttk.Radiobutton(g1, text="X-Polarization", variable=self.pol_var, value="X", command=self.reload_int_data).pack(anchor="w")
        ttk.Radiobutton(g1, text="Y-Polarization", variable=self.pol_var, value="Y", command=self.reload_int_data).pack(anchor="w")
        
        ttk.Label(g1, text="Z-Layer Slice:").pack(anchor="w", pady=(10,0))
        self.scale_z = ttk.Scale(g1, from_=0, to=10, variable=self.current_z_idx, command=lambda v: self.update_int_plots())
        self.scale_z.pack(fill="x")
        self.lbl_z = ttk.Label(g1, text="Z = N/A", font=("Consolas", 10))
        self.lbl_z.pack(anchor="e")

        # Group 2: Rendering Style
        g2 = ttk.LabelFrame(ctrl, text="Rendering Options", padding=10)
        g2.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(g2, text="Colormap:").pack(anchor="w")
        cmaps = ["jet", "inferno", "viridis", "plasma", "magma", "coolwarm", "hsv"]
        ttk.Combobox(g2, textvariable=self.cmap_var, values=cmaps, state="readonly").pack(fill="x")
        self.cmap_var.trace("w", lambda *a: self.update_int_plots())
        
        ttk.Label(g2, text="Interpolation (2D):").pack(anchor="w", pady=(10,0))
        ttk.Combobox(g2, textvariable=self.interp_method, values=["none", "linear", "cubic", "nearest"], state="readonly").pack(fill="x")
        
        ttk.Checkbutton(g2, text="Show Contour Lines", variable=self.show_contours, command=self.update_int_plots).pack(anchor="w", pady=5)
        
        # Group 3: Vectors
        g3 = ttk.LabelFrame(ctrl, text="Vector Field Overlay", padding=10)
        g3.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(g3, text="Enable Vectors", variable=self.show_vectors, command=self.update_int_plots).pack(anchor="w")
        ttk.Radiobutton(g3, text="Quiver (Arrows)", variable=self.vector_type, value="quiver", command=self.update_int_plots).pack(anchor="w")
        ttk.Radiobutton(g3, text="Streamlines (Flow)", variable=self.vector_type, value="stream", command=self.update_int_plots).pack(anchor="w")

        # Group 4: 3D Settings
        g4 = ttk.LabelFrame(ctrl, text="3D Settings", padding=10)
        g4.pack(fill="x", padx=5, pady=5)
        ttk.Label(g4, text="Transparency (Alpha):").pack(anchor="w")
        ttk.Scale(g4, from_=0.01, to=1.0, variable=self.alpha_val, command=lambda v: self.update_int_3d()).pack(fill="x")
        ttk.Label(g4, text="Point Size:").pack(anchor="w")
        ttk.Scale(g4, from_=0.1, to=10.0, variable=self.pt_size, command=lambda v: self.update_int_3d()).pack(fill="x")
        
        ttk.Button(g4, text="Load Geometry Overlay...", command=self.manual_load_geometry).pack(fill="x", pady=(10, 2))
        ttk.Button(g4, text="Redraw 3D", command=self.update_int_3d).pack(fill="x", pady=2)
        
        # Group 5: Far Field Calculation (NTFF)
        g5 = ttk.LabelFrame(ctrl, text="Far Field Analysis (NTFF)", padding=10)
        g5.pack(fill="x", padx=5, pady=5)
        ttk.Label(g5, text="Calculate far field from internal field\nusing Near-to-Far Field Transformation", 
                  font=("TkDefaultFont", 9), foreground="blue").pack(anchor="w", pady=(0, 5))
        ttk.Button(g5, text="Calculate Far Field (NTFF)", command=self.calc_ntff_compare,
                   width=30).pack(fill="x", pady=2)
        ttk.Label(g5, text="Note: Uses current Z-slice from above", 
                  font=("TkDefaultFont", 8), foreground="gray").pack(anchor="w", pady=(2, 0))

        # --- Plots ---
        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=1)
        
        # Split vertical: 2D Top, 3D Bottom
        vpaned = ttk.PanedWindow(plot_frame, orient="vertical")
        vpaned.pack(fill="both", expand=True)
        
        # 2D Canvas
        self.fig2d = plt.Figure(figsize=(5,4), dpi=100)
        self.ax2d = self.fig2d.add_subplot(111)
        # Adjust layout to reduce margins
        self.fig2d.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        self.cv2d = FigureCanvasTkAgg(self.fig2d, vpaned)
        self.tb2d = NavigationToolbar2Tk(self.cv2d, vpaned)
        vpaned.add(self.cv2d.get_tk_widget(), weight=1)
        
        # 3D Canvas
        self.fig3d = plt.Figure(figsize=(5,4), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.fig3d.subplots_adjust(left=0, right=1, top=1, bottom=0) # Maximize 3D area
        self.cv3d = FigureCanvasTkAgg(self.fig3d, vpaned)
        vpaned.add(self.cv3d.get_tk_widget(), weight=2) # Give more weight to 3D

    def _setup_scat_tab(self):
        # Simple Far Field
        f = ttk.Frame(self.tab_scat)
        f.pack(fill="both", expand=True)
        
        ctrl = ttk.Frame(f)
        ctrl.pack(side="top", fill="x", padx=10, pady=5)
        
        ttk.Label(ctrl, text="Mueller Element:").pack(side="left")
        els = [f"S{i}{j}" for i in range(1,5) for j in range(1,5)]
        ttk.Combobox(ctrl, textvariable=self.scat_element, values=els, state="readonly").pack(side="left", padx=5)
        ttk.Button(ctrl, text="Update View", command=self.update_scat_plots).pack(side="left")
        
        # Coordinate conversion option removed - always use original coordinates
        
        # --- NTFF Calculation from Internal Field ---
        ttk.Label(ctrl, text="|").pack(side="left", padx=(20, 5))
        ntff_frame = ttk.LabelFrame(ctrl, text="NTFF: Calculate Far Field from Internal Field", padding=6)
        ntff_frame.pack(side="left", padx=(10, 0))
        ttk.Button(ntff_frame, text="Calculate NTFF", command=self.calc_ntff_compare, 
                   style="Accent.TButton").pack(side="left", padx=2)
        ttk.Label(ntff_frame, text="(Uses current Z-slice from Internal Field tab)", 
                  font=("TkDefaultFont", 8), foreground="gray").pack(side="left", padx=5)
        
        # --- NTFF Comparison (2D Overlay) ---
        ttk.Label(ctrl, text="|").pack(side="left", padx=(20, 5))
        ttk.Label(ctrl, text="Analysis:").pack(side="left", padx=(5, 5))
        dual = ttk.LabelFrame(ctrl, text="Compare two runs (mueller 3D)", padding=6)
        dual.pack(side="left", padx=(20, 0))
        self.dual_run1 = tk.StringVar(value="run_rayleigh_free_45")
        self.dual_run2 = tk.StringVar(value="run_rayleigh_surf_45")
        ttk.Entry(dual, textvariable=self.dual_run1, width=18).pack(side="left", padx=2)
        ttk.Entry(dual, textvariable=self.dual_run2, width=18).pack(side="left", padx=2)
        ttk.Checkbutton(dual, text="Log S11", variable=self.log_mueller).pack(side="left", padx=4)
        ttk.Button(dual, text="Compare", command=self.compare_two_scat).pack(side="left", padx=2)

        # Plot (shared for single/compare)
        self.fig_scat = plt.Figure(figsize=(5,4), dpi=100)
        self.ax_scat = self.fig_scat.add_subplot(111, projection='3d')
        self.cv_scat = FigureCanvasTkAgg(self.fig_scat, f)
        self.cv_scat.get_tk_widget().pack(fill="both", expand=True)
        # default compare on startup (ignore errors)
        self.root.after(100, lambda: self._safe_compare_default())

    # --- Logic ---
    def browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.current_run_dir.set(d)
            self.load_all_data()

    def load_all_data(self):
        self.status_var.set("Loading Data...")
        self.root.update()
        self.reload_int_data()
        self.reload_scat_data()
        self.status_var.set("Ready")

    def reload_int_data(self):
        d = self.current_run_dir.get()
        pol = self.pol_var.get()
        p = Path(d) / f"IntField-{pol}"
        if not p.exists(): return
        
        try:
            # Load raw data
            # x y z |E|^2 Exr Exi Eyr Eyi Ezr Ezi
            raw = np.loadtxt(p, skiprows=1)
            if raw.ndim == 1: raw = raw.reshape(1, -1)
            
            self.int_data = {
                'x': raw[:,0], 'y': raw[:,1], 'z': raw[:,2], 'I': raw[:,3],
                'Ex': raw[:,4] + 1j*raw[:,5],
                'Ey': raw[:,6] + 1j*raw[:,7],
                'Ez': raw[:,8] + 1j*raw[:,9] # Store Ez too for NTFF
            }
            
            # Layers
            self.layers = sorted(np.unique(np.round(self.int_data['z'], 6)))
            self.scale_z.config(to=len(self.layers)-1)
            self.current_z_idx.set(len(self.layers)//2)
            
            # Attempt to load geometry if available (heuristics)
            # Check for log file to find geom file name, or check local .geom files
            self.geom_data = None
            log_p = Path(d) / "log"
            if log_p.exists():
                with open(log_p, 'r') as f:
                    content = f.read()
                    import re
                    m = re.search(r"-shape read (\S+)", content)
                    if m:
                        geom_file = m.group(1)
                        # Try absolute or relative
                        gp = Path(geom_file)
                        if not gp.exists():
                            # Try in repo root
                            gp = Path(d).parent / gp.name
                        
                        if gp.exists():
                            self.load_geometry(gp)

            self.update_int_plots()
            self.update_int_3d()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def manual_load_geometry(self):
        f = filedialog.askopenfilename(filetypes=[("Geometry", "*.geom"), ("All Files", "*.*")])
        if f:
            self.load_geometry(Path(f))
            self.update_int_3d()
            messagebox.showinfo("Info", "Geometry Overlay Loaded")

    def load_geometry(self, path):
        try:
            # Load x y z Re(m) Im(m)
            # Usually integers?
            g = np.loadtxt(path)
            if g.ndim == 1: g = g.reshape(1, -1)
            # Filter where m != 1 (vacuum) or m != 1.00001 (air)
            # We assume object has higher index
            m_real = g[:, 3]
            mask_obj = m_real > 1.1 # Heuristic for "Object"
            
            if np.any(mask_obj):
                self.geom_data = {
                    'x': g[mask_obj, 0],
                    'y': g[mask_obj, 1],
                    'z': g[mask_obj, 2]
                }
                # Scale geometry coordinates?
                # ADDA output coordinates in IntField are usually centered and scaled by d?
                # Actually, IntField coordinates are usually physical if -grid not used?
                # Or dipole indices centered?
                # Let's check ranges.
                
                # Align geom to int_data range
                ix, iy, iz = self.int_data['x'], self.int_data['y'], self.int_data['z']
                gx, gy, gz = self.geom_data['x'], self.geom_data['y'], self.geom_data['z']
                
                # ADDA centers the particle at 0,0,0 usually.
                # Geom file usually 0..N.
                # So we need to shift geom to center.
                
                gcx, gcy, gcz = (gx.min()+gx.max())/2, (gy.max()+gy.min())/2, (gz.max()+gz.min())/2
                icx, icy,icz = (ix.min()+ix.max())/2, (iy.max()+iy.min())/2, (iz.max()+iz.min())/2
                
                # Shift
                self.geom_data['x'] -= gcx
                self.geom_data['y'] -= gcy
                self.geom_data['z'] -= gcz
                
                # Scale? If IntField is in physical units and Geom in indices
                # We need resolution d.
                # d approx = (ix.max() - ix.min()) / (gx.max() - gx.min())
                if (gx.max() - gx.min()) > 0:
                    scale = (ix.max() - ix.min()) / (gx.max() - gx.min())
                    # If scale is close to 1 (both indices) or physical?
                    # Let's assume proportional scaling is needed if ranges differ significantly
                    # But usually IntField uses physical units (k*d) or just d units.
                    # Let's heuristic: if ranges match (within 10%), don't scale.
                    # If ranges differ by factor of ~0.05 or something (d), scale.
                    
                    # Actually, let's just plot normalized or trust the shift.
                    # For visualization, we just want to see the shape.
                    # We will apply this shift.
                    pass
                    
        except: pass

    def _get_polarization_vectors(self, log_file):
        """Extract polarization vectors from ADDA log file (from visualize_all.py)"""
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                import re
                pol_x_match = re.search(
                    r"Incident polarization X\(per\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)", content
                )
                pol_y_match = re.search(
                    r"Incident polarization Y\(par\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)", content
                )
                if pol_x_match and pol_y_match:
                    inc_pol_x = np.array([
                        float(pol_x_match.group(1)), 
                        float(pol_x_match.group(2)), 
                        float(pol_x_match.group(3))
                    ])
                    inc_pol_y = np.array([
                        float(pol_y_match.group(1)), 
                        float(pol_y_match.group(2)), 
                        float(pol_y_match.group(3))
                    ])
                    return inc_pol_x, inc_pol_y
        except Exception:
            pass
        # Default fallback
        return np.array([-0.7071, 0.0, -0.7071]), np.array([0.0, 1.0, 0.0])
    
    def _has_surf_mode(self, log_file):
        """Check if ADDA simulation used -surf mode"""
        try:
            import re
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Check command line for -surf
                if re.search(r"-surf\s+", content):
                    return True
                # Also check for substrate info in log
                if re.search(r"substrate|Substrate", content, re.IGNORECASE):
                    return True
        except Exception:
            pass
        return False
    
    def _get_propagation_direction(self, log_file):
        """Extract propagation direction from ADDA log file"""
        try:
            import re
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Look for "Incident propagation vector: (x,y,z)" format
                # Example: "Incident propagation vector: (0.707107,0,-0.707107)"
                prop_match = re.search(
                    r"Incident propagation vector:\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)", 
                    content
                )
                if prop_match:
                    prop = np.array([
                        float(prop_match.group(1)),
                        float(prop_match.group(2)),
                        float(prop_match.group(3))
                    ])
                    # Normalize to unit vector
                    prop = prop / np.linalg.norm(prop)
                    return prop
        except Exception as e:
            print(f"Warning: Failed to parse propagation direction: {e}")
        # Default: negative z direction (typical for ADDA)
        return np.array([0.0, 0.0, -1.0])
    
    def _beam_to_lab_direction(self, theta_beam, phi_beam, prop, inc_pol_x, inc_pol_y):
        """Convert beam frame to lab frame (from visualize_all.py)"""
        theta_rad = np.radians(theta_beam)
        phi_rad = np.radians(phi_beam)
        cos_theta = np.cos(theta_rad)[:, np.newaxis]
        sin_theta = np.sin(theta_rad)[:, np.newaxis]
        cos_phi = np.cos(phi_rad)[:, np.newaxis]
        sin_phi = np.sin(phi_rad)[:, np.newaxis]
        robs_beam = cos_theta * prop + sin_theta * (cos_phi * inc_pol_x + sin_phi * inc_pol_y)
        norm = np.linalg.norm(robs_beam, axis=1, keepdims=True)
        return robs_beam / norm
    
    def _direction_to_lab_angles(self, robs):
        """Convert direction vector to lab frame angles (from visualize_all.py)"""
        theta_lab = np.degrees(np.arccos(np.clip(robs[:, 2], -1, 1)))
        phi_lab = np.degrees(np.arctan2(robs[:, 1], robs[:, 0]))
        phi_lab[phi_lab < 0] += 360
        return theta_lab, phi_lab

    def reload_scat_data(self):
        d = self.current_run_dir.get()
        p = Path(d) / "mueller_scatgrid"
        log_p = Path(d) / "log"
        if not p.exists():
            self.scat_data = None
            self.status_var.set("未找到 mueller_scatgrid")
            return
        try:
            raw = np.loadtxt(p, skiprows=1)
            if raw.ndim == 1: raw = raw.reshape(1,-1)
            
            # Use original coordinates directly (no conversion, already in lab frame)
            theta_lab = raw[:,0]
            phi_lab = raw[:,1]
            
            # Store data
            self.scat_data = {
                'theta': theta_lab, 
                'phi': phi_lab, 
                'm': raw[:,2:]
            }
            self.update_scat_plots()
        except Exception as exc:
            self.scat_data = None
            messagebox.showerror("远场读取失败", str(exc))

    def calc_ntff_compare(self):
        """Angle-spectrum far-field from Z slice(s) in air layer, with Mueller comparison."""
        if not self.int_data:
            messagebox.showwarning("Missing data", "Load IntField first.")
            return

        # Ask user how many slices to average (default: current slice only)
        try:
            n_slices_str = simpledialog.askstring(
                "Multi-slice averaging", 
                "Number of Z slices to average (air layer only, must be odd):\n" +
                f"Current Z index: {self.current_z_idx.get()}\n" +
                f"Total layers: {len(self.layers)}\n" +
                "Enter 1 for single slice, 3/5/7 for averaging:",
                initialvalue="1"
            )
            if n_slices_str is None:
                return
            n_slices = int(n_slices_str)
            if n_slices < 1 or n_slices % 2 == 0:
                messagebox.showerror("Invalid", "Number of slices must be odd and >= 1")
                return
        except (ValueError, TypeError):
            messagebox.showerror("Invalid", "Please enter a valid number")
            return

        # Select slices around current Z index
        idx_center = self.current_z_idx.get()
        idx_center = max(0, min(idx_center, len(self.layers)-1))
        half_span = (n_slices - 1) // 2
        idx_start = max(0, idx_center - half_span)
        idx_end = min(len(self.layers), idx_center + half_span + 1)
        selected_indices = list(range(idx_start, idx_end))
        
        if len(selected_indices) == 0:
            messagebox.showwarning("No slices", "No valid slices selected")
            return
        
        # Collect data from all selected slices
        all_x, all_y, all_Ex, all_Ey = [], [], [], []
        z_vals_used = []
        
        for idx in selected_indices:
            z_val = self.layers[idx]
            mask = self._get_slice_mask(z_val)
            if np.any(mask):
                all_x.append(self.int_data['x'][mask])
                all_y.append(self.int_data['y'][mask])
                all_Ex.append(self.int_data['Ex'][mask])
                all_Ey.append(self.int_data['Ey'][mask])
                z_vals_used.append(z_val)
        
        if len(all_x) == 0:
            messagebox.showwarning("Empty slices", "No valid data in selected slices")
            return
        
        # Combine all slices (for averaging)
        x_slice = np.concatenate(all_x)
        y_slice = np.concatenate(all_y)
        Ex_slice = np.concatenate(all_Ex)
        Ey_slice = np.concatenate(all_Ey)
        
        z_val_display = f"{z_vals_used[0]:.4f}" if len(z_vals_used) == 1 else f"{z_vals_used[0]:.4f}~{z_vals_used[-1]:.4f} ({len(z_vals_used)} slices)"

        # 波长：尝试从 log 解析，否则询问
        wl = None
        log_path = Path(self.current_run_dir.get()) / "log"
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    for line in f:
                        if line.lower().startswith("lambda:"):
                            wl = float(line.split(":")[1].strip().split()[0])
                            break
            except Exception:
                wl = None
        if wl is None:
            try:
                wl = float(simpledialog.askstring("Lambda", "Input wavelength (same units as coordinates), e.g. 0.5:", initialvalue="0.5"))
            except Exception:
                wl = None
        if not wl or wl <= 0:
            messagebox.showerror("Invalid lambda", "Wavelength must be positive.")
            return
        k0 = 2 * np.pi / wl

        # Grid for interpolation
        grid_res = 256
        xi = np.linspace(x_slice.min(), x_slice.max(), grid_res)
        yi = np.linspace(y_slice.min(), y_slice.max(), grid_res)
        XI, YI = np.meshgrid(xi, yi)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]

        # 插值到网格（实部/虚部分开，避免 griddata 处理复数问题）
        def interp_complex(vals):
            re = griddata((x_slice, y_slice), np.real(vals), (XI, YI), method="cubic", fill_value=0)
            im = griddata((x_slice, y_slice), np.imag(vals), (XI, YI), method="cubic", fill_value=0)
            re = np.nan_to_num(re, nan=0.0)
            im = np.nan_to_num(im, nan=0.0)
            return re + 1j * im

        Ex_grid = interp_complex(Ex_slice)
        Ey_grid = interp_complex(Ey_slice)

        # 加窗，减少截断伪影
        wx = np.hanning(grid_res)
        wy = np.hanning(grid_res)
        window = np.outer(wy, wx)
        Ex_grid *= window
        Ey_grid *= window

        # zero-pad to improve k sampling (moderate)
        pad = grid_res // 4
        Ex_grid = np.pad(Ex_grid, ((pad, pad), (pad, pad)), mode="constant")
        Ey_grid = np.pad(Ey_grid, ((pad, pad), (pad, pad)), mode="constant")

        # 2D FFT - only Ex and Ey components (Ez not needed for 2D slice)
        Ex_k = np.fft.fftshift(np.fft.fft2(Ex_grid))
        Ey_k = np.fft.fftshift(np.fft.fft2(Ey_grid))
        # Intensity from Ex and Ey only
        I_fft = np.abs(Ex_k)**2 + np.abs(Ey_k)**2
        # k axes
        kx_vals = np.fft.fftshift(np.fft.fftfreq(Ex_grid.shape[1], d=dx)) * 2 * np.pi
        ky_vals = np.fft.fftshift(np.fft.fftfreq(Ex_grid.shape[0], d=dy)) * 2 * np.pi
        KX, KY = np.meshgrid(kx_vals, ky_vals)

        # propagating mask - only consider propagating modes (inside light cone)
        mask_prop = (KX**2 + KY**2) <= k0**2
        
        # No weighting - use FFT intensity directly
        # The FFT already gives the angular spectrum representation
        I_ang = I_fft.copy()
        I_ang[~mask_prop] = 0  # Zero out evanescent modes

        # log normalization
        def norm_and_log(arr, mask=None):
            if mask is None:
                mask = np.ones_like(arr, dtype=bool)
            arr_masked = np.where(mask, arr, np.nan)
            maxv = np.nanmax(arr_masked)
            if not np.isfinite(maxv) or maxv <= 0:
                return np.full_like(arr, np.nan)
            arr_n = arr_masked / maxv
            out = np.log10(arr_n + 1e-12)
            out[~mask] = np.nan
            return out

        SX = KX / k0
        SY = KY / k0
        mask_prop_log = mask_prop & np.isfinite(SX) & np.isfinite(SY)
        if not np.any(mask_prop_log):
            messagebox.showwarning("No propagating modes", "Light cone is empty. Check wavelength or coordinate units.")
            return

        I_fft_log = norm_and_log(I_fft, mask_prop_log)
        I_ang_log = norm_and_log(I_ang, mask_prop_log)

        def dyn_limits(arr_log, db=6):
            vmax = np.nanmax(arr_log)
            vmin = np.nanmin(arr_log)
            return max(vmax - db, vmin), vmax

        vmin_fft, vmax_fft = dyn_limits(I_fft_log, db=6)
        vmin_ang, vmax_ang = dyn_limits(I_ang_log, db=6)

        # View range in k/k0
        lim = 1.5
        extent = [-lim, lim, -lim, lim]

        top = tk.Toplevel(self.root)
        top.title(f"Far-field spectrum (Z={z_val_display}, λ={wl}, {len(z_vals_used)} slice{'s' if len(z_vals_used) > 1 else ''})")
        top.geometry("1100x760")

        fig = plt.Figure(figsize=(12, 6.5), dpi=100)
        fig.patch.set_facecolor("white")
        ax1 = fig.add_subplot(131, facecolor="white")
        ax2 = fig.add_subplot(132, facecolor="white")
        ax3 = fig.add_subplot(133, facecolor="white")

        cmap_vis = plt.cm.get_cmap("jet").copy()
        cmap_vis.set_bad(color="lightgray")
        im1 = ax1.imshow(I_fft_log, extent=extent, origin='lower', cmap=cmap_vis, vmin=vmin_fft, vmax=vmax_fft)
        c1 = plt.Circle((0,0), 1.0, color='black', linestyle='--', fill=False, alpha=0.6)
        ax1.add_patch(c1)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Raw 2D FFT (log10 normalized)")
        ax1.set_xlabel("kx / k0")
        ax1.set_ylabel("ky / k0")
        theta_ticks = [0, 15, 30, 45, 60, 75, 90]
        theta_grid = [np.sin(np.radians(t)) for t in theta_ticks]
        for t, s in zip(theta_ticks, theta_grid):
            ax1.add_patch(plt.Circle((0,0), s, color='gray', linestyle=':', fill=False, alpha=0.4, linewidth=0.6))
            ax1.text(s/np.sqrt(2), s/np.sqrt(2), f"{t}°", fontsize=8, color='gray')

        im2 = ax2.imshow(I_ang_log, extent=extent, origin='lower', cmap=cmap_vis, vmin=vmin_ang, vmax=vmax_ang)
        c2 = plt.Circle((0,0), 1.0, color='black', linestyle='--', fill=False, alpha=0.6)
        ax2.add_patch(c2)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title("Propagating modes (log10 normalized)")
        ax2.set_xlabel("kx / k0")
        ax2.set_ylabel("ky / k0")
        for t, s in zip(theta_ticks, theta_grid):
            ax2.add_patch(plt.Circle((0,0), s, color='gray', linestyle=':', fill=False, alpha=0.4, linewidth=0.6))
            ax2.text(s/np.sqrt(2), s/np.sqrt(2), f"{t}°", fontsize=8, color='gray')

        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="log10(norm)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="log10(norm)")

        # Mueller comparison
        mu_path = Path(self.current_run_dir.get()) / "mueller_scatgrid"
        corr_txt = "No mueller_scatgrid available"
        if mu_path.exists():
            try:
                mu_raw = np.loadtxt(mu_path, skiprows=1)
                if mu_raw.ndim == 1: mu_raw = mu_raw.reshape(1, -1)
                th_deg = mu_raw[:, 0]
                ph_deg = mu_raw[:, 1]
                s11 = mu_raw[:, 2]
                th_rad = np.radians(th_deg)
                ph_rad = np.radians(ph_deg)

                th_u = np.unique(th_rad)
                ph_u = np.unique(ph_rad)
                TH, PH = np.meshgrid(th_u, ph_u)
                mu_grid = np.zeros_like(TH)
                for t, p, s in zip(th_rad, ph_rad, s11):
                    ti = np.where(th_u == t)[0][0]
                    pi = np.where(ph_u == p)[0][0]
                    mu_grid[pi, ti] = s
                mu_log = np.log10(mu_grid / mu_grid.max() + 1e-12)

                # 将自算角谱映射到 mueller_scatgrid 的 theta-phi 网格
                # NTFF结果：kx/k0, ky/k0 -> theta (from z-axis), phi (azimuthal)
                # mueller_scatgrid已经是lab frame，不需要转换
                k_xy = np.sqrt(KX**2 + KY**2)
                theta_ntff_rad = np.arcsin(np.clip(k_xy / k0, 0, 1))  # [0, π/2] radians
                phi_ntff_rad = np.mod(np.arctan2(KY, KX), 2 * np.pi)  # [0, 2π] radians
                
                # Only use propagating modes (upper hemisphere: theta 0-90 degrees)
                valid = mask_prop & np.isfinite(theta_ntff_rad) & np.isfinite(phi_ntff_rad) & np.isfinite(I_ang_log) & (theta_ntff_rad >= 0) & (theta_ntff_rad <= np.pi/2)
                
                # Map NTFF to mueller grid (both in lab frame, radians)
                pts_ntff = np.column_stack([theta_ntff_rad[valid], phi_ntff_rad[valid]])
                vals_ntff = I_ang_log[valid]
                
                # 直接映射到mueller的theta/phi网格（都是lab frame，弧度单位）
                ours_on_mu = griddata(pts_ntff, vals_ntff, (TH, PH), method="linear", fill_value=np.nan)

                vmin_mu, vmax_mu = dyn_limits(ours_on_mu, db=6)
                vmin_mu, vmax_mu = dyn_limits(ours_on_mu, db=6)
                im3 = ax3.imshow(ours_on_mu, extent=[0, np.pi, 0, 2*np.pi], origin="lower",
                                 cmap=cmap_vis, vmin=vmin_mu, vmax=vmax_mu)
                ax3.set_title("Angle spectrum mapped to (theta,phi)")
                ax3.set_xlabel("theta (deg)")
                ax3.set_ylabel("phi (deg)")
                ax3.set_xticks(np.linspace(0, np.pi, 7))
                ax3.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 180, 7)])
                ax3.set_yticks(np.linspace(0, 2*np.pi, 9))
                ax3.set_yticklabels([f"{int(x)}" for x in np.linspace(0, 360, 9)])
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="log10(norm)")

                # correlation on overlap points
                mask_common = np.isfinite(ours_on_mu)
                mu_vals = mu_log[mask_common]
                our_vals = ours_on_mu[mask_common]
                if mu_vals.size > 10:
                    corr, _ = pearsonr(mu_vals.ravel(), our_vals.ravel())
                    corr_txt = f"Pearson corr vs. mueller S11 (log10): {corr:.3f}"
                else:
                    corr_txt = "Not enough overlap points for correlation"
            except Exception as exc:
                corr_txt = f"Mueller comparison failed: {exc}"
        else:
            ax3.axis("off")

        self.status_var.set(corr_txt)

        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, top)
        
    def update_int_plots(self):
        if not self.int_data: return
        
        # 1. Get Slice Data
        idx = self.current_z_idx.get()
        idx = max(0, min(idx, len(self.layers)-1))
        z_val = self.layers[idx]
        self.lbl_z.config(text=f"Z = {z_val:.4f}")
        
        mask = self._get_slice_mask(z_val)
        if not np.any(mask):
            messagebox.showwarning("切片为空", f"Z={z_val:.4f} 没有数据点")
            return
        x = self.int_data['x'][mask]
        y = self.int_data['y'][mask]
        v = self.int_data['I'][mask]
        ex = self.int_data['Ex'][mask]
        ey = self.int_data['Ey'][mask]
        
        self.ax2d.clear()
        
        # 2. Rendering
        method = self.interp_method.get()
        cmap = self.cmap_var.get()
        
        if method == "none":
            # Scatter
            im = self.ax2d.scatter(x, y, c=v, cmap=cmap, s=50, marker='s')
        else:
            # Interpolation
            # Create grid
            xi = np.linspace(min(x), max(x), 100)
            yi = np.linspace(min(y), max(y), 100)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((x, y), v, (XI, YI), method=method)
            
            im = self.ax2d.imshow(ZI, extent=[min(x),max(x),min(y),max(y)], origin='lower', cmap=cmap, aspect='equal')
            
            if self.show_contours.get():
                self.ax2d.contour(XI, YI, ZI, colors='white', alpha=0.3, linewidths=0.5)

        # 3. Vectors
        if self.show_vectors.get():
            # For quiver, we might need to downsample if dense
            # For streamplot, we need grid
            ex_r = np.real(ex)
            ey_r = np.real(ey)
            
            if self.vector_type.get() == "stream" and method != "none":
                # Grid vectors
                EXI = griddata((x, y), ex_r, (XI, YI), method='linear')
                EYI = griddata((x, y), ey_r, (XI, YI), method='linear')
                # Mask nans
                mask_nan = np.isnan(EXI) | np.isnan(EYI)
                EXI[mask_nan] = 0
                EYI[mask_nan] = 0
                
                self.ax2d.streamplot(xi, yi, EXI, EYI, color='white', density=1.0, linewidth=0.8, arrowsize=1.0)
            else:
                # Quiver (scattered)
                self.ax2d.quiver(x, y, ex_r, ey_r, color='white', alpha=0.8)

        self.ax2d.set_title(f"Near Field |E|^2 @ Z={z_val}")
        self.canvas_int_2d = self.cv2d
        self.cv2d.draw()

    def _get_slice_mask(self, z_val: float):
        """宽容匹配切片，避免浮点误差导致整层丢失。"""
        z_all = self.int_data['z']
        mask = np.isclose(z_all, z_val, atol=1e-3)
        if np.any(mask):
            return mask
        # 若仍为空，选距离最近的层
        idx_near = np.argmin(np.abs(z_all - z_val))
        near_val = z_all[idx_near]
        return np.isclose(z_all, near_val, atol=1e-6)

    def update_int_3d(self):
        """3D volume visualization using layered surfaces (exactly like visualize_all.py)"""
        if not self.int_data: return
        self.ax3d.clear()
        
        x = self.int_data['x']
        y = self.int_data['y']
        z = self.int_data['z']
        v = self.int_data['I']
        
        # Convert to indices (exactly like visualize_all.py)
        coords = np.column_stack([x, y, z])
        ix, iy, iz, nx_act, ny_act, nz_act = self._coords_to_indices(coords)
        
        # Build 3D array
        e_3d = np.zeros((nx_act, ny_act, nz_act))
        for i in range(len(ix)):
            if 0 <= ix[i] < nx_act and 0 <= iy[i] < ny_act and 0 <= iz[i] < nz_act:
                e_3d[int(ix[i]), int(iy[i]), int(iz[i])] = v[i]
        
        # Create meshgrid for coordinates (exactly like visualize_all.py)
        x_coords_3d, y_coords_3d, z_coords_3d = np.meshgrid(
            np.arange(nx_act), np.arange(ny_act), np.arange(nz_act), indexing="ij"
        )
        
        # Select layers to show (exactly like visualize_all.py)
        layers_to_show = list(range(0, nz_act, max(1, max(nz_act // 15, 1))))
        
        # Get colormap (use jet as default like visualize_all.py, but allow user choice)
        cmap_name = self.cmap_var.get()
        cmap_obj = plt.get_cmap(cmap_name)
        
        # Plot each layer as a surface (exactly like visualize_all.py)
        for z_layer in layers_to_show:
            e_layer = e_3d[:, :, z_layer]
            x_flat = x_coords_3d[:, :, z_layer].flatten()
            y_flat = y_coords_3d[:, :, z_layer].flatten()
            e_flat = e_layer.flatten()
            
            mask = e_flat > 0
            if not np.any(mask):
                continue
            
            x_valid = x_flat[mask]
            y_valid = y_flat[mask]
            e_valid = e_flat[mask]
            
            # Interpolate to denser grid (exactly like visualize_all.py)
            interp_factor = 2
            x_dense = np.linspace(0, nx_act - 1, nx_act * interp_factor)
            y_dense = np.linspace(0, ny_act - 1, ny_act * interp_factor)
            x_dense_grid, y_dense_grid = np.meshgrid(x_dense, y_dense)
            
            e_interp = griddata(
                np.column_stack([x_valid, y_valid]),
                e_valid,
                (x_dense_grid, y_dense_grid),
                method="cubic",
                fill_value=0,
            )
            
            # Normalize this layer (exactly like visualize_all.py)
            e_min = e_interp[e_interp > 0].min() if np.any(e_interp > 0) else 0
            e_max = e_interp.max()
            if e_max > e_min:
                e_norm = (e_interp - e_min) / (e_max - e_min)
            else:
                e_norm = np.zeros_like(e_interp)
            
            # Get colors (exactly like visualize_all.py: plt.cm.jet(e_norm))
            # But use user-selected colormap instead of hardcoded jet
            colors = cmap_obj(e_norm)
            
            # Create z surface at this layer
            z_surface = np.full_like(x_dense_grid, z_layer)
            
            # Plot surface (exactly like visualize_all.py)
            self.ax3d.plot_surface(
                x_dense_grid,
                y_dense_grid,
                z_surface,
                facecolors=colors,
                alpha=self.alpha_val.get(),
                shade=False,
                edgecolor="none",
                linewidth=0,
                antialiased=True,
            )
        
        self.ax3d.set_xlabel("X index")
        self.ax3d.set_ylabel("Y index")
        self.ax3d.set_zlabel("Z index")
        self.ax3d.set_title("3D Internal Field Volume")
        self.ax3d.set_xlim([0, nx_act - 1])
        self.ax3d.set_ylim([0, ny_act - 1])
        self.ax3d.set_zlim([0, nz_act - 1])
        self.ax3d.view_init(elev=20, azim=-45)
        self.ax3d.set_box_aspect([1, 1, 1])
        self.cv3d.draw()
    
    def _coords_to_indices(self, coords):
        """Convert coordinates to indices (from visualize_all.py)"""
        x_unique = sorted(np.unique(coords[:, 0]))
        y_unique = sorted(np.unique(coords[:, 1]))
        z_unique = sorted(np.unique(coords[:, 2]))
        x_map = {val: idx for idx, val in enumerate(x_unique)}
        y_map = {val: idx for idx, val in enumerate(y_unique)}
        z_map = {val: idx for idx, val in enumerate(z_unique)}
        
        ix = np.array([x_map[x] for x in coords[:, 0]])
        iy = np.array([y_map[y] for y in coords[:, 1]])
        iz = np.array([z_map[z] for z in coords[:, 2]])
        
        return ix, iy, iz, len(x_unique), len(y_unique), len(z_unique)

    def update_scat_plots(self):
        if not self.scat_data: 
            messagebox.showwarning("No data", "Please load mueller_scatgrid data first.")
            return
        
        self.ax_scat.clear()
        
        el = self.scat_element.get()
        # Parse Sij
        row = int(el[1])-1
        col = int(el[2])-1
        idx = row*4 + col

        val = self.scat_data['m'][:, idx]
        # Use lab frame coordinates
        theta_lab = self.scat_data['theta']
        phi_lab = self.scat_data['phi']
        
        # Only show upper hemisphere (like visualize_all.py)
        mask_upper = theta_lab <= 90
        if not np.any(mask_upper):
            messagebox.showwarning("No upper hemisphere data", "No data points in upper hemisphere (theta <= 90)")
            return
        
        theta_upper = theta_lab[mask_upper]
        phi_upper = phi_lab[mask_upper]
        val_upper = val[mask_upper]

        # Choose scaling and colormap
        cmap_name = self.cmap_var.get()
        if el == "S11":
            plot_vals = val_upper  # linear S11
            vmin, vmax = plot_vals.min(), plot_vals.max()
            if vmax <= vmin:
                vmax = vmin + 1e-10
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            plot_vals = val_upper
            vmax = np.max(np.abs(plot_vals))
            if vmax <= 0:
                vmax = 1e-10
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap_name = "coolwarm"

        # Surface interpolation over upper hemisphere grid (like visualize_all.py)
        theta_min, theta_max = theta_upper.min(), theta_upper.max()
        phi_min, phi_max = phi_upper.min(), phi_upper.max()
        n_theta = min(91, len(np.unique(theta_upper)))
        n_phi = min(361, len(np.unique(phi_upper)))
        
        theta_grid = np.linspace(theta_min, theta_max, n_theta)
        phi_grid = np.linspace(phi_min, phi_max, n_phi)
        theta_grid_mesh, phi_grid_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        
        points = np.column_stack([theta_upper, phi_upper])
        # Use fill_value=np.nan to better handle missing data
        surf_vals = griddata(points, plot_vals, (theta_grid_mesh, phi_grid_mesh), method="linear", fill_value=np.nan)
        
        # Mask out NaN values for plotting
        valid_mask = np.isfinite(surf_vals)
        if not np.any(valid_mask):
            messagebox.showwarning("No valid data", "Interpolation produced no valid data points")
            return
        
        # Convert to Cartesian for 3D surface
        theta_surf = np.radians(theta_grid_mesh)
        phi_surf = np.radians(phi_grid_mesh)
        XS = np.sin(theta_surf) * np.cos(phi_surf)
        YS = np.sin(theta_surf) * np.sin(phi_surf)
        ZS = np.cos(theta_surf)
        
        # Apply colormap, handling NaN values
        colors = plt.get_cmap(cmap_name)(norm(surf_vals))
        colors[..., -1] = 0.92  # Set alpha
        colors[~valid_mask, -1] = 0  # Make NaN regions transparent

        self.ax_scat.plot_surface(XS, YS, ZS, rstride=1, cstride=1,
                                  facecolors=colors, linewidth=0, antialiased=True, shade=False)
        self.ax_scat.set_title(f"3D {el} Scattering Pattern (Lab Frame, Upper Hemisphere)")
        self.ax_scat.set_axis_off()
        self.ax_scat.set_box_aspect([1,1,1])
        try:
            self.ax_scat.set_proj_type("ortho")
        except Exception:
            pass
        
        # Clear previous colorbars/axes
        for a in list(self.fig_scat.axes):
            if a is not self.ax_scat:
                try:
                    self.fig_scat.delaxes(a)
                except Exception:
                    pass
        if hasattr(self, "_scat_cbar") and self._scat_cbar:
            try:
                self._scat_cbar.remove()
            except Exception:
                pass
            self._scat_cbar = None
        
        # Create colorbar with proper mappable
        mappable = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
        mappable.set_array(surf_vals[valid_mask])  # Set array with valid values only
        self._scat_cbar = self.fig_scat.colorbar(mappable, ax=self.ax_scat, fraction=0.046, pad=0.04, label=el)
        self.status_var.set(f"{el}: min={plot_vals.min():.3e}, max={plot_vals.max():.3e} (linear)")
        self.cv_scat.draw()

    def compare_two_scat(self):
        """Load mueller_scatgrid from two runs and show side-by-side 3D surfaces in main plot."""
        run1 = Path(self.dual_run1.get().strip())
        run2 = Path(self.dual_run2.get().strip())
        if not run1.exists() or not run2.exists():
            messagebox.showwarning("Missing run", "Please enter valid run directories for both runs.")
            return

        def load_mueller(run_dir: Path, log_scale: bool):
            p = run_dir / "mueller_scatgrid"
            if not p.exists():
                raise FileNotFoundError(f"{p} not found")
            data = np.loadtxt(p, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Use original coordinates directly (no conversion, already in lab frame)
            s11 = data[:, 2]
            theta_lab = data[:, 0]
            phi_lab = data[:, 1]
            
            # Only upper hemisphere
            mask_upper = theta_lab <= 90
            theta_upper = theta_lab[mask_upper]
            phi_upper = phi_lab[mask_upper]
            s11_upper = s11[mask_upper]
            
            # Convert to Cartesian
            theta_rad = np.radians(theta_upper)
            phi_rad = np.radians(phi_upper)
            x = np.sin(theta_rad) * np.cos(phi_rad)
            y = np.sin(theta_rad) * np.sin(phi_rad)
            z = np.cos(theta_rad)
            
            if log_scale:
                s_val = np.log10(np.maximum(s11_upper, 1e-16))
            else:
                s_val = s11_upper
            return {"x": x, "y": y, "z": z, "s": s_val, "raw": s11_upper, "theta": theta_upper, "phi": phi_upper}

        try:
            d1 = load_mueller(run1, self.log_mueller.get())
            d2 = load_mueller(run2, self.log_mueller.get())
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        def make_surface(ax, data, title):
            # Use lab frame theta/phi directly (already converted and filtered to upper hemisphere)
            theta_lab = data["theta"]  # Already in degrees, upper hemisphere only
            phi_lab = data["phi"]  # Already in degrees
            s_vals = data["s"]
            
            # Create grid for upper hemisphere only (like visualize_all.py)
            theta_min, theta_max = theta_lab.min(), theta_lab.max()
            phi_min, phi_max = phi_lab.min(), phi_lab.max()
            n_theta = min(91, len(np.unique(theta_lab)))
            n_phi = min(361, len(np.unique(phi_lab)))
            
            theta_grid = np.linspace(theta_min, theta_max, n_theta)
            phi_grid = np.linspace(phi_min, phi_max, n_phi)
            theta_grid_mesh, phi_grid_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
            
            # Interpolate using lab frame coordinates
            points = np.column_stack([theta_lab, phi_lab])
            surf_vals = griddata(points, s_vals, (theta_grid_mesh, phi_grid_mesh), method="linear", fill_value=0)
            finite_mask = np.isfinite(surf_vals)
            fill_val = np.nanmin(surf_vals) if np.any(finite_mask) else np.nanmin(s_vals)
            surf_vals = np.where(finite_mask, surf_vals, fill_val)
            
            # Create independent norm for this surface
            vmin = np.nanmin(surf_vals)
            vmax = np.nanmax(surf_vals)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            
            # Convert to Cartesian for 3D surface
            theta_surf = np.radians(theta_grid_mesh)
            phi_surf = np.radians(phi_grid_mesh)
            X = np.sin(theta_surf) * np.cos(phi_surf)
            Y = np.sin(theta_surf) * np.sin(phi_surf)
            Z = np.cos(theta_surf)
            
            colors = cm.get_cmap("jet")(norm(surf_vals))
            colors[..., -1] = 0.92
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                            linewidth=0, antialiased=True, shade=False, edgecolor="none")
            ax.set_axis_off()
            ax.set_title(title)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
            return norm, surf_vals

        # clear figure
        self.fig_scat.clf()
        ax1 = self.fig_scat.add_subplot(121, projection="3d")
        ax2 = self.fig_scat.add_subplot(122, projection="3d")
        norm1, surf_vals1 = make_surface(ax1, d1, f"{run1.name}")
        norm2, surf_vals2 = make_surface(ax2, d2, f"{run2.name}")
        
        label = "log10(S11)" if self.log_mueller.get() else "S11"
        # Create independent colorbars for each surface
        mappable1 = cm.ScalarMappable(norm=norm1, cmap="jet")
        mappable1.set_array(surf_vals1)
        mappable2 = cm.ScalarMappable(norm=norm2, cmap="jet")
        mappable2.set_array(surf_vals2)
        self.fig_scat.colorbar(mappable1, ax=ax1, fraction=0.046, pad=0.04, label=label)
        self.fig_scat.colorbar(mappable2, ax=ax2, fraction=0.046, pad=0.04, label=label)
        self.cv_scat.draw()
        self.status_var.set(f"Run1: {len(d1['s'])} pts; Run2: {len(d2['s'])} pts; log={self.log_mueller.get()}")

    def _safe_compare_default(self):
        try:
            self.compare_two_scat()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ADDAVisualizerApp(root)
    root.mainloop()
