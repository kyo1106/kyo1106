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
        
        # --- NTFF Comparison (2D Overlay) ---
        ttk.Label(ctrl, text="Analysis:").pack(side="left", padx=(20, 5))
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

    def reload_scat_data(self):
        d = self.current_run_dir.get()
        p = Path(d) / "mueller_scatgrid"
        if not p.exists():
            self.scat_data = None
            self.status_var.set("未找到 mueller_scatgrid")
            return
        try:
            raw = np.loadtxt(p, skiprows=1)
            if raw.ndim == 1: raw = raw.reshape(1,-1)
            self.scat_data = {'theta': raw[:,0], 'phi': raw[:,1], 'm': raw[:,2:]}
            self.update_scat_plots()
        except Exception as exc:
            self.scat_data = None
            messagebox.showerror("远场读取失败", str(exc))

    def calc_ntff_compare(self):
        """Angle-spectrum far-field from current Z slice, with Mueller comparison."""
        if not self.int_data:
            messagebox.showwarning("Missing data", "Load IntField first.")
            return

        # 当前 Z 切片数据
        idx = self.current_z_idx.get()
        idx = max(0, min(idx, len(self.layers)-1))
        z_val = self.layers[idx]
        mask = self._get_slice_mask(z_val)
        if not np.any(mask):
            messagebox.showwarning("Empty slice", f"Z={z_val:.4f} has no points")
            return

        x_slice = self.int_data['x'][mask]
        y_slice = self.int_data['y'][mask]
        Ex_slice = self.int_data['Ex'][mask]
        Ey_slice = self.int_data['Ey'][mask]

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

        # 2D FFT
        Ex_k = np.fft.fftshift(np.fft.fft2(Ex_grid))
        Ey_k = np.fft.fftshift(np.fft.fft2(Ey_grid))
        I_fft = np.abs(Ex_k)**2 + np.abs(Ey_k)**2
        # k axes
        kx_vals = np.fft.fftshift(np.fft.fftfreq(Ex_grid.shape[1], d=dx)) * 2 * np.pi
        ky_vals = np.fft.fftshift(np.fft.fftfreq(Ex_grid.shape[0], d=dy)) * 2 * np.pi
        KX, KY = np.meshgrid(kx_vals, ky_vals)

        # propagating mask and weighting
        mask_prop = (KX**2 + KY**2) <= k0**2
        kz = np.zeros_like(KX)
        kz[mask_prop] = np.sqrt(k0**2 - KX[mask_prop]**2 - KY[mask_prop]**2)
        weight = np.zeros_like(I_fft)
        weight[mask_prop] = (kz[mask_prop] / k0)**2
        I_ang = I_fft * weight

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
        top.title(f"Far-field spectrum (Z={z_val:.4f}, λ={wl})")
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

                # 将自算角谱映射到 theta-phi 网格
                theta_map = np.arcsin(np.sqrt(KX**2 + KY**2) / k0)
                phi_map = np.mod(np.arctan2(KY, KX), 2 * np.pi)
                valid = np.isfinite(theta_map) & np.isfinite(I_ang_log)
                pts = np.stack([theta_map[valid], phi_map[valid]], axis=1)
                vals = I_ang_log[valid]
                ours_on_mu = griddata(pts, vals, (TH, PH), method="linear", fill_value=np.nan)

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
        if not self.int_data: return
        self.ax3d.clear()
        x = self.int_data['x']
        y = self.int_data['y']
        z = self.int_data['z']
        v = self.int_data['I']

        step = max(1, len(x)//5000)
        sc = self.ax3d.scatter(x[::step], y[::step], z[::step],
                               c=v[::step],
                               cmap=self.cmap_var.get(),
                               alpha=self.alpha_val.get(),
                               s=self.pt_size.get(),
                               norm=matplotlib.colors.Normalize(vmin=v.min(), vmax=v.max()))

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("3D Volumetric Cloud")
        self.ax3d.set_box_aspect([1, 1, 1])
        self.cv3d.draw()

    def update_scat_plots(self):
        if not self.scat_data: return
        self.ax_scat.clear()
        
        el = self.scat_element.get()
        # Parse Sij
        row = int(el[1])-1
        col = int(el[2])-1
        idx = row*4 + col

        val = self.scat_data['m'][:, idx]
        # Spherical -> Cartesian
        tr = np.radians(self.scat_data['theta'])
        pr = np.radians(self.scat_data['phi'])
        X = np.sin(tr) * np.cos(pr)
        Y = np.sin(tr) * np.sin(pr)
        Z = np.cos(tr)

        # Choose scaling and colormap
        cmap_name = self.cmap_var.get()
        if el == "S11":
            plot_vals = val  # linear S11
            norm = matplotlib.colors.Normalize(vmin=plot_vals.min(), vmax=plot_vals.max())
        else:
            plot_vals = val
            vmax = np.max(np.abs(plot_vals))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap_name = "coolwarm"

        # Surface interpolation over full sphere grid (lower res for speed)
        th_grid = np.linspace(0, np.pi, 91)
        ph_grid = np.linspace(0, 2*np.pi, 181)
        TH, PH = np.meshgrid(th_grid, ph_grid)
        surf_vals = griddata((tr, pr), plot_vals, (TH, PH), method="nearest")
        finite_mask = np.isfinite(surf_vals)
        fill_val = np.nanmin(surf_vals) if np.any(finite_mask) else plot_vals.min()
        surf_vals = np.where(finite_mask, surf_vals, fill_val)
        XS = np.sin(TH) * np.cos(PH)
        YS = np.sin(TH) * np.sin(PH)
        ZS = np.cos(TH)
        colors = plt.get_cmap(cmap_name)(norm(surf_vals))
        colors[..., -1] = 0.92

        self.ax_scat.plot_surface(XS, YS, ZS, rstride=1, cstride=1,
                                  facecolors=colors, linewidth=0, antialiased=True, shade=False)
        self.ax_scat.set_title(f"3D {el} Scattering Pattern")
        self.ax_scat.set_axis_off()
        self.ax_scat.set_box_aspect([1,1,1])
        try:
            self.ax_scat.set_proj_type("ortho")
        except Exception:
            pass
        # clear previous colorbars/axes
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
        mappable = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
        mappable.set_array([])
        self._scat_cbar = self.fig_scat.colorbar(mappable, ax=self.ax_scat, fraction=0.046, pad=0.04, label=el)
        self.status_var.set(f"{el}: min={val.min():.3e}, max={val.max():.3e} (linear)")
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
            theta = np.radians(data[:, 0])
            phi = np.radians(data[:, 1])
            s11 = data[:, 2]
            # ADDA mueller_scatgrid is already in lab frame (theta,phi)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            if log_scale:
                s_val = np.log10(np.maximum(s11, 1e-16))
            else:
                s_val = s11
            return {"x": x, "y": y, "z": z, "s": s_val, "raw": s11, "theta": theta, "phi": phi}

        try:
            d1 = load_mueller(run1, self.log_mueller.get())
            d2 = load_mueller(run2, self.log_mueller.get())
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        s_all = np.concatenate([d1["s"], d2["s"]])
        vmin = np.nanmin(s_all)
        vmax = np.nanmax(s_all)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        shared_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        def make_surface(ax, data, title, norm):
            # plot spherical surface via interpolation on unit sphere grid
            # coarser grid to speed up rendering
            th_grid = np.linspace(0, np.pi, 61)
            ph_grid = np.linspace(0, 2*np.pi, 121)
            TH, PH = np.meshgrid(th_grid, ph_grid)
            # map lab xyz to theta/phi for interpolation
            theta_lab = np.arccos(np.clip(data["z"], -1, 1))
            phi_lab = np.mod(np.arctan2(data["y"], data["x"]), 2*np.pi)
            surf_vals = griddata((theta_lab, phi_lab), data["s"], (TH, PH), method="linear")
            finite_mask = np.isfinite(surf_vals)
            fill_val = np.nanmin(surf_vals) if np.any(finite_mask) else vmin
            surf_vals = np.where(finite_mask, surf_vals, fill_val)
            X = np.sin(TH) * np.cos(PH)
            Y = np.sin(TH) * np.sin(PH)
            Z = np.cos(TH)
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

        # clear figure
        self.fig_scat.clf()
        ax1 = self.fig_scat.add_subplot(121, projection="3d")
        ax2 = self.fig_scat.add_subplot(122, projection="3d")
        make_surface(ax1, d1, f"{run1.name}", shared_norm)
        make_surface(ax2, d2, f"{run2.name}", shared_norm)
        label = "log10(S11)" if self.log_mueller.get() else "S11"
        mappable1 = cm.ScalarMappable(norm=shared_norm, cmap="jet")
        mappable2 = cm.ScalarMappable(norm=shared_norm, cmap="jet")
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
