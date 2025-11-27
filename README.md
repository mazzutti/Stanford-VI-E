# Stanford-VI-E Seismic Modeling

AVO-based seismic modeling and rock physics analysis using the Stanford VI-E synthetic reservoir dataset.

**Live Documentation**: https://mazzutti.github.io/Stanford-VI-E/

## 📖 Dataset Reference

Stanford VI-E_Paper.pdf explains the details of Stanford VI-E data. If you use the data, reference the paper as follows: 

> Lee, J. and Mukerji, T., 2012, "The Stanford VI-E reservoir: A synthetic data set for joint seismic-EM time-lapse monitoring algorithms": 25th Annual Report, Stanford Center for Reservoir Forecasting, Stanford University, Stanford, CA.

The data are text files formatted by GSLIB conventions:  
http://scrf.stanford.edu/resources.software.gslib.help.02.php

---

## 🚀 Quick Start

### Run Complete AVO Modeling Pipeline
```bash
# Run full modeling workflow (AVO + rock physics)
python -m src

# Run with verbose output
python -m src --verbose

# Skip cache cleanup
python -m src --skip-cleanup
```

### Generate Visualizations
```bash
# Generate seismic AVO plots
python generate_plots.py

# Generate rock physics attribute plots
python -m src --run-tool plot_rock_physics_attributes --verbose

# Generate interactive 3D viewers
python -m src --run-tool plot_3d_interactive --domain depth
python -m src --run-tool plot_3d_interactive --domain time
```

---

## 🛠️ CLI Tools

The project includes integrated CLI tools for various analysis and visualization tasks:

### Visualization Tools

#### `plot_original_properties`
Generate visualizations of original Stanford VI-E properties (Vp, Vs, Rho) in 2D or 3D.

```bash
# 2D static PNG plots (default)
python -m src --run-tool plot_original_properties

# 3D interactive HTML plots with Plotly
python -m src --run-tool plot_original_properties --plot-type 3d

# With options
python -m src --run-tool plot_original_properties \
    --plot-type 2d \
    --output-dir docs/images \
    --data-dir . \
    --verbose
```

**Generates (2D mode):**
- P-wave velocity (Vp) plot with 3 orthogonal slices
- S-wave velocity (Vs) plot with 3 orthogonal slices
- Density (ρ) plot with 3 orthogonal slices
- Saves to `docs/images/original_*.png` (PNG format, ~1.5MB each)

**Generates (3D mode):**
- Interactive 3D visualization with orthogonal slices for each property
- Rotatable, zoomable views with hover information
- Saves to `docs/images/original_*_3d.html` (HTML format, ~5.7MB each)

#### `plot_rock_physics_attributes`
Generate static PNG visualizations of rock physics attributes (Lambda-Rho, Mu-Rho, AVO Intercept/Gradient).

```bash
# Basic usage
python -m src --run-tool plot_rock_physics_attributes

# With options
python -m src --run-tool plot_rock_physics_attributes \
    --domain depth \
    --output-dir docs/images \
    --verbose
```

**Generates:**
- Individual attribute plots (3-slice view each)
- Multi-attribute comparison plot
- Saves to `docs/images/rock_physics_*.png`

#### `plot_3d_interactive`
Generate interactive Plotly 3D viewers (HTML) for seismic volumes.

```bash
python -m src --run-tool plot_3d_interactive --domain depth
python -m src --run-tool plot_3d_interactive --domain time
```

#### `plot_3d_slices`
Generate 3D orthogonal slice visualizations.

```bash
python -m src --run-tool plot_3d_slices --domain depth
```

### Analysis Tools

#### `analysis_rock_physics`
Run complete rock physics analysis pipeline with cache clearing and visualization.

```bash
python -m src --run-tool analysis_rock_physics
```

#### `rock_physics_attributes`
Compute rock physics attributes and save to cache.

```bash
python -m src --run-tool rock_physics_attributes --domain depth
```

### Maintenance Tools

#### `cleanup_cache`
Clean up old cache files to free disk space.

```bash
# Dry run (see what would be deleted)
python -m src --run-tool cleanup_cache --dry-run

# Actually clean up
python -m src --run-tool cleanup_cache --cache-dir .cache
```

### List All Available Tools
```bash
python -m src --help
```

---

## 📁 Project Structure

```
Stanford-VI-E/
├── docs/                      # GitHub Pages documentation
│   ├── index.html            # Main dashboard
│   ├── images/               # Generated visualizations
│   └── views/                # Interactive 3D viewers
├── src/                      # Source code
│   ├── cli/                  # CLI tools and parsers
│   ├── modeling/             # Seismic modeling
│   ├── analysis/             # Rock physics analysis
│   ├── plotting/             # Visualization components
│   └── io/                   # Data I/O
├── .cache/                   # Computed results cache
├── generate_plots.py         # Seismic plot generation script
└── README.md                 # This file
```

---

## 🔬 Features

### AVO Seismic Modeling
- **4 angle stacks**: 0°, 15°, 22.5°, 30°
- **Aki-Richards linearization** of Zoeppritz equations
- **25 Hz Ricker wavelet** convolution
- Full-stack integration with quality weighting
- Depth and time domain analysis

### Rock Physics Attributes
- **Lambda-Rho (λρ)**: Fluid sensitivity indicator
- **Mu-Rho (μρ)**: Lithology sensitivity indicator  
- **AVO Intercept (A)**: Normal incidence reflectivity
- **AVO Gradient (B)**: Angle-dependent reflectivity

### Interactive Visualization
- Plotly-based 3D volume viewers
- Orthogonal slice navigation
- Modal image viewer for static plots
- Responsive web dashboard

---

## 📊 Visualization Gallery

Visit the live documentation to explore:
- Full-stack AVO seismograms
- Individual angle stack visualizations
- Rock physics attribute maps
- Interactive 3D volume viewers

**View Online**: https://mazzutti.github.io/Stanford-VI-E/

---

## 💻 Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Type checking
mypy src/

# Code coverage
pytest --cov=src --cov-report=html
```

---

## 📝 License

See repository for license details.

**Updated**: November 2025
