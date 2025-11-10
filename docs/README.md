# AVO Seismic Modeling Dashboard

Modern web dashboard for AVO-based seismic modeling analysis using the Stanford VI-E dataset.

## 🌐 View Online

**Live Site**: https://mazzutti.github.io/Stanford-VI-E/

## 🚀 Setup GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Source: **Branch: master**, **Folder: /docs**
3. Click **Save**
4. Site will be live in 1-2 minutes

## 🚀 AVO Analysis Overview

**Technique**: Full-Stack AVO Seismogram

**Key Features:**
- 4 angle stacks: 0°, 15°, 22.5°, 30°
- Aki-Richards linearization of Zoeppritz equations
- 25 Hz Ricker wavelet
- Full-stack integration for comprehensive analysis
- **Dual-domain output**: True seismograms in BOTH time (149 samples) and depth (200 layers)
- Perfect alignment between depth seismograms and rock physics attributes

**Applications:**
- Angle-dependent amplitude analysis
- Facies discrimination
- Reservoir characterization
- Fluid and lithology identification

## 🔧 Update Results

```bash
# Run AVO modeling (generates both time & depth domain seismograms)
python -m src

# Generate seismic plots (time domain)
python generate_plots.py

# Generate seismic plots (depth domain)
python generate_plots_depth.py

# Generate rock physics attribute plots
python -m src --run-tool plot_rock_physics_attributes

# Push to GitHub
git add docs/
git commit -m "Update AVO results"
git push origin master
```

## 📱 Features

- Modern data science design
- Depth & time domain visualization
- Angle stack analysis
- Fully responsive (mobile/tablet/desktop)

---

**Repository**: https://github.com/mazzutti/Stanford-VI-E  
**Updated**: November 2024
