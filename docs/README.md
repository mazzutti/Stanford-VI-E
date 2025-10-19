# Seismic Modeling Results Dashboard

Modern web dashboard comparing AVO, AI, and EI seismic modeling techniques.

## 🌐 View Online

**Live Site**: https://mazzutti.github.io/Stanford-VI-E/

## 🚀 Setup GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Source: **Branch: master**, **Folder: /docs**
3. Click **Save**
4. Site will be live in 1-2 minutes

## 🚀 Key Results

| Technique | Cohen's d | Best For |
|-----------|-----------|----------|
| Full-Stack (AVO) | 0.474 | Angle analysis |
| AI Synthetic | 0.470 | Quick mapping |
| **Multi-Angle EI** | **14.046** | **Facies classification** |

**Highlights:**
- Multi-Angle EI is 29.7× better at facies discrimination
- 6 angles optimally stacked (0°, 5°, 10°, 15°, 20°, 25°)
- Excellent gradient correlation: r = 0.474 (157.9× better than AVO)

## 🔧 Update Results

```bash
# Run modeling
python -m src.modeling

# Generate plots
python -m src.plot_2d_slices --domain depth
python -m src.plot_2d_slices --domain time

# Push to GitHub
git add docs/
git commit -m "Update results"
git push origin master
```

## 📱 Features

- Modern data science design
- Interactive 3D viewers
- Depth & time domain switching
- Fully responsive (mobile/tablet/desktop)



---

**Repository**: https://github.com/mazzutti/Stanford-VI-E  
**Updated**: October 2025
