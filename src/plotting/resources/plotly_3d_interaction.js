/**
 * Plotly 3D Interaction Script
 * 
 * Provides responsive interaction features for 3D Plotly visualizations:
 * - Aggressive wheel zoom (configurable sensitivity)
 * - Dynamic colorbar scaling on zoom and window resize
 * - Persistent axis titles and reversed Z-axis
 * - Full-screen responsive layout
 * 
 * Configuration constants (injected from Python):
 * - _WHEEL_ZOOM_SENSITIVITY: Zoom multiplier (default: 2.5)
 * - _COLORBAR_MIN_LEN: Minimum colorbar length (default: 0.15)
 * - _COLORBAR_MAX_LEN: Maximum colorbar length (default: 0.95)
 * - _COLORBAR_DEFAULT_LEN: Default colorbar length (default: 0.7)
 * - _RESIZE_THROTTLE_MS: Resize event throttle in ms (default: 300)
 * - _RETRY_ATTEMPTS: Title capture retry attempts (default: 5)
 */

// Simple fullscreen setup
function setupFullscreen() {
  try {
    const div = document.querySelector('.plotly-graph-div');
    if (div) {
      div.style.height = '100%';
      div.style.width = '100%';
      div.style.margin = '0';
      div.style.padding = '0';
    }
    
    // Also ensure body and html take up full space
    const htmlEl = document.documentElement;
    const bodyEl = document.body;
    if (htmlEl) {
      htmlEl.style.height = '100%';
      htmlEl.style.width = '100%';
      htmlEl.style.margin = '0';
      htmlEl.style.padding = '0';
      htmlEl.style.overflow = 'hidden';
    }
    if (bodyEl) {
      bodyEl.style.height = '100%';
      bodyEl.style.width = '100%';
      bodyEl.style.margin = '0';
      bodyEl.style.padding = '0';
      bodyEl.style.overflow = 'hidden';
      bodyEl.style.display = 'flex';
      bodyEl.style.flexDirection = 'column';
    }
  } catch(e) { }
}

// Ensure fullscreen on load
document.addEventListener('DOMContentLoaded', setupFullscreen);
setTimeout(setupFullscreen, 100);

// Enhanced zoom control - more aggressive zoom in/out
function enableAggressiveZoom() {
  const div = document.querySelector('.plotly-graph-div');
  if (!div) return;
  
  // Store the default wheel scale - INCREASED for much more aggressive zoom
  let wheelScale = __WHEEL_ZOOM_SENSITIVITY; // Much more aggressive - __WHEEL_ZOOM_SENSITIVITY x sensitivity
  
  // Centralized wheel handling: zoom camera + recreate depth trace with scaled colorbar.
  // Recreating the trace (vs restyle) forces Plotly's renderer to redraw the colorbar visually.
  function performWheelZoom(e, source) {
    // Get current camera eye position
    const sceneCamera = (div && div._fullLayout && div._fullLayout.scene && div._fullLayout.scene.camera) ? div._fullLayout.scene.camera : null;
    if (sceneCamera) {
      const camera = sceneCamera;
      const eye = camera.eye;

      // Calculate zoom direction based on wheel delta - more aggressive multiplier
      const zoomFactor = e.deltaY > 0 ? (1 + (wheelScale - 1) * 0.5) : (1 - (wheelScale - 1) * 0.5);

      // New camera position (zoom towards/away from center)
      const newEye = {
        x: eye.x / zoomFactor,
        y: eye.y / zoomFactor,
        z: eye.z / zoomFactor
      };

      // Apply camera change via relayout (use promise API when available)
      try {
        const rel = Plotly.relayout(div, {'scene.camera.eye': newEye});
      } catch(e) {
        // Silently handle relayout errors
      }

      // Scale colorbar by recreating trace with new colorbar.len
      if (window._originalColorbars && Array.isArray(window._originalColorbars)) {
        window._originalColorbars.forEach(function(cb) {
          try {
            const traceIdx = cb.index;
            if (div && div.data && div.data[traceIdx]) {
              const origLen = (cb && cb.len) ? cb.len : __COLORBAR_DEFAULT_LEN;
              const newLen = Math.max(__COLORBAR_MIN_LEN, Math.min(__COLORBAR_MAX_LEN, origLen / zoomFactor));
              
              const currentTrace = div.data[traceIdx];
              const updatedTrace = JSON.parse(JSON.stringify(currentTrace));
              if (updatedTrace.colorbar) {
                updatedTrace.colorbar.len = newLen;
                Plotly.deleteTraces(div, [traceIdx]);
                Plotly.addTraces(div, [updatedTrace], [traceIdx]);
              }
            }
          } catch(e) {
            // Silently handle trace recreation errors
          }
        });
      }
    }
  }

  // Scale colorbar when window is resized (responsive scaling based on viewport)
  let lastWindowWidth = window.innerWidth;
  let lastWindowHeight = window.innerHeight;
  
  function scaleColorbarOnResize() {
    const currentWidth = window.innerWidth;
    const currentHeight = window.innerHeight;
    const widthRatio = currentWidth / (lastWindowWidth || currentWidth);
    const heightRatio = currentHeight / (lastWindowHeight || currentHeight);
    
    // Use geometric mean to scale colorbar proportionally
    const resizeRatio = Math.sqrt(widthRatio * heightRatio);
    
    if (window._originalColorbars && Array.isArray(window._originalColorbars)) {
      window._originalColorbars.forEach(function(cb) {
        try {
          const traceIdx = cb.index;
          if (div && div.data && div.data[traceIdx]) {
            const origLen = (cb && cb.len) ? cb.len : __COLORBAR_DEFAULT_LEN;
            const newLen = Math.max(__COLORBAR_MIN_LEN, Math.min(__COLORBAR_MAX_LEN, origLen * resizeRatio));
            
            // Clone and recreate trace with new colorbar length
            const currentTrace = div.data[traceIdx];
            const updatedTrace = JSON.parse(JSON.stringify(currentTrace));
            if (updatedTrace.colorbar) {
              updatedTrace.colorbar.len = newLen;
              Plotly.deleteTraces(div, [traceIdx]);
              Plotly.addTraces(div, [updatedTrace], [traceIdx]);
            }
          }
        } catch(e) {
          // Silently handle trace recreation errors
        }
      });
    }
    
    // Update last dimensions for next resize
    lastWindowWidth = currentWidth;
    lastWindowHeight = currentHeight;
  }
  
  // Throttle resize events (max once per __RESIZE_THROTTLE_MS ms)
  let resizeTimer = null;
  window.addEventListener('resize', function() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(scaleColorbarOnResize, __RESIZE_THROTTLE_MS);
  }, false);
}

// CRITICAL: Ensure Z-axis reversed and titles persist on all layout changes
const div = document.querySelector('.plotly-graph-div');
window._isApplyingFix = false;
// When true, restoreAxisProperties will reapply captured colorbar lengths.
// Default=false to allow dynamic colorbar scaling during user interactions.
window._forceRestoreColorbars = false;
window._originalTitles = null;

if (div) {
  // Enable aggressive zoom after a short delay to let Plotly initialize
  setTimeout(enableAggressiveZoom, 1000);
  
  // Extract original titles from Plotly's internal layout
  function captureOriginalTitles() {
    try {
      // Try _fullLayout first (where Plotly stores processed layout)
      let scene = (div._fullLayout && div._fullLayout.scene) ? div._fullLayout.scene : null;
      
      // Fallback to layout.scene
      if (!scene && div.layout && div.layout.scene) {
        scene = div.layout.scene;
      }
      
      if (!scene) {
        return false;
      }
      
      const xAxis = scene.xaxis;
      const yAxis = scene.yaxis;
      const zAxis = scene.zaxis;
      
      // Extract title - could be string or object with text property
      const xTitle = xAxis && xAxis.title ? (typeof xAxis.title === 'object' ? xAxis.title.text : xAxis.title) : null;
      const yTitle = yAxis && yAxis.title ? (typeof yAxis.title === 'object' ? yAxis.title.text : yAxis.title) : null;
      const zTitle = zAxis && zAxis.title ? (typeof zAxis.title === 'object' ? zAxis.title.text : zAxis.title) : null;
      
      if (xTitle && yTitle && zTitle) {
        window._originalTitles = {
          xaxis: {text: xTitle},
          yaxis: {text: yTitle},
          zaxis: {text: zTitle}
        };
        // Capture original colorbar lengths (if present) from traces in _fullData
        try {
          const fullData = div._fullData || div.data || [];
          const cbs = [];
          for (let i = 0; i < fullData.length; i++) {
            const t = fullData[i];
            if (t && t.colorbar) {
              // colorbar.len may be undefined; default to __COLORBAR_DEFAULT_LEN
              const len = t.colorbar.len || __COLORBAR_DEFAULT_LEN;
              cbs.push({index: i, len: len});
            }
          }
          if (cbs.length) {
            window._originalColorbars = cbs;
          }
        } catch(e) { }
        return true;
      } else {
        return false;
      }
    } catch(e) {
      return false;
    }
  }
  
  // Restore axis titles and Z-axis settings
  function restoreAxisProperties() {
    if (!window._originalTitles) {
      return;
    }
    
    const updates = {};
    
    // Restore Z-axis autorange
    updates['scene.zaxis.autorange'] = 'reversed';
    
    // Restore titles - handle both string and object formats
    if (window._originalTitles.xaxis) {
      // Plotly expects an object with .text property for titles
      if (typeof window._originalTitles.xaxis === 'object') {
        updates['scene.xaxis.title'] = window._originalTitles.xaxis;
      } else {
        updates['scene.xaxis.title'] = {text: window._originalTitles.xaxis};
      }
    }
    if (window._originalTitles.yaxis) {
      if (typeof window._originalTitles.yaxis === 'object') {
        updates['scene.yaxis.title'] = window._originalTitles.yaxis;
      } else {
        updates['scene.yaxis.title'] = {text: window._originalTitles.yaxis};
      }
    }
    if (window._originalTitles.zaxis) {
      if (typeof window._originalTitles.zaxis === 'object') {
        updates['scene.zaxis.title'] = window._originalTitles.zaxis;
      } else {
        updates['scene.zaxis.title'] = {text: window._originalTitles.zaxis};
      }
    }
    
    // Automatic colorbar restoration intentionally disabled.
    // Restoring colorbar lengths here would interfere with user-driven
    // dynamic scaling during wheel zoom or window resize.

    // Apply other layout updates (titles, z autorange)
    Plotly.relayout(div, updates);
  }
  
  // Try to capture titles immediately
  setTimeout(function() {
    if (!captureOriginalTitles()) {
      // Retry a few times if not ready
        let attempts = 0;
        const retryInterval = setInterval(function() {
          if (captureOriginalTitles() || attempts++ > __RETRY_ATTEMPTS) {
          clearInterval(retryInterval);
        }
      }, 100);
    }
  }, 500);
  
  div.on('plotly_relayout', function(data) {
    try {
      // Try to capture titles if we haven't yet
      if (!window._originalTitles) {
        captureOriginalTitles();
      }
      
      // PREVENT INFINITE LOOP
      if (window._isApplyingFix) {
        window._isApplyingFix = false;
        return;
      }
      
      const scene = div.layout ? div.layout.scene : null;
      if (!scene) {
        return;
      }
      
      // Check if properties need fixing
      let needsFix = false;
      
      // Check Z-axis autorange
      const zAutorange = scene.zaxis ? scene.zaxis.autorange : null;
      if (zAutorange !== 'reversed') {
        needsFix = true;
      }
      
      // Check axis titles
      const xTitle = scene.xaxis && scene.xaxis.title ? scene.xaxis.title : null;
      const yTitle = scene.yaxis && scene.yaxis.title ? scene.yaxis.title : null;
      const zTitle = scene.zaxis && scene.zaxis.title ? scene.zaxis.title : null;
      
      if ((xTitle === 'X' || xTitle === null) || 
          (yTitle === 'Y' || yTitle === null) || 
          (zTitle === 'Z' || zTitle === null)) {
        needsFix = true;
      }
      
      if (needsFix) {
        window._isApplyingFix = true;
        setTimeout(function() {
          restoreAxisProperties();
          setTimeout(function() { 
            window._isApplyingFix = false;
          }, 50);
        }, 10);
      }
    } catch(e) { 
      window._isApplyingFix = false;
    }
  });
}
