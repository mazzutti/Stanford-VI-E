// ================================================
// MODAL MANAGEMENT WITH ACCESSIBILITY
// ================================================

let lastFocusedElement = null;
let focusableElements = [];
let firstFocusableElement = null;
let lastFocusableElement = null;
// Holds the current modal keydown handler so it can be removed when modal closes
let modalKeydownHandler = null;

// Trap focus within modal
function trapFocus(modal) {
  focusableElements = modal.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  firstFocusableElement = focusableElements[0];
  lastFocusableElement = focusableElements[focusableElements.length - 1];

  // Remove previous handler if present
  try {
    if (modalKeydownHandler) {
      modal.removeEventListener('keydown', modalKeydownHandler);
      modalKeydownHandler = null;
    }
  } catch (err) {
    // ignore if previous modal reference isn't available
  }

  modalKeydownHandler = function (e) {
    const isTabPressed = e.key === 'Tab' || e.keyCode === 9;
    if (!isTabPressed) return;

    if (e.shiftKey) {
      if (document.activeElement === firstFocusableElement) {
        lastFocusableElement.focus();
        e.preventDefault();
      }
    } else {
      if (document.activeElement === lastFocusableElement) {
        firstFocusableElement.focus();
        e.preventDefault();
      }
    }
  };

  modal.addEventListener('keydown', modalKeydownHandler);
}

// Global modal and viewer functions
function openModal(imageSrc, altText) {
  const modal = document.getElementById("imageModal");
  const modalImg = document.getElementById("modalImage");
  if (modal && modalImg) {
    lastFocusedElement = document.activeElement;
    modal.classList.add("active");
    modalImg.src = imageSrc;
    modalImg.alt = altText || '';
    document.body.style.overflow = "hidden";
    trapFocus(modal);
    
    // Focus on close button
    const closeButton = modal.querySelector('.viewer-modal-close');
    if (closeButton) closeButton.focus();
  }
}

function openViewer(url) {
  const modal = document.getElementById("viewer3DModal");
  const iframe = document.getElementById("viewer3DIframe");
  if (modal && iframe) {
    lastFocusedElement = document.activeElement;
    // show loader and reset error
    const loader = document.getElementById('viewerIframeLoader');
    const errEl = document.getElementById('viewerIframeError');
    if (loader) loader.style.display = 'flex';
    if (errEl) errEl.style.display = 'none';

    iframe.src = url;
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
    trapFocus(modal);

    // Focus on close button
    const closeButton = modal.querySelector('.viewer-modal-close');
    if (closeButton) closeButton.focus();

    // Setup load/error handlers with timeout
    let loadHandled = false;
    const onLoad = function() {
      loadHandled = true;
      if (loader) loader.style.display = 'none';
      iframe.removeEventListener('load', onLoad);
      iframe.removeEventListener('error', onError);
      if (iframeLoadTimeout) {
        clearTimeout(iframeLoadTimeout);
        iframeLoadTimeout = null;
      }
    };
    const onError = function() {
      if (loader) loader.style.display = 'none';
      if (errEl) errEl.style.display = 'block';
      iframe.removeEventListener('load', onLoad);
      iframe.removeEventListener('error', onError);
      if (iframeLoadTimeout) {
        clearTimeout(iframeLoadTimeout);
        iframeLoadTimeout = null;
      }
    };

    iframe.addEventListener('load', onLoad);
    iframe.addEventListener('error', onError);

    // Timeout in case load never completes (network or blocked)
    var iframeLoadTimeout = setTimeout(function() {
      if (!loadHandled) {
        onError();
      }
    }, 8000);
  }
}

// Graceful fallback: global helper to open the seismogram viewer from inline onclick
function openSeismoViewer(buttonElement) {
  try {
    console.debug('openSeismoViewer called', buttonElement);
    if (!buttonElement) return;
    const domainToggleText = document.getElementById('domainToggleText');
    const domain = domainToggleText ? domainToggleText.textContent.trim().toLowerCase() : 'time';
    const timeUrl = buttonElement.getAttribute('data-time-viewer');
    const depthUrl = buttonElement.getAttribute('data-depth-viewer');

    if (domain === 'depth' && depthUrl) {
      openViewer(depthUrl);
    } else if (timeUrl) {
      openViewer(timeUrl);
    } else if (depthUrl) {
      openViewer(depthUrl);
    }
  } catch (err) {
    console.error('openSeismoViewer error:', err);
  }
}

function closeViewer() {
  const modal = document.getElementById("viewer3DModal");
  const iframe = document.getElementById("viewer3DIframe");
  if (modal && iframe) {
    modal.classList.remove("active");
    iframe.src = "";
    document.body.style.overflow = "";
    
    // Remove event listener
    if (modalKeydownHandler) {
      modal.removeEventListener('keydown', modalKeydownHandler);
      modalKeydownHandler = null;
    }
    
    // Return focus to last focused element
    if (lastFocusedElement) lastFocusedElement.focus();
    // reset loader and error UI
    const loader = document.getElementById('viewerIframeLoader');
    const errEl = document.getElementById('viewerIframeError');
    if (loader) loader.style.display = 'none';
    if (errEl) errEl.style.display = 'none';
  }
}

function openImageModal(imageSrc, imageAlt) {
  const modal = document.getElementById("imageModal");
  const img = document.getElementById("modalImage");
  if (modal && img) {
    lastFocusedElement = document.activeElement;
    img.src = imageSrc;
    img.alt = imageAlt || '';
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
    trapFocus(modal);
    
    // Focus on close button
    const closeButton = modal.querySelector('.viewer-modal-close');
    if (closeButton) closeButton.focus();
  }
}

function closeImageModal() {
  const modal = document.getElementById("imageModal");
  const img = document.getElementById("modalImage");
  if (modal && img) {
    modal.classList.remove("active");
    img.src = "";
    document.body.style.overflow = "";
    
    // Remove event listener
    if (modalKeydownHandler) {
      modal.removeEventListener('keydown', modalKeydownHandler);
      modalKeydownHandler = null;
    }
    
    // Return focus to last focused element
    if (lastFocusedElement) lastFocusedElement.focus();
  }
}

// Close modals with Escape key
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    const activeModal = document.querySelector('.viewer-modal.active');
    if (activeModal) {
      if (activeModal.id === 'viewer3DModal') {
        closeViewer();
      } else if (activeModal.id === 'imageModal') {
        closeImageModal();
      }
    }
  }
});

// ================================================
// SCROLL PROGRESS INDICATOR WITH DEBOUNCE
// ================================================

let scrollTimeout;
function updateScrollProgress() {
  const scrollIndicator = document.getElementById('scrollIndicator');
  if (scrollIndicator) {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollPercentage = (scrollTop / (documentHeight - windowHeight)) * 100;
    
    scrollIndicator.style.transform = `scaleX(${scrollPercentage / 100})`;
    scrollIndicator.setAttribute('aria-valuenow', Math.round(scrollPercentage));
  }
}

// Debounced scroll handler with passive listener for better performance
window.addEventListener('scroll', function() {
  if (scrollTimeout) {
    window.cancelAnimationFrame(scrollTimeout);
  }
  scrollTimeout = window.requestAnimationFrame(function() {
    updateScrollProgress();
    
    // Show/hide back-to-top button
    const backToTop = document.getElementById('backToTop');
    if (backToTop) {
      if (window.scrollY > 300) {
        backToTop.classList.add('visible');
      } else {
        backToTop.classList.remove('visible');
      }
    }
  });
}, { passive: true });

// ================================================
// DATE AND STATS UTILITIES
// ================================================

// Set current date in footer
const lastUpdatedElement = document.getElementById("lastUpdated");
if (lastUpdatedElement) {
  const now = new Date();
  const options = { year: "numeric", month: "long", day: "numeric" };
  lastUpdatedElement.textContent = now.toLocaleDateString("en-US", options);
}

// ================================================
// TAB NAVIGATION WITH ARIA SUPPORT
// ================================================

// Stats data for each tab
const statsData = {
  dataset: [
    { value: "3", label: "Primary Properties" },
    { value: "6M", label: "Grid Cells" },
    { value: "3", label: "Geological Layers" },
    { value: "25 m", label: "Horizontal Resolution" },
  ],
  seismic: [
    { value: "4", label: "Angle Stacks Generated" },
    { value: "6M", label: "Data Points Analyzed" },
    { value: "150×200×200", label: "Grid Dimensions" },
    { value: "2", label: "Domains (Depth/Time)" },
  ],
  rockphysics: [
    { value: "4", label: "Key Attributes" },
    { value: "14.046", label: "Best Cohen's d" },
    { value: "2", label: "Huge Effects" },
    { value: "4", label: "Angles Analyzed" },
  ],
};

// Function to update stats bar
function updateStatsBar(tabName) {
  const statsBar = document.getElementById("shared-stats-bar");
  const data = statsData[tabName];

  if (statsBar && data) {
    statsBar.innerHTML = data
      .map(
        (item) => `
      <div class="stat-item">
        <span class="stat-value">${item.value}</span>
        <span class="stat-label">${item.label}</span>
      </div>
    `
      )
      .join("");
  }
}

// Tab switching functionality
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", function () {
    const tabName = this.getAttribute("data-tab");

    // Remove active class from all buttons and tabs
    document
      .querySelectorAll(".tab-button")
      .forEach((btn) => {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
      });
    document
      .querySelectorAll(".tab-content")
      .forEach((content) => content.classList.remove("active"));

    // Add active class to clicked button and corresponding tab
    this.classList.add("active");
    this.setAttribute("aria-selected", "true");
    document.getElementById(tabName + "-tab").classList.add("active");

    // Update stats bar with new data
    updateStatsBar(tabName);

    // Show/hide domain toggle button - only visible in seismic tab
    const domainToggle = document.getElementById("domainToggleFixed");
    if (domainToggle) {
      if (tabName === "seismic") {
        domainToggle.classList.add("visible");
      } else {
        domainToggle.classList.remove("visible");
      }
    }

    // Note: do not auto-scroll when switching tabs — preserve user's current scroll position
  });
});

// ================================================
// DOM INITIALIZATION WITH ERROR HANDLING
// ================================================
document.addEventListener("DOMContentLoaded", function () {
  console.log("Initializing page...");

  try {
    // 1. Initialize stats bar with default active tab
    const activeTab = document.querySelector(".tab-button.active");
    if (activeTab) {
      const tabName = activeTab.getAttribute("data-tab");
      updateStatsBar(tabName);
    }

    // 2. Initialize domain toggle visibility
    const domainToggle = document.getElementById("domainToggleFixed");
    if (activeTab && domainToggle) {
      const tabName = activeTab.getAttribute("data-tab");
      if (tabName === "seismic") {
        domainToggle.classList.add("visible");
      }
    }

    // 3. Initialize image modal for zoomable images
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImage");
    const modalCaption = document.getElementById("modalCaption");
    const closeModal = document.querySelector(".modal-close");

    if (modal && modalImg) {
      // Setup zoomable images
      document.querySelectorAll(".zoomable-image").forEach(function (img) {
        img.style.cursor = "pointer";
        img.addEventListener("click", function () {
          openImageModal(this.src, this.alt);
        });
      });

      // Setup domain images
      document.querySelectorAll(".domain-image").forEach((img) => {
        img.style.cursor = "pointer";
        img.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();
          openImageModal(this.src, this.alt);
        });
      });

      // Close modal on background click
      modal.addEventListener("click", function (e) {
        if (e.target === modal) {
          closeImageModal();
        }
      });
    }

    // 4. Initialize domain-specific content visibility (for seismic tab)
    document.querySelectorAll(".domain-content").forEach((content) => {
      if (content.dataset.domain === "depth") {
        content.style.display = "block";
        content.style.opacity = "1";
      } else {
        content.style.display = "none";
        content.style.opacity = "0";
      }
    });

    // 5. Initialize back to top button
    const backToTop = document.getElementById('backToTop');
    if (backToTop) {
      backToTop.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    // 6. Initialize scroll progress
    updateScrollProgress();

    // 7. Setup lazy loading for images with IntersectionObserver
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            if (img.dataset.src) {
              img.src = img.dataset.src;
              img.removeAttribute('data-src');
            }
            observer.unobserve(img);
          }
        });
      }, {
        rootMargin: '50px 0px',
        threshold: 0.01
      });

      // Observe images with data-src attribute
      document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
      });
    }

    // 8. Wire seismogram 3D button (dual-domain viewer)
    try {
      const seismoBtn = document.querySelector('.seismogram-3d-button');
      const domainToggle = document.getElementById('domainToggleFixed');
      const domainToggleText = document.getElementById('domainToggleText');

      if (seismoBtn) {
        seismoBtn.addEventListener('click', function (e) {
          e.preventDefault();
          // Determine active domain from the toggle text (defaults to Time)
          const domain = domainToggleText ? domainToggleText.textContent.trim().toLowerCase() : 'time';
          const timeUrl = this.getAttribute('data-time-viewer');
          const depthUrl = this.getAttribute('data-depth-viewer');

          if (domain === 'depth' && depthUrl) {
            openViewer(depthUrl);
          } else if (timeUrl) {
            openViewer(timeUrl);
          } else if (depthUrl) {
            // fallback
            openViewer(depthUrl);
          }
        });
      }

      // 9. Make domain toggle interactive (switch Time <-> Depth)
      if (domainToggle && domainToggleText) {
        domainToggle.addEventListener('click', function () {
          const current = domainToggleText.textContent.trim();
          if (current.toLowerCase() === 'time') {
            domainToggleText.textContent = 'Depth';
            // change icon to depth indicator
            const icon = document.getElementById('domainToggleIcon');
            if (icon) icon.className = 'fas fa-layer-group';
            // show depth domain images
            document.querySelectorAll('.domain-content').forEach((content) => {
              if (content.dataset.domain === 'depth') {
                content.style.display = 'block';
                content.style.opacity = '1';
              } else {
                content.style.display = 'none';
                content.style.opacity = '0';
              }
            });
          } else {
            domainToggleText.textContent = 'Time';
            const icon = document.getElementById('domainToggleIcon');
            if (icon) icon.className = 'fas fa-clock';
            // show time domain images
            document.querySelectorAll('.domain-content').forEach((content) => {
              if (content.dataset.domain === 'time') {
                content.style.display = 'block';
                content.style.opacity = '1';
              } else {
                content.style.display = 'none';
                content.style.opacity = '0';
              }
            });
          }
        });
      }
    } catch (err) {
      console.warn('Seismogram button wiring failed:', err);
    }

    // Provide global switchDomain for inline onclick handlers used in HTML
    window.switchDomain = function(domain) {
      try {
        domain = (domain || '').toString().toLowerCase();
        const depthBtn = document.getElementById('depth-btn');
        const timeBtn = document.getElementById('time-btn');
        const domainDescription = document.getElementById('domain-description');

        // Update active classes and aria attributes
        if (domain === 'depth') {
          if (depthBtn) {
            depthBtn.classList.add('active');
            depthBtn.setAttribute('aria-selected','true');
            depthBtn.setAttribute('aria-pressed','true');
          }
          if (timeBtn) {
            timeBtn.classList.remove('active');
            timeBtn.setAttribute('aria-selected','false');
            timeBtn.setAttribute('aria-pressed','false');
          }

          // Show depth domain content
          document.querySelectorAll('.domain-content').forEach((content) => {
            if (content.dataset.domain === 'depth') {
              content.style.display = 'block';
              content.style.opacity = '1';
            } else {
              content.style.display = 'none';
              content.style.opacity = '0';
            }
          });

          if (domainDescription) {
            domainDescription.innerHTML = '<strong>Currently Viewing: Depth Domain</strong> — True depth-domain seismograms (200 layers, 0-199m) converted from time domain. Aligned with rock physics attributes for integrated interpretation.';
          }
        } else {
          if (timeBtn) {
            timeBtn.classList.add('active');
            timeBtn.setAttribute('aria-selected','true');
            timeBtn.setAttribute('aria-pressed','true');
          }
          if (depthBtn) {
            depthBtn.classList.remove('active');
            depthBtn.setAttribute('aria-selected','false');
            depthBtn.setAttribute('aria-pressed','false');
          }

          // Show time domain content
          document.querySelectorAll('.domain-content').forEach((content) => {
            if (content.dataset.domain === 'time') {
              content.style.display = 'block';
              content.style.opacity = '1';
            } else {
              content.style.display = 'none';
              content.style.opacity = '0';
            }
          });

          if (domainDescription) {
            domainDescription.innerHTML = '<strong>Currently Viewing: Time Domain</strong> — Standard two-way time (TWT) seismograms for conventional seismic workflows.';
          }
        }

        // Also update any fixed toggle text/button if present
        const domainToggleText = document.getElementById('domainToggleText');
        const domainToggleIcon = document.getElementById('domainToggleIcon');
        if (domainToggleText) domainToggleText.textContent = domain === 'depth' ? 'Depth' : 'Time';
        if (domainToggleIcon) domainToggleIcon.className = domain === 'depth' ? 'fas fa-layer-group' : 'fas fa-clock';
      } catch (e) {
        console.error('switchDomain error:', e);
      }
    };

    console.log("Page initialization complete!");
  } catch (error) {
    console.error("Error during page initialization:", error);
  }
});
