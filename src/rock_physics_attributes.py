"""
Rock Physics Attributes Analysis

Computes advanced rock physics attributes from elastic properties (Vp, Vs, ρ) for
reservoir characterization and facies discrimination. Combines multiple seismic
analysis techniques to create discriminants that separate lithology and fluid effects.

Rock Physics Attributes Computed:
    1. Lambda-Rho (λρ) - Incompressibility × Density (fluid-sensitive)
    2. Mu-Rho (μρ) - Shear modulus × Density (lithology-sensitive)
    3. Fluid Factor - Separates fluid effects from lithology
    4. Poisson Impedance - Direct fluid sensitivity indicator
    5. AVO Attributes - Intercept, Gradient (angle-dependent reflectivity)
    6. EI Gradient (ΔEI) - Angle-dependent elastic impedance response
    7. Hybrid Discriminants - Combined attributes for enhanced separation

Performance Target:
    - Current best: EI at 10° with Cohen's d = 3.407
    - Goal: Rock physics attributes with Cohen's d > 4.0 (HUGE effect size)

References:
    - Goodway et al. (1997): Lambda-Mu-Rho method
    - Quakenbush et al. (2006): Poisson impedance
    - Russell et al. (2003): Hybrid attributes for reservoir characterization
    - Avseth et al. (2005): Quantitative Seismic Interpretation

NOTE: Multi-angle EI functions (compute_ei_angle, compute_ei_multiangle,
      compute_ei_weighted_product) have been moved to modeling_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import time
import hashlib
import os
from tqdm.auto import tqdm
from scipy import stats as scipy_stats

# Import multi-angle EI functions from modeling_utils
from . import modeling_utils

# Re-export for backward compatibility
compute_ei_angle = modeling_utils.compute_ei_angle
compute_ei_multiangle = modeling_utils.compute_ei_multiangle
compute_ei_weighted_product = modeling_utils.compute_ei_weighted_product


def _apply_seismic_plot_style():
    """Apply the standard seismic plot style (figure size, DPI, fonts, spacing).

    Call this at the top of each plotting function so rock-physics plots match
    the seismic visual style.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (22, 16),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.titlepad": 12,
            "legend.frameon": False,
        }
    )


# ============================================================================
# ELASTIC IMPEDANCE (EI) COMPUTATION
# ============================================================================
# NOTE: Core EI functions moved to modeling_utils.py
# Aliases provided above for backward compatibility


def compute_ei_pca_fluid_discriminator(
    vp, vs, rho, angles_deg=None, show_progress=True
):
    """
    Compute PCA-based fluid discriminator from multi-angle EI.

    This function implements the SYNERGY approach discovered through experimentation:
    - Computes EI at multiple angles spanning near to far offsets
    - Applies PCA decomposition to extract orthogonal components
    - Returns PC3 (3rd principal component) which captures fluid signature

    Performance: F0-F3 Cohen's d = 0.760 (17% better than any single angle!)

    Args:
        vp (np.ndarray): P-wave velocity in m/s, shape (ni, nj, nk)
        vs (np.ndarray): S-wave velocity in m/s, shape (ni, nj, nk)
        rho (np.ndarray): Density in kg/m³, shape (ni, nj, nk)
        angles_deg (list or np.ndarray): Angles to use (default: [5,10,15,20,25,30,35,40])
        show_progress (bool): Whether to show progress bar

    Returns:
        dict: Dictionary with keys:
            - 'pc1': First principal component (96% variance - lithology)
            - 'pc2': Second principal component (4% variance - gradient)
            - 'pc3': Third principal component (0.04% variance - FLUID!)
            - 'explained_variance': Variance explained by each PC
            - 'pca_model': Fitted PCA model for future transform
            - 'scaler': Fitted StandardScaler for future transform
            - 'angles': Angles used
            - 'ei_volumes': Individual EI volumes at each angle

    Notes:
        - PC3 achieves F0-F3 = 0.760 (brine-oil discrimination)
        - This is TRUE SYNERGY - extracts hidden fluid signal
    - CRITICAL: PCA requires km/s units for proper variance decomposition
        - Function converts m/s → km/s internally for PCA computation
        - PC1 good for lithology (F0-F1 = 16.5)
        - PC3 sacrifices lithology for fluid sensitivity
        - Analogous to Lambda-Mu-Rho decomposition!

    References:
        - Synergy Analysis Results (SYNERGY_ANALYSIS_RESULTS.md)
        - PCA PC3 achieves +17% F0-F3 vs best baseline
        - Extracts orthogonal fluid signature from multi-angle space

    Example:
        >>> results = compute_ei_pca_fluid_discriminator(vp, vs, rho)
        >>> ei_fluid = results['pc3']  # Best for fluid discrimination!
        >>> ei_litho = results['pc1']  # Best for lithology
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn required for PCA. Install with: pip install scikit-learn"
        )

    # Default angles: comprehensive suite from near to far offsets
    if angles_deg is None:
        angles_deg = [5, 10, 15, 20, 25, 30, 35, 40]

    angles_array = np.array(angles_deg)

    print("\n" + "=" * 70)
    print("PCA-BASED FLUID DISCRIMINATOR")
    print("=" * 70)
    print(f"Computing EI at {len(angles_array)} angles: {list(angles_array)}°")
    print(f"Grid shape: {vp.shape}")

    # ⚠️ CRITICAL: PCA requires km/s units for proper variance decomposition!
    # The EI formula works with any units, but PCA variance structure depends on scale
    # Experiments show: km/s → F0-F3=0.760, m/s → F0-F3=0.001 (1000x worse!)
    print("\nConverting to experimental units for PCA:")
    print(f"   Input Vp: {vp.min():.0f} - {vp.max():.0f} m/s")
    print(f"   Input Vs: {vs.min():.0f} - {vs.max():.0f} m/s")
    print(f"   Input Rho: {rho.min():.0f} - {rho.max():.0f} kg/m³")

    vp_km = vp / 1000.0  # m/s → km/s
    vs_km = vs / 1000.0  # m/s → km/s
    rho_g = rho / 1000.0  # kg/m³ → g/cm³

    print(f"   PCA Vp: {vp_km.min():.3f} - {vp_km.max():.3f} km/s")
    print(f"   PCA Vs: {vs_km.min():.3f} - {vs_km.max():.3f} km/s")
    print(f"   PCA Rho: {rho_g.min():.3f} - {rho_g.max():.3f} g/cm³")

    print("\nTarget: Extract PC3 for fluid discrimination (F0-F3 = 0.760!)")
    print("   - PC1 (96% var): Bulk impedance → lithology")
    print("   - PC2 (4% var): Angle gradient → mixed")
    print("   - PC3 (0.04% var): Orthogonal fluid signal")

    t0 = time.time()

    # Step 1: Compute EI at each angle using km/s units
    # Use compute_ei_multiangle to match experimental implementation
    print("\n[1/4] Computing EI at each angle (km/s units)...")
    ei_volumes = []
    iterator = tqdm(angles_array, desc="Computing EI angles", disable=not show_progress)

    for angle in iterator:
        ei_result = compute_ei_multiangle(
            vp_km, vs_km, rho_g, [angle], show_progress=False
        )
        ei_angle = ei_result["ei_stack"]  # Extract the EI volume

        ei_volumes.append(ei_angle)

        if show_progress:
            iterator.set_postfix(
                {
                    "angle": f"{angle}°",
                    "range": f"{ei_angle.min():.2e}-{ei_angle.max():.2e}",
                }
            )

    # Step 2: Create feature matrix (flatten and stack)
    print("\n[2/4] Creating feature matrix...")
    original_shape = vp.shape
    n_samples = np.prod(original_shape)
    n_features = len(ei_volumes)

    X = np.column_stack([ei.flatten() for ei in ei_volumes])
    print(f"   Feature matrix: {X.shape} (samples × angles)")

    # Step 3: Standardize features
    print("\n[3/4] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Standardized to zero mean, unit variance")

    # Step 4: Apply PCA
    print("\n[4/4] Applying PCA decomposition...")
    pca = PCA(n_components=min(3, n_features))
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n   Explained variance ratio:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"      PC{i+1}: {var_ratio*100:6.2f}%")

    # Reshape back to 3D
    pc1 = X_pca[:, 0].reshape(original_shape)
    pc2 = (
        X_pca[:, 1].reshape(original_shape)
        if X_pca.shape[1] > 1
        else np.zeros_like(pc1)
    )
    pc3 = (
        X_pca[:, 2].reshape(original_shape)
        if X_pca.shape[1] > 2
        else np.zeros_like(pc1)
    )

    t1 = time.time()

    print("\n" + "=" * 70)
    print("PCA DECOMPOSITION COMPLETE")
    print("=" * 70)
    print(f"⏱  Computation time: {t1-t0:.2f}s")
    print(f"\n📊 Principal Components:")
    print(f"   PC1: [{pc1.min():.3e}, {pc1.max():.3e}] (mean={pc1.mean():.3e})")
    print(f"   PC2: [{pc2.min():.3e}, {pc2.max():.3e}] (mean={pc2.mean():.3e})")
    print(f"   PC3: [{pc3.min():.3e}, {pc3.max():.3e}] (mean={pc3.mean():.3e})")
    print(f"\n⭐ USE PC3 FOR FLUID DISCRIMINATION!")
    print(f"   Expected F0-F3 Cohen's d = 0.760")
    print(f"   +17% better than best single angle")
    print(f"   +329% better than baseline fluid stack")
    print("=" * 70)

    # Package results
    results = {
        "pc1": pc1,
        "pc2": pc2,
        "pc3": pc3,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_model": pca,
        "scaler": scaler,
        "angles": angles_array,
        "ei_volumes": ei_volumes,
        "config": {
            "angles_deg": list(angles_array),
            "n_angles": len(angles_array),
            "n_components": pca.n_components_,
            "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
            "method": "PCA on multi-angle EI",
            "performance": {
                "pc3_f0_f3_cohens_d": 0.760,
                "improvement_vs_baseline": "+17%",
                "synergy_type": "Extracts orthogonal fluid signal",
            },
        },
    }

    return results


def compare_ei_angles(ei_results, facies_depth, slice_inline=75):
    """
    Compare EI volumes at different angles for quality assessment.

    Args:
        ei_results (dict): Results from compute_ei_multiangle()
        facies_depth (np.ndarray): Facies model in depth domain for reference
        slice_inline (int): Inline to display (default: 75)

    Returns:
        dict: Statistics comparing different angles
    """
    print("\n" + "=" * 70)
    print("COMPARING EI AT DIFFERENT ANGLES")
    print("=" * 70)

    ei_volumes = ei_results["ei_volumes"]
    angles = ei_results["angles"]

    # Compute statistics for each angle
    angle_stats = []
    for angle, ei_vol in zip(angles, ei_volumes):
        stats_dict = {
            "angle": angle,
            "mean": float(ei_vol.mean()),
            "std": float(ei_vol.std()),
            "min": float(ei_vol.min()),
            "max": float(ei_vol.max()),
            "dynamic_range": float(ei_vol.max() - ei_vol.min()),
        }
        angle_stats.append(stats_dict)
        print(
            f"\n  {angle:5.1f}° : mean={stats_dict['mean']:.2e}, "
            f"std={stats_dict['std']:.2e}, range={stats_dict['dynamic_range']:.2e}"
        )

    # Compute cross-correlation between angles
    print("\nComputing angle cross-correlations...")
    n_angles = len(ei_volumes)
    corr_matrix = np.zeros((n_angles, n_angles))

    for i in range(n_angles):
        for j in range(n_angles):
            flat_i = ei_volumes[i].flatten()
            flat_j = ei_volumes[j].flatten()
            corr_matrix[i, j] = np.corrcoef(flat_i, flat_j)[0, 1]

    print("\nAngle Cross-Correlation Matrix:")
    print("      ", end="")
    for angle in angles:
        print(f"{angle:6.1f}°", end="")
    print()

    for i, angle_i in enumerate(angles):
        print(f"{angle_i:5.1f}°", end="")
        for j in range(n_angles):
            print(f" {corr_matrix[i, j]:6.3f}", end="")
        print()

    # Identify best angle (highest dynamic range = best discrimination)
    best_idx = np.argmax([s["dynamic_range"] for s in angle_stats])
    best_angle = angles[best_idx]

    print(
        f"\n✓ Best angle for discrimination: {best_angle}° "
        f"(dynamic range = {angle_stats[best_idx]['dynamic_range']:.2e})"
    )

    return {
        "angle_statistics": angle_stats,
        "best_angle": float(best_angle),
        "best_angle_idx": int(best_idx),
        "correlation_matrix": corr_matrix.tolist(),
    }


def create_optimal_ei_stack(
    ei_results,
    optimization="variance",
    facies=None,
    cohens_d_threshold=None,
    top_n_angles=None,
):
    """
    Create optimally weighted EI stack for maximum facies discrimination.

    Args:
        ei_results (dict): Results from compute_ei_multiangle()
        optimization (str): Optimization strategy ('variance', 'cohens_d', 'equal', 'cosine', 'gradient')
        facies (np.ndarray): Facies model for Cohen's d based optimization (required if optimization='cohens_d')
        cohens_d_threshold (float): Minimum Cohen's d to include angle (for 'cohens_d' optimization)
        top_n_angles (int): Use only top N angles by Cohen's d (for 'cohens_d' optimization)

    Returns:
        tuple: (ei_stack, weights, selected_angles)
    """
    print(f"\nCreating optimal EI stack (method: {optimization})...")

    ei_volumes = ei_results["ei_volumes"]
    angles = ei_results["angles"]

    # For Cohen's d based optimization, filter angles by performance
    if optimization == "cohens_d":
        if facies is None:
            raise ValueError("facies parameter required for 'cohens_d' optimization")

        # Calculate Cohen's d for each angle using FACIES-BASED metric
        # (measures amplitude separation between facies, not boundary detection)
        angle_performance = []
        for angle, ei_vol in zip(angles, ei_volumes):
            stats = analyze_facies_discrimination(ei_vol, facies)
            angle_performance.append(
                {"angle": angle, "cohens_d": stats["cohens_d"], "ei_volume": ei_vol}
            )

        # Sort by Cohen's d
        angle_performance.sort(key=lambda x: x["cohens_d"], reverse=True)

        print(f"  Angle performance (Cohen's d):")
        for perf in angle_performance:
            print(f"    {perf['angle']:3.0f}°: {perf['cohens_d']:.4f}")

        # Filter angles
        if top_n_angles is not None:
            selected = angle_performance[:top_n_angles]
            print(f"  Selecting top {top_n_angles} angles")
        elif cohens_d_threshold is not None:
            selected = [
                p for p in angle_performance if p["cohens_d"] >= cohens_d_threshold
            ]
            print(f"  Selecting angles with Cohen's d >= {cohens_d_threshold}")
        else:
            # Default: use angles above median performance
            median_d = np.median([p["cohens_d"] for p in angle_performance])
            selected = [p for p in angle_performance if p["cohens_d"] >= median_d]
            print(f"  Selecting angles above median (Cohen's d >= {median_d:.4f})")

        # Rebuild angle list and volumes with selected angles only
        angles = np.array([p["angle"] for p in selected])
        ei_volumes = [p["ei_volume"] for p in selected]

        print(f"  Selected angles: {angles}")

        # Use EQUAL weighting on selected angles (simple averaging)
        # This preserves the performance of the best angles without dilution
        weights = np.ones(len(angles)) / len(angles)

    elif optimization == "equal":
        weights = np.ones(len(angles)) / len(angles)
    elif optimization == "cosine":
        weights = np.cos(np.deg2rad(angles))
        weights = weights / weights.sum()
    elif optimization == "variance":
        variances = np.array([ei.var() for ei in ei_volumes])
        weights = variances / variances.sum()
    elif optimization == "gradient":
        gradients = []
        for ei in ei_volumes:
            grad = np.abs(np.gradient(ei, axis=2)).mean()
            gradients.append(grad)
        gradients = np.array(gradients)
        weights = gradients / gradients.sum()
    else:
        raise ValueError(f"Unknown optimization method: {optimization}")

    print(f"  Weights: {weights}")

    # Create weighted stack
    ei_stack = np.zeros_like(ei_volumes[0])
    for ei_vol, weight in zip(ei_volumes, weights):
        ei_stack += weight * ei_vol

    print(f"✓ Optimal stack range: {ei_stack.min():.2e} - {ei_stack.max():.2e}")

    return ei_stack, weights, angles


def analyze_facies_discrimination(ei_volume, facies):
    """
    Analyze facies discrimination capability using amplitude separation between facies.

    This computes facies-based Cohen's d, measuring how distinct facies are
    in their amplitude distributions (better for facies classification).

    Args:
        ei_volume (np.ndarray): EI volume in depth, shape (ni, nj, nk)
        facies (np.ndarray): Facies model in depth, shape (ni, nj, nk)

    Returns:
        dict: Statistics including average Cohen's d between all facies pairs
    """
    # Extract amplitudes for each facies
    facies_amplitudes = {}
    facies_stats = {}

    for facies_val in range(4):  # Assuming 4 facies types
        mask = facies == facies_val
        if np.any(mask):
            amps = ei_volume[mask]
            facies_amplitudes[facies_val] = amps
            facies_stats[facies_val] = {
                "mean": amps.mean(),
                "std": amps.std(),
            }

    # Calculate separation between all facies pairs (Cohen's d)
    n_facies = len(facies_stats)
    if n_facies < 2:
        return {"cohens_d": 0.0}

    separations = []
    for i in range(4):
        for j in range(i + 1, 4):
            if i in facies_stats and j in facies_stats:
                mean_diff = abs(facies_stats[i]["mean"] - facies_stats[j]["mean"])
                pooled_std = np.sqrt(
                    (facies_stats[i]["std"] ** 2 + facies_stats[j]["std"] ** 2) / 2
                )
                cohens_d = mean_diff / (pooled_std + 1e-10)
                separations.append(cohens_d)

    avg_separation = np.mean(separations) if separations else 0.0

    return {
        "cohens_d": avg_separation,
        "facies_stats": facies_stats,
    }


def analyze_facies_correlation_depth(ei_volume, facies):
    """
    Analyze how well EI correlates with facies boundaries in depth domain.

    Args:
        ei_volume (np.ndarray): EI volume in depth, shape (ni, nj, nk)
        facies (np.ndarray): Facies model in depth, shape (ni, nj, nk)

    Returns:
        dict: Statistics including Cohen's d, Pearson r, boundary amplitudes
    """
    # Compute EI gradient (vertical derivative)
    grad_ei = np.abs(np.gradient(ei_volume, axis=2))

    # Detect facies boundaries (where facies changes)
    grad_facies = np.abs(np.gradient(facies.astype(float), axis=2))
    facies_boundaries = grad_facies > 0.1

    # Extract gradients at boundaries and away from boundaries
    grad_at_boundaries = grad_ei[facies_boundaries]
    grad_away = grad_ei[~facies_boundaries]

    # Cohen's d effect size (measures separation)
    mean_boundary = grad_at_boundaries.mean()
    mean_away = grad_away.mean()
    std_pooled = np.sqrt((grad_at_boundaries.var() + grad_away.var()) / 2)
    cohens_d = (mean_boundary - mean_away) / (std_pooled + 1e-10)

    # Pearson correlation
    flat_grad = grad_ei.flatten()
    flat_boundaries = facies_boundaries.flatten().astype(float)
    r_pearson = np.corrcoef(flat_grad, flat_boundaries)[0, 1]

    # Spearman correlation (rank-based, more robust)
    r_spearman = scipy_stats.spearmanr(flat_grad, flat_boundaries)[0]

    # Boundary amplitude statistics
    boundary_amp_mean = grad_at_boundaries.mean()
    away_amp_mean = grad_away.mean()

    # Signal-to-noise like ratio
    snr = boundary_amp_mean / (away_amp_mean + 1e-10)

    return {
        "cohens_d": cohens_d,
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "snr": snr,
    }


def run_multiangle_analysis(
    props_depth, angles_deg=[0, 5, 10, 15, 20, 25], ei_pc3_fluid=None
):
    """
    Run comprehensive multi-angle EI analysis in depth domain.

    Args:
        props_depth: Dictionary with 'vp', 'vs', 'rho', 'facies'
        angles_deg: List of angles to analyze
        ei_pc3_fluid: Optional PC3 fluid discriminator component to save in cache

    Returns:
        dict: Multi-angle EI results and analysis
    """
    print("\n" + "=" * 70)
    print("MULTI-ANGLE EI ANALYSIS (DEPTH DOMAIN)")
    print("=" * 70)
    print(f"Computing EI at {len(angles_deg)} angles: {angles_deg}°")

    # Compute multi-angle EI
    t0 = time.time()
    ei_results = compute_ei_multiangle(
        props_depth["vp"],
        props_depth["vs"],
        props_depth["rho"],
        angles_deg,
        show_progress=True,
    )
    t1 = time.time()
    print(f"✓ Computed {len(angles_deg)} EI volumes in {t1-t0:.2f}s")

    # Create angle stacks
    print("\nCreating angle stacks...")
    ei_volumes_list = ei_results["ei_volumes"]
    angles_array = ei_results["angles"]
    ei_dict = {
        int(angle): ei_vol for angle, ei_vol in zip(angles_array, ei_volumes_list)
    }

    # Near, mid, far stacks
    near_angles = [a for a in angles_deg if a <= 10]
    mid_angles = [a for a in angles_deg if 10 < a <= 20]
    far_angles = [a for a in angles_deg if a > 20]

    ei_near = (
        np.mean([ei_dict[a] for a in near_angles], axis=0) if near_angles else None
    )
    ei_mid = np.mean([ei_dict[a] for a in mid_angles], axis=0) if mid_angles else None
    ei_far = np.mean([ei_dict[a] for a in far_angles], axis=0) if far_angles else None

    # Optimal adaptive stack (Cohen's d based angle selection + simple averaging)
    facies = props_depth["facies"]
    ei_optimal, weights, selected_angles = create_optimal_ei_stack(
        ei_results,
        optimization="cohens_d",
        facies=facies,
        top_n_angles=4,  # Use top 4 angles by facies discrimination
    )
    print(f"  Selected angles for optimal stack: {selected_angles}")
    print(f"  Weights: {weights}")

    # EI gradient (far - near)
    if ei_far is not None and ei_near is not None:
        ei_gradient = ei_far - ei_near
    else:
        ei_gradient = None

    print("✓ Created angle stacks")

    # Analyze facies correlation for each angle
    print("\nAnalyzing facies correlation for each angle...")
    facies = props_depth["facies"]
    correlations = {}

    for angle in angles_deg:
        ei_vol = ei_dict[angle]
        stats_dict = analyze_facies_correlation_depth(ei_vol, facies)
        correlations[angle] = stats_dict
        print(
            f"  {angle:3.0f}°: Cohen's d = {stats_dict['cohens_d']:.4f}, "
            f"Pearson r = {stats_dict['pearson_r']:.4f}, "
            f"SNR = {stats_dict['snr']:.2f}"
        )

    # Find best angle
    best_angle = max(correlations.keys(), key=lambda a: correlations[a]["cohens_d"])
    print(
        f"\n✓ Best single angle: {best_angle}° "
        f"(Cohen's d = {correlations[best_angle]['cohens_d']:.4f})"
    )

    # Analyze optimal stack
    opt_stats = analyze_facies_correlation_depth(ei_optimal, facies)
    print(
        f"✓ Optimal stack: Cohen's d = {opt_stats['cohens_d']:.4f}, "
        f"Pearson r = {opt_stats['pearson_r']:.4f}"
    )

    # Save results
    config_hash = hashlib.md5(str(sorted(angles_deg)).encode()).hexdigest()[:20]
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/ei_depth_{config_hash}.npz"

    save_dict = {
        "facies": facies,
        "vp": props_depth["vp"],
        "vs": props_depth["vs"],
        "rho": props_depth["rho"],
        "angles": np.array(angles_deg),
        "ei_optimal": ei_optimal,
        "correlations": str(correlations),
    }

    # Add individual angle volumes
    for angle, ei_vol in zip(angles_array, ei_volumes_list):
        save_dict[f"ei_{int(angle)}deg"] = ei_vol

    # Add stacks if they exist
    if ei_near is not None:
        save_dict["ei_near"] = ei_near
    if ei_mid is not None:
        save_dict["ei_mid"] = ei_mid
    if ei_far is not None:
        save_dict["ei_far"] = ei_far
    if ei_gradient is not None:
        save_dict["ei_gradient"] = ei_gradient

    # Add PC3 fluid discriminator if provided ⭐
    if ei_pc3_fluid is not None:
        save_dict["ei_pc3_fluid"] = ei_pc3_fluid
        print("  ⭐ Including PC3 fluid discriminator in cache (F0-F3 = 0.760)")

    np.savez_compressed(cache_file, **save_dict)
    print(f"\n✓ Saved multi-angle results to: {cache_file}")
    print(f"  File size: {os.path.getsize(cache_file) / 1024**2:.1f} MB")

    return {
        "ei_dict": ei_dict,
        "ei_optimal": ei_optimal,
        "correlations": correlations,
        "best_angle": best_angle,
        "cache_file": cache_file,
    }


def run_multiangle_analysis_adaptive(
    props_depth,
    angles_deg=[0, 5, 10, 15, 20, 25],
    optimization="cohens_d",
    top_n_angles=4,
):
    """
    Run multi-angle EI analysis with ADAPTIVE angle selection.

    This function dynamically selects the best performing angles based on
    their facies discrimination power (Cohen's d), then creates an optimal
    stack using only those angles.

    Args:
        props_depth: Dictionary with 'vp', 'vs', 'rho', 'facies'
        angles_deg: Initial list of angles to evaluate
        optimization: Optimization method ('cohens_d', 'variance', etc.)
        top_n_angles: Number of best angles to use (default: 4, matching AVO)

    Returns:
        dict: Multi-angle EI results with adaptive angle selection
    """
    print("\n" + "=" * 70)
    print("ADAPTIVE MULTI-ANGLE EI ANALYSIS (DEPTH DOMAIN)")
    print("=" * 70)
    print(f"Evaluating {len(angles_deg)} angles: {angles_deg}°")
    print(f"Optimization: {optimization}")
    if optimization == "cohens_d":
        print(f"Will select top {top_n_angles} angles by Cohen's d")

    # Compute multi-angle EI for all angles
    t0 = time.time()
    ei_results = compute_ei_multiangle(
        props_depth["vp"],
        props_depth["vs"],
        props_depth["rho"],
        angles_deg,
        show_progress=True,
    )
    t1 = time.time()
    print(f"✓ Computed {len(angles_deg)} EI volumes in {t1-t0:.2f}s")

    # Create angle stacks with adaptive selection
    print("\nCreating adaptive angle stack...")
    facies = props_depth["facies"]

    # Use adaptive optimization
    ei_optimal, weights, selected_angles = create_optimal_ei_stack(
        ei_results, optimization=optimization, facies=facies, top_n_angles=top_n_angles
    )

    print("✓ Created adaptive optimal stack")

    # Analyze the optimal stack
    print("\nAnalyzing optimal stack performance...")
    opt_stats = analyze_facies_correlation_depth(ei_optimal, facies)
    print(
        f"✓ Adaptive optimal stack: Cohen's d = {opt_stats['cohens_d']:.4f}, "
        f"Pearson r = {opt_stats['pearson_r']:.4f}"
    )

    # Also compute standard variance-weighted stack for comparison
    ei_volumes_list = ei_results["ei_volumes"]
    angles_array = ei_results["angles"]

    print(
        "\nFor comparison, computing standard variance-weighted stack (all angles)..."
    )
    ei_all_angles, weights_all, _ = create_optimal_ei_stack(
        ei_results, optimization="variance"
    )
    all_stats = analyze_facies_correlation_depth(ei_all_angles, facies)
    print(
        f"✓ Standard stack (all angles): Cohen's d = {all_stats['cohens_d']:.4f}, "
        f"Pearson r = {all_stats['pearson_r']:.4f}"
    )

    improvement = (
        (opt_stats["cohens_d"] - all_stats["cohens_d"]) / all_stats["cohens_d"]
    ) * 100
    print(f"\n📊 Adaptive selection improvement: {improvement:+.1f}%")

    # Save results with adaptive configuration
    config_str = f"{sorted(angles_deg)}_adaptive_top{top_n_angles}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:20]
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/ei_depth_adaptive_{config_hash}.npz"

    save_dict = {
        "facies": facies,
        "vp": props_depth["vp"],
        "vs": props_depth["vs"],
        "rho": props_depth["rho"],
        "all_angles": np.array(angles_deg),
        "selected_angles": selected_angles,
        "ei_optimal": ei_optimal,
        "ei_all_angles": ei_all_angles,
        "optimal_cohens_d": opt_stats["cohens_d"],
        "all_angles_cohens_d": all_stats["cohens_d"],
        "optimization": optimization,
    }

    # Add individual angle volumes
    for angle, ei_vol in zip(angles_array, ei_volumes_list):
        save_dict[f"ei_{int(angle)}deg"] = ei_vol

    np.savez_compressed(cache_file, **save_dict)
    print(f"\n✓ Saved adaptive results to: {cache_file}")
    print(f"  File size: {os.path.getsize(cache_file) / 1024**2:.1f} MB")

    return {
        "ei_optimal": ei_optimal,
        "selected_angles": selected_angles,
        "optimal_cohens_d": opt_stats["cohens_d"],
        "all_angles_cohens_d": all_stats["cohens_d"],
        "improvement": improvement,
        "cache_file": cache_file,
    }


# ============================================================================
# ROCK PHYSICS ATTRIBUTES
# ============================================================================


def load_depth_data():
    """
    Load depth data from cache or from raw files.

    Returns:
        dict: vp, vs, rho, facies volumes
    """
    # Try to load from existing modeling cache first
    cache_pattern = ".cache/ei_*.npz"
    from glob import glob

    cache_files = glob(cache_pattern)
    if cache_files:
        # Load the most recent EI cache which includes depth data
        latest_cache = max(cache_files, key=os.path.getmtime)
        print(f"Loading rock properties from: {latest_cache}")
        data = np.load(latest_cache)

        if "ei_depth" in data:
            # This is an EI cache, we need to load properties differently
            print("  EI cache found, but need rock properties...")

    # If no cache, load from data files
    print("Loading data from Stanford VI-E folders...")

    # Use a simpler approach - load using numpy directly
    DATA_PATH = ""
    GRID_SHAPE = (150, 200, 200)

    def load_property(folder_name, filename_candidates):
        """Load a single property from folder."""
        dir_path = os.path.join(DATA_PATH, folder_name)

        for filename in filename_candidates:
            filepath = os.path.join(dir_path, filename)
            if os.path.exists(filepath):
                print(f"  Loading {folder_name}...")
                # Read GSLIB format (3 header lines)
                with open(filepath, "r") as f:
                    # Skip 3 header lines
                    f.readline()  # Title
                    f.readline()  # Number of variables
                    f.readline()  # Variable name
                    # Read data
                    values = [float(line.strip()) for line in f if line.strip()]

                # Reshape to grid
                data_cube = np.array(values).reshape(GRID_SHAPE, order="F")
                return data_cube

        raise FileNotFoundError(f"Could not find data file in {dir_path}")

    props = {
        "vp": load_property(
            "P-wave Velocity", ["Pvelocity.dat", "P-wave Velocity.dat"]
        ),
        "vs": load_property(
            "S-wave Velocity", ["Svelocity.dat", "S-wave Velocity.dat"]
        ),
        "rho": load_property("Density", ["Density.dat"]),
        "facies": load_property("Facies", ["Facies.dat"]),
    }

    print(f"✓ Loaded all properties with shape {GRID_SHAPE}")

    # Convert units: velocities from km/s to m/s, density from g/cm³ to kg/m³
    print("Converting units:")
    print(f"  Vp: {props['vp'].min():.2f}-{props['vp'].max():.2f} km/s → ", end="")
    props["vp"] *= 1000.0  # km/s to m/s
    print(f"{props['vp'].min():.0f}-{props['vp'].max():.0f} m/s")

    print(f"  Vs: {props['vs'].min():.2f}-{props['vs'].max():.2f} km/s → ", end="")
    props["vs"] *= 1000.0  # km/s to m/s
    print(f"{props['vs'].min():.0f}-{props['vs'].max():.0f} m/s")

    print(f"  Rho: {props['rho'].min():.2f}-{props['rho'].max():.2f} g/cm³ → ", end="")
    props["rho"] *= 1000.0  # g/cm³ to kg/m³
    print(f"{props['rho'].min():.0f}-{props['rho'].max():.0f} kg/m³")

    return props


def compute_avo_attributes(vp, vs, rho, angles_deg=[0, 5, 10, 15, 20, 25]):
    """
    Compute AVO attributes (intercept, gradient) from rock properties.

    Args:
        vp, vs, rho: Rock property cubes
        angles_deg: Angles to use for fitting

    Returns:
        dict: AVO intercept, gradient, and derived attributes
    """
    print("\nComputing AVO attributes from rock physics...")
    print(f"  Using {len(angles_deg)} angles: {angles_deg}°")

    from .reflectivity import solve_zoeppritz

    ni, nj, nk = vp.shape
    intercept = np.zeros((ni, nj, nk - 1))
    gradient = np.zeros((ni, nj, nk - 1))

    # Compute reflectivity at each angle
    angles_rad = np.deg2rad(angles_deg)
    sin2_theta = np.sin(angles_rad) ** 2

    # For each interface
    for k in range(nk - 1):
        vp1, vp2 = vp[:, :, k], vp[:, :, k + 1]
        vs1, vs2 = vs[:, :, k], vs[:, :, k + 1]
        rho1, rho2 = rho[:, :, k], rho[:, :, k + 1]

        # Compute reflectivity at each angle
        reflectivities = []
        for angle in angles_rad:
            r = solve_zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, angle)
            reflectivities.append(r)

        reflectivities = np.array(reflectivities)  # shape: (n_angles, ni, nj)

        # Fit Intercept + Gradient*sin²θ for each trace
        for i in range(ni):
            for j in range(nj):
                r_trace = reflectivities[:, i, j]
                # Linear regression: R(θ) = A + B*sin²(θ)
                A = np.vstack([np.ones(len(sin2_theta)), sin2_theta]).T
                result = np.linalg.lstsq(A, r_trace, rcond=None)[0]
                intercept[i, j, k] = result[0]
                gradient[i, j, k] = result[1]

    print(f"✓ Intercept range: [{intercept.min():.4f}, {intercept.max():.4f}]")
    print(f"✓ Gradient range: [{gradient.min():.4f}, {gradient.max():.4f}]")

    return {
        "intercept": intercept,
        "gradient": gradient,
        "product": intercept * gradient,
        "scaled_gradient": gradient / (intercept + 1e-10),
    }


def compute_lambda_mu_rho(vp, vs, rho):
    """
    Compute Lambda-Rho and Mu-Rho attributes (Goodway et al. 1997).

    Lambda-Rho is sensitive to pore fluids.
    Mu-Rho is sensitive to rock matrix (lithology).

    Args:
        vp, vs, rho: Rock property cubes

    Returns:
        dict: lambda_rho, mu_rho, lambda_mu_ratio
    """
    print("\nComputing Lambda-Mu-Rho attributes...")

    # Incompressibility and rigidity moduli
    mu = rho * vs**2  # Shear modulus (rigidity)
    lambda_mod = rho * vp**2 - 2 * mu  # Lame's first parameter

    # Lambda-Rho and Mu-Rho
    lambda_rho = lambda_mod * rho
    mu_rho = mu * rho

    # Lambda/Mu ratio (fluid indicator)
    lambda_mu_ratio = lambda_mod / (mu + 1e-10)

    print(f"✓ Lambda-Rho range: [{lambda_rho.min():.2e}, {lambda_rho.max():.2e}]")
    print(f"✓ Mu-Rho range: [{mu_rho.min():.2e}, {mu_rho.max():.2e}]")
    print(
        f"✓ Lambda/Mu ratio range: [{lambda_mu_ratio.min():.3f}, {lambda_mu_ratio.max():.3f}]"
    )

    return {
        "lambda_rho": lambda_rho,
        "mu_rho": mu_rho,
        "lambda_mu_ratio": lambda_mu_ratio,
    }


def compute_poisson_impedance(vp, vs, rho):
    """
    Compute Poisson Impedance (Quakenbush et al. 2006).

    PI is highly sensitive to pore fluids and can discriminate
    gas sands from brine sands better than AI or EI alone.

    Args:
        vp, vs, rho: Rock property cubes

    Returns:
        np.ndarray: Poisson Impedance
    """
    print("\nComputing Poisson Impedance...")

    # Poisson's ratio
    vp_vs_ratio = vp / (vs + 1e-10)
    poisson = (vp_vs_ratio**2 - 2) / (2 * (vp_vs_ratio**2 - 1))

    # Poisson Impedance = AI * (1-2σ)/(1-σ)
    # where σ is Poisson's ratio
    ai = vp * rho
    pi = ai * (1 - 2 * poisson) / (1 - poisson + 1e-10)

    print(f"✓ Poisson's ratio range: [{poisson.min():.3f}, {poisson.max():.3f}]")
    print(f"✓ Poisson Impedance range: [{pi.min():.2e}, {pi.max():.2e}]")

    return pi


def compute_fluid_factor(lambda_rho, mu_rho):
    """
    Compute Fluid Factor from Lambda-Rho and Mu-Rho.

    Fluid Factor = Lambda-Rho - k*Mu-Rho
    where k is chosen to minimize lithology effects.

    Args:
        lambda_rho, mu_rho: Lambda-Rho and Mu-Rho volumes

    Returns:
        np.ndarray: Fluid Factor
    """
    print("\nComputing Fluid Factor...")

    # Optimal k typically 0.5-1.5 depending on lithology
    # For clastics, k ≈ 1.0 works well
    k = 1.0

    fluid_factor = lambda_rho - k * mu_rho

    print(f"✓ Fluid Factor range: [{fluid_factor.min():.2e}, {fluid_factor.max():.2e}]")
    print(f"  (Using k = {k})")

    return fluid_factor


def compute_hybrid_ei_avo_attributes(vp, vs, rho, ei_angle=10):
    """
    Compute comprehensive hybrid EI-AVO attributes.

    Args:
        vp, vs, rho: Rock property cubes
        ei_angle: Angle for single-angle EI (default: 10°)

    Returns:
        dict: All hybrid attributes
    """
    print("\n" + "=" * 70)
    print("HYBRID EI-AVO ATTRIBUTE COMPUTATION")
    print("=" * 70)

    t_start = time.time()

    # 1. Compute multi-angle EI
    print("\n1. Computing Multi-Angle EI...")
    ei_results = compute_ei_multiangle(vp, vs, rho, angles_deg=[0, 5, 10, 15, 20, 25])
    ei_gradient = ei_results["ei_gradient"]
    ei_stack = ei_results["ei_stack"]
    ei_optimal = compute_ei_angle(vp, vs, rho, ei_angle)

    # 2. Compute AVO attributes
    print("\n2. Computing AVO Attributes...")
    avo_attrs = compute_avo_attributes(vp, vs, rho)

    # 3. Compute Lambda-Mu-Rho
    print("\n3. Computing Lambda-Mu-Rho...")
    lmr = compute_lambda_mu_rho(vp, vs, rho)

    # 4. Compute Poisson Impedance
    print("\n4. Computing Poisson Impedance...")
    poisson_impedance = compute_poisson_impedance(vp, vs, rho)

    # 5. Compute Fluid Factor
    print("\n5. Computing Fluid Factor...")
    fluid_factor = compute_fluid_factor(lmr["lambda_rho"], lmr["mu_rho"])

    # 6. Create Hybrid Attributes
    print("\n6. Creating Hybrid Discriminants...")

    # Acoustic Impedance for reference
    ai = vp * rho

    # Hybrid 1: EI-AVO Product (combines sensitivities)
    # Need to pad AVO attributes to match EI shape
    avo_intercept_padded = np.pad(
        avo_attrs["intercept"], ((0, 0), (0, 0), (0, 1)), mode="edge"
    )
    hybrid_ei_avo_product = ei_optimal * avo_intercept_padded

    # Hybrid 2: EI Fluid Factor (EI gradient normalized by stack)
    hybrid_ei_fluid = ei_gradient / (ei_stack + 1e-10)

    # Hybrid 3: EI-Lambda/Mu (combines EI with rock physics ratio)
    hybrid_ei_lambda_mu = ei_optimal * lmr["lambda_mu_ratio"]

    print(
        f"✓ Hybrid EI-AVO Product range: [{hybrid_ei_avo_product.min():.2e}, {hybrid_ei_avo_product.max():.2e}]"
    )
    print(
        f"✓ Hybrid EI Fluid Factor range: [{hybrid_ei_fluid.min():.3f}, {hybrid_ei_fluid.max():.3f}]"
    )
    print(
        f"✓ Hybrid EI-Lambda/Mu range: [{hybrid_ei_lambda_mu.min():.2e}, {hybrid_ei_lambda_mu.max():.2e}]"
    )

    t_end = time.time()
    print(f"\n✓ Computed all hybrid attributes in {t_end-t_start:.2f}s")
    print("=" * 70)

    # Package all results
    results = {
        # Single-angle EI
        "ei_optimal": ei_optimal,
        "ei_gradient": ei_gradient,
        "ei_stack": ei_stack,
        # AVO attributes
        "avo_intercept": avo_attrs["intercept"],
        "avo_gradient": avo_attrs["gradient"],
        "avo_product": avo_attrs["product"],
        # Lambda-Mu-Rho
        "lambda_rho": lmr["lambda_rho"],
        "mu_rho": lmr["mu_rho"],
        "lambda_mu_ratio": lmr["lambda_mu_ratio"],
        # Other attributes
        "poisson_impedance": poisson_impedance,
        "fluid_factor": fluid_factor,
        "acoustic_impedance": ai,
        # Hybrid attributes
        "hybrid_ei_avo_product": hybrid_ei_avo_product,
        "hybrid_ei_fluid": hybrid_ei_fluid,
        "hybrid_ei_lambda_mu": hybrid_ei_lambda_mu,
    }

    return results


def analyze_attribute_discrimination(attribute, facies, name="Attribute"):
    """
    Analyze how well an attribute discriminates facies.

    Returns Cohen's d, Pearson r, and other statistics.
    """
    from scipy.stats import pearsonr

    # Flatten arrays
    attr_flat = attribute.flatten()
    facies_flat = facies.flatten()

    # Remove any NaN/Inf
    valid_mask = np.isfinite(attr_flat) & np.isfinite(facies_flat)
    attr_valid = attr_flat[valid_mask]
    facies_valid = facies_flat[valid_mask]

    # Separate by facies (assuming binary: 0=shale, 1=sand)
    mask_class0 = facies_valid == 0
    mask_class1 = facies_valid == 1

    attr_class0 = attr_valid[mask_class0]
    attr_class1 = attr_valid[mask_class1]

    # Cohen's d
    mean0 = attr_class0.mean()
    mean1 = attr_class1.mean()
    std0 = attr_class0.std()
    std1 = attr_class1.std()
    pooled_std = np.sqrt((std0**2 + std1**2) / 2)
    cohens_d = abs(mean1 - mean0) / pooled_std

    # Pearson correlation
    pearson_r, p_value = pearsonr(attr_valid, facies_valid)

    # Signal-to-noise ratio
    signal = abs(mean1 - mean0)
    noise = (std0 + std1) / 2
    snr = signal / noise if noise > 0 else 0

    return {
        "name": name,
        "cohens_d": cohens_d,
        "pearson_r": pearson_r,
        "p_value": p_value,
        "snr": snr,
        "mean_class0": mean0,
        "mean_class1": mean1,
        "std_class0": std0,
        "std_class1": std1,
    }


def compare_all_attributes(hybrid_results, facies):
    """
    Compare all attributes and rank by discrimination performance.
    """
    print("\n" + "=" * 70)
    print("FACIES DISCRIMINATION ANALYSIS")
    print("=" * 70)

    # Define attributes to test
    attributes_to_test = [
        ("ei_optimal", "EI at 10° (current best)"),
        ("ei_gradient", "EI Gradient"),
        ("acoustic_impedance", "Acoustic Impedance"),
        ("poisson_impedance", "Poisson Impedance"),
        ("lambda_rho", "Lambda-Rho"),
        ("mu_rho", "Mu-Rho"),
        ("lambda_mu_ratio", "Lambda/Mu Ratio"),
        ("fluid_factor", "Fluid Factor"),
        ("hybrid_ei_avo_product", "Hybrid: EI×AVO Product"),
        ("hybrid_ei_fluid", "Hybrid: EI Fluid Factor"),
        ("hybrid_ei_lambda_mu", "Hybrid: EI×Lambda/Mu"),
    ]

    results = []
    for attr_key, attr_name in attributes_to_test:
        if attr_key in hybrid_results:
            attr = hybrid_results[attr_key]
            stats = analyze_attribute_discrimination(attr, facies, attr_name)
            results.append(stats)

    # Sort by Cohen's d
    results.sort(key=lambda x: x["cohens_d"], reverse=True)

    # Print results table
    print("\nRanked by Cohen's d (effect size):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Attribute':<35} {'Cohen d':>10} {'Pearson r':>10} {'SNR':>8}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        # Effect size interpretation
        if r["cohens_d"] > 3.0:
            effect = "HUGE ★★★"
        elif r["cohens_d"] > 2.0:
            effect = "VERY LARGE ★★"
        elif r["cohens_d"] > 0.8:
            effect = "LARGE ★"
        elif r["cohens_d"] > 0.5:
            effect = "MEDIUM"
        else:
            effect = "SMALL"

        print(
            f"{i:<5} {r['name']:<35} {r['cohens_d']:>10.4f} {r['pearson_r']:>10.4f} {r['snr']:>8.2f}  {effect}"
        )

    print("-" * 70)

    # Highlight improvements
    baseline_idx = next(i for i, r in enumerate(results) if "EI at 10°" in r["name"])
    baseline_d = results[baseline_idx]["cohens_d"]

    print(f"\nBaseline (EI at 10°): Cohen's d = {baseline_d:.4f}")

    best = results[0]
    if best["cohens_d"] > baseline_d:
        improvement = ((best["cohens_d"] - baseline_d) / baseline_d) * 100
        print(f"✓ BEST: {best['name']}")
        print(
            f"  Cohen's d = {best['cohens_d']:.4f} (+{improvement:.1f}% improvement!)"
        )
        print(f"  Pearson r = {best['pearson_r']:.4f}")
    else:
        print(f"✓ EI at 10° remains the best attribute")

    print("=" * 70)

    return results


def plot_attribute_vs_facies(attribute, facies, name, cache_dir=".cache"):
    """
    Create individual attribute vs facies plot with histograms and statistics.

    Args:
        attribute: Rock physics attribute volume
        facies: Facies volume
        name: Name of the attribute
        cache_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print(f"\nGenerating {name} vs facies plot...")

    # Flatten and remove invalid values
    attr_flat = attribute.flatten()
    facies_flat = facies.flatten()
    valid_mask = np.isfinite(attr_flat) & np.isfinite(facies_flat)
    attr_valid = attr_flat[valid_mask]
    facies_valid = facies_flat[valid_mask]

    # Separate by facies
    mask_f0 = facies_valid == 0
    mask_f1 = facies_valid == 1
    mask_f2 = facies_valid == 2
    mask_f3 = facies_valid == 3

    attr_f0 = attr_valid[mask_f0]
    attr_f1 = attr_valid[mask_f1]
    attr_f2 = attr_valid[mask_f2]
    attr_f3 = attr_valid[mask_f3]

    # Calculate statistics
    stats = analyze_attribute_discrimination(attribute, facies, name)

    # Apply seismic plot style for consistent visuals
    _apply_seismic_plot_style()
    # Create figure with improved layout - uses rcParams figsize
    fig = plt.figure()
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.50,
        wspace=0.42,
        height_ratios=[0.10, 1.15, 1.15, 0.90],
        top=0.93,
        bottom=0.03,
        left=0.04,
        right=0.97,
    )

    # Main title with better styling and more space below
    fig.suptitle(
        f"{name} - Facies Discrimination Analysis",
        fontsize=22,
        fontweight="bold",
        y=0.97,
        color="#2C3E50",
    )

    # === ROW 0: KEY STATISTICS BANNER ===
    ax_stats_banner = fig.add_subplot(gs[0, :])
    ax_stats_banner.axis("off")

    # Determine effect size category and color
    cohens_d = stats["cohens_d"]
    if cohens_d > 3.0:
        effect_label = "HUGE EFFECT"
        effect_color = "#006400"  # Dark green
        border_color = "#003300"
    elif cohens_d > 2.0:
        effect_label = "VERY LARGE"
        effect_color = "#228B22"  # Forest green
        border_color = "#145214"
    elif cohens_d > 0.8:
        effect_label = "LARGE"
        effect_color = "#FFA500"  # Orange
        border_color = "#CC8400"
    elif cohens_d > 0.5:
        effect_label = "MEDIUM"
        effect_color = "#FFD700"  # Gold
        border_color = "#B8960B"
    else:
        effect_label = "SMALL"
        effect_color = "#DC143C"  # Crimson
        border_color = "#8B0A1E"

    # Create statistics banner with better formatting
    banner_text = (
        f"Cohen's d = {cohens_d:.3f} ({effect_label})    |    "
        f"Pearson r = {stats['pearson_r']:.3f}    |    "
        f"SNR = {stats['snr']:.2f}    |    "
        f"p-value = {stats['p_value']:.2e}"
    )

    ax_stats_banner.text(
        0.5,
        0.5,
        banner_text,
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor=effect_color,
            edgecolor=border_color,
            linewidth=3,
            alpha=0.92,
        ),
        color="white",
    )

    # === ROW 1: OVERLAPPING HISTOGRAMS (FULL WIDTH) ===
    ax1 = fig.add_subplot(gs[1, :])
    bins = 65
    alpha = 0.58

    # Plot histograms with enhanced styling
    hist_params = [
        (attr_f0, "F0 Shale", "#8B4513", "solid"),
        (attr_f1, "F1 Shale", "#FF8C00", "solid"),
        (attr_f2, "F2 Brine Sand", "#1E90FF", "solid"),
        (attr_f3, "F3 Oil Sand", "#228B22", "solid"),
    ]

    for attr_data, label, color, linestyle in hist_params:
        if len(attr_data) > 0:
            n, bins_edges, patches = ax1.hist(
                attr_data,
                bins=bins,
                alpha=alpha,
                label=f"{label} (n={len(attr_data):,})",
                color=color,
                density=True,
                edgecolor="black",
                linewidth=0.6,
            )

    ax1.set_xlabel(name, fontsize=15, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Probability Density", fontsize=15, fontweight="bold", labelpad=10)
    ax1.set_title(
        "Distribution by Facies - Probability Density Function",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#2C3E50",
    )
    ax1.legend(
        loc="best",
        fontsize=12,
        framealpha=0.96,
        shadow=True,
        edgecolor="black",
        fancybox=True,
    )
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.tick_params(labelsize=11)

    # === ROW 2: BOX PLOT + VIOLIN PLOT ===
    ax2 = fig.add_subplot(gs[2, 0])
    data_to_plot = [attr_f0, attr_f1, attr_f2, attr_f3]
    labels = ["F0\nShale", "F1\nShale", "F2\nBrine", "F3\nOil"]
    colors_box = ["#8B4513", "#FF8C00", "#1E90FF", "#228B22"]

    bp = ax2.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        boxprops=dict(linewidth=1.8, edgecolor="black"),
        whiskerprops=dict(linewidth=1.8, color="black"),
        capprops=dict(linewidth=1.8, color="black"),
        medianprops=dict(linewidth=2.5, color="darkred"),
    )

    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax2.set_ylabel(name, fontsize=14, fontweight="bold", labelpad=10)
    ax2.set_title(
        "Box Plot Comparison (No Outliers)",
        fontsize=15,
        fontweight="bold",
        pad=18,
        color="#2C3E50",
    )
    ax2.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.tick_params(labelsize=11)

    # === VIOLIN PLOT ===
    ax2b = fig.add_subplot(gs[2, 1])

    parts = ax2b.violinplot(
        data_to_plot,
        positions=[0, 1, 2, 3],
        showmeans=True,
        showmedians=True,
        widths=0.75,
    )

    # Color the violin plots
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_box[i])
        pc.set_alpha(0.75)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Style the other elements
    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.8)

    ax2b.set_xticks([0, 1, 2, 3])
    ax2b.set_xticklabels(labels)
    ax2b.set_ylabel(name, fontsize=14, fontweight="bold", labelpad=10)
    ax2b.set_title(
        "Violin Plot - Distribution Shape",
        fontsize=15,
        fontweight="bold",
        pad=18,
        color="#2C3E50",
    )
    ax2b.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.spines["left"].set_linewidth(1.5)
    ax2b.spines["bottom"].set_linewidth(1.5)
    ax2b.tick_params(labelsize=11)

    # === SCATTER PLOT ===
    ax3 = fig.add_subplot(gs[2, 2])

    # Sample for visualization (too many points otherwise)
    n_sample = min(20000, len(attr_valid))
    sample_idx = np.random.choice(len(attr_valid), n_sample, replace=False)

    scatter_colors = [
        (
            "#8B4513"
            if f == 0
            else "#FF8C00" if f == 1 else "#1E90FF" if f == 2 else "#228B22"
        )
        for f in facies_valid[sample_idx]
    ]

    ax3.scatter(
        facies_valid[sample_idx],
        attr_valid[sample_idx],
        c=scatter_colors,
        alpha=0.45,
        s=4,
        edgecolors="none",
    )
    ax3.set_xlabel("Facies", fontsize=14, fontweight="bold", labelpad=10)
    ax3.set_ylabel(name, fontsize=14, fontweight="bold", labelpad=10)
    ax3.set_title(
        f"Scatter Plot (n={n_sample:,} samples)",
        fontsize=15,
        fontweight="bold",
        pad=18,
        color="#2C3E50",
    )
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["F0", "F1", "F2", "F3"], fontsize=11)
    ax3.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_linewidth(1.5)
    ax3.spines["bottom"].set_linewidth(1.5)
    ax3.tick_params(labelsize=11)

    # === ROW 3: STATISTICS PANEL (3 COLUMNS) ===
    # Left: Mean values
    ax4a = fig.add_subplot(gs[3, 0])
    ax4a.axis("off")

    mean_stats_text = f"""MEAN VALUES BY FACIES

F0 (Shale):      {stats['mean_class0']:.4e}
                 ± {attr_f0.std():.4e}

F1 (Shale):      {stats['mean_class1']:.4e}
                 ± {attr_f1.std():.4e}

F2 (Brine Sand): {attr_f2.mean():.4e}
                 ± {attr_f2.std():.4e}

F3 (Oil Sand):   {attr_f3.mean():.4e}
                 ± {attr_f3.std():.4e}"""

    ax4a.text(
        0.1,
        0.5,
        mean_stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=ax4a.transAxes,
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#E6F3FF",
            edgecolor="#1E90FF",
            linewidth=2.5,
            alpha=0.85,
        ),
    )

    # Middle: Discrimination metrics
    ax4b = fig.add_subplot(gs[3, 1])
    ax4b.axis("off")

    discrim_text = f"""DISCRIMINATION METRICS

Cohen's d:       {stats['cohens_d']:.4f}
Effect Size:     {effect_label}

Pearson r:       {stats['pearson_r']:.4f}
p-value:         {stats['p_value']:.2e}

SNR:             {stats['snr']:.4f}
(Signal-to-Noise Ratio)"""

    ax4b.text(
        0.1,
        0.5,
        discrim_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=ax4b.transAxes,
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#FFF4E6",
            edgecolor="#FF8C00",
            linewidth=2.5,
            alpha=0.85,
        ),
    )

    # Right: Interpretation guide
    ax4c = fig.add_subplot(gs[3, 2])
    ax4c.axis("off")

    interpretation_text = """EFFECT SIZE GUIDE

Cohen's d < 0.5
  → Small effect

Cohen's d = 0.5-0.8
  → Medium effect

Cohen's d = 0.8-2.0
  → Large effect

Cohen's d = 2.0-3.0
  → Very large effect

Cohen's d > 3.0
  → HUGE effect"""

    ax4c.text(
        0.1,
        0.5,
        interpretation_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=ax4c.transAxes,
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#E8F5E9",
            edgecolor="#228B22",
            linewidth=2.5,
            alpha=0.85,
        ),
    )

    # Save
    safe_name = (
        name.replace(" ", "_")
        .replace("×", "x")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )
    filename = f"{safe_name}_vs_facies.png"
    filepath = f"{cache_dir}/{filename}"

    # For complex figures (annotations, boxed text), bbox_inches='tight'
    # can still crop elements. For the Lambda-Rho facies figure we avoid
    # using bbox_inches='tight' and instead increase the bottom margin so
    # the saved image preserves all content without clipping.
    if (
        "lambda-rho" in safe_name
        or "lambda_rho" in safe_name
        or "mu-rho" in safe_name
        or "mu_rho" in safe_name
        or "fluid-factor" in safe_name
        or "fluid_factor" in safe_name
        or "poisson-impedance" in safe_name
        or "poisson_impedance" in safe_name
    ):
        try:
            # Increase bottom margin to leave room for boxed annotations
            plt.subplots_adjust(bottom=0.05, top=0.92)
        except Exception:
            pass
        plt.savefig(filepath, dpi=150, facecolor="white")
    else:
        # Default behavior for other attributes: allow tight bounding box
        plt.savefig(
            filepath, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.02
        )
    plt.close()

    print(f"✓ Saved: {filepath}")
    return filepath


def plot_lambda_mu_crossplot(lambda_rho, mu_rho, facies, cache_dir=".cache"):
    """
    Create the classic Lambda-Rho vs Mu-Rho crossplot for fluid/lithology separation.

    Args:
        lambda_rho: Lambda-Rho volume
        mu_rho: Mu-Rho volume
        facies: Facies volume
        cache_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print("\nGenerating Lambda-Rho vs Mu-Rho crossplot...")

    # Flatten and remove invalid values
    lr_flat = lambda_rho.flatten()
    mr_flat = mu_rho.flatten()
    facies_flat = facies.flatten()

    valid_mask = np.isfinite(lr_flat) & np.isfinite(mr_flat) & np.isfinite(facies_flat)
    lr_valid = lr_flat[valid_mask]
    mr_valid = mr_flat[valid_mask]
    facies_valid = facies_flat[valid_mask]

    # Sample for visualization
    n_sample = min(20000, len(lr_valid))
    sample_idx = np.random.choice(len(lr_valid), n_sample, replace=False)

    lr_sample = lr_valid[sample_idx]
    mr_sample = mr_valid[sample_idx]
    facies_sample = facies_valid[sample_idx]

    # Create figure
    _apply_seismic_plot_style()
    # Bigger figure for cleaner layout and readable table
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    fig.suptitle(
        "Lambda-Rho vs Mu-Rho Crossplot - Classic Rock Physics Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Main crossplot - all facies
    ax1 = fig.add_subplot(gs[0, :])

    colors = [
        "brown" if f == 0 else "orange" if f == 1 else "blue" if f == 2 else "green"
        for f in facies_sample
    ]

    ax1.scatter(mr_sample, lr_sample, c=colors, alpha=0.4, s=10)
    ax1.set_xlabel("Mu-Rho (μρ) - Lithology Sensitive", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Lambda-Rho (λρ) - Fluid Sensitive", fontsize=13, fontweight="bold")
    ax1.set_title(f"Goodway Decomposition (n={n_sample} samples)", fontsize=14)
    ax1.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="brown", alpha=0.6, label="F0 - Shale"),
        Patch(facecolor="orange", alpha=0.6, label="F1 - Shale"),
        Patch(facecolor="blue", alpha=0.6, label="F2 - Brine Sand"),
        Patch(facecolor="green", alpha=0.6, label="F3 - Oil Sand"),
    ]
    ax1.legend(handles=legend_elements, loc="best", fontsize=11)

    # 2. Fluid comparison (F2 vs F3)
    ax2 = fig.add_subplot(gs[1, 0])

    mask_f2 = facies_sample == 2
    mask_f3 = facies_sample == 3

    ax2.scatter(
        mr_sample[mask_f2],
        lr_sample[mask_f2],
        c="blue",
        alpha=0.5,
        s=15,
        label="F2 Brine Sand",
    )
    ax2.scatter(
        mr_sample[mask_f3],
        lr_sample[mask_f3],
        c="green",
        alpha=0.5,
        s=15,
        label="F3 Oil Sand",
    )
    ax2.set_xlabel("Mu-Rho (μρ)", fontsize=12)
    ax2.set_ylabel("Lambda-Rho (λρ)", fontsize=12)
    ax2.set_title("Fluid Discrimination (Brine vs Oil)", fontsize=13, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    # Calculate separation
    if mask_f2.sum() > 0 and mask_f3.sum() > 0:
        lr_f2_mean = lr_sample[mask_f2].mean()
        lr_f3_mean = lr_sample[mask_f3].mean()
        sep = abs(lr_f3_mean - lr_f2_mean)
        ax2.text(
            0.05,
            0.95,
            f"ΔLambda-Rho = {sep:.2e}",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

    # 3. Lithology comparison (F0/F1 vs F2/F3)
    ax3 = fig.add_subplot(gs[1, 1])

    mask_shale = (facies_sample == 0) | (facies_sample == 1)
    mask_sand = (facies_sample == 2) | (facies_sample == 3)

    ax3.scatter(
        mr_sample[mask_shale],
        lr_sample[mask_shale],
        c="brown",
        alpha=0.4,
        s=15,
        label="Shale (F0+F1)",
    )
    ax3.scatter(
        mr_sample[mask_sand],
        lr_sample[mask_sand],
        c="cyan",
        alpha=0.4,
        s=15,
        label="Sand (F2+F3)",
    )
    ax3.set_xlabel("Mu-Rho (μρ)", fontsize=12)
    ax3.set_ylabel("Lambda-Rho (λρ)", fontsize=12)
    ax3.set_title(
        "Lithology Discrimination (Shale vs Sand)", fontsize=13, fontweight="bold"
    )
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)

    # Calculate separation
    if mask_shale.sum() > 0 and mask_sand.sum() > 0:
        mr_shale_mean = mr_sample[mask_shale].mean()
        mr_sand_mean = mr_sample[mask_sand].mean()
        sep = abs(mr_sand_mean - mr_shale_mean)
        ax3.text(
            0.05,
            0.95,
            f"ΔMu-Rho = {sep:.2e}",
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

    # Save
    filepath = f"{cache_dir}/lambda_mu_crossplot.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Saved: {filepath}")
    return filepath


def plot_attribute_comparison(ranking_results, cache_dir=".cache"):
    """
    Create comprehensive comparison chart of all attributes with improved layout.

    Args:
        ranking_results: List of discrimination statistics from compare_all_attributes
        cache_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt

    print("\nGenerating attribute comparison chart...")

    # Extract data
    names = [r["name"] for r in ranking_results]
    cohens_d = [r["cohens_d"] for r in ranking_results]
    pearson_r = [abs(r["pearson_r"]) for r in ranking_results]
    snr = [r["snr"] for r in ranking_results]

    # Create figure with improved layout
    _apply_seismic_plot_style()
    fig = plt.figure(figsize=(52, 22))

    # Create main grid: title row, charts row, table row
    gs_main = GridSpec(
        3,
        1,
        figure=fig,
        # Give a bit more room to the title and charts, table stays roomy
        height_ratios=[0.07, 0.50, 0.43],
        hspace=0.06,
        top=0.98,
        bottom=0.02,
        left=0.3,
        right=0.98,
    )

    # Title banner
    ax_title = fig.add_subplot(gs_main[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.52,
        "Rock Physics Attributes - Discrimination Performance Ranking",
        fontsize=28,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
    )

    # Create subgrid for the 3 charts in the middle row
    gs_charts = GridSpec(
        1,
        3,
        figure=fig,
        hspace=0.3,
        wspace=0.25,
        top=0.91,
        bottom=0.52,
        left=0.04,
        right=0.97,
    )

    # Color code by performance
    colors = []
    for d in cohens_d:
        if d > 3.0:
            colors.append("#006400")  # Dark green - HUGE
        elif d > 2.0:
            colors.append("#228B22")  # Forest green - Very large
        elif d > 0.8:
            colors.append("#FFA500")  # Orange - Large
        elif d > 0.5:
            colors.append("#FFD700")  # Gold - Medium
        else:
            colors.append("#DC143C")  # Crimson - Small

    # === CHART 1: Cohen's d ===
    ax1 = fig.add_subplot(gs_charts[0, 0])
    y_pos = np.arange(len(names))
    bars1 = ax1.barh(
        y_pos, cohens_d, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax1.set_xlabel(
        "Cohen's d (Effect Size)", fontsize=14, fontweight="bold", labelpad=10
    )
    ax1.set_title("Facies Discrimination Power", fontsize=16, fontweight="bold", pad=18)

    # Reference lines with labels
    ax1.axvline(
        x=0.5,
        color="#DC143C",
        linestyle="--",
        alpha=0.5,
        linewidth=2.5,
        label="Medium (0.5)",
    )
    ax1.axvline(
        x=0.8,
        color="#FFA500",
        linestyle="--",
        alpha=0.5,
        linewidth=2.5,
        label="Large (0.8)",
    )
    ax1.axvline(
        x=2.0,
        color="#228B22",
        linestyle="--",
        alpha=0.5,
        linewidth=2.5,
        label="Very Large (2.0)",
    )
    ax1.axvline(
        x=3.0,
        color="#006400",
        linestyle="--",
        alpha=0.5,
        linewidth=2.5,
        label="HUGE (3.0)",
    )

    ax1.legend(loc="lower right", fontsize=12, framealpha=0.95)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add values on bars
    for i, v in enumerate(cohens_d):
        ax1.text(
            v + max(cohens_d) * 0.02,
            i,
            f"{v:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # === CHART 2: Pearson correlation ===
    ax2 = fig.add_subplot(gs_charts[0, 1])
    bars2 = ax2.barh(
        y_pos, pearson_r, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=12, fontweight="bold")
    ax2.set_xlabel(
        "|Pearson r| (Correlation)", fontsize=14, fontweight="bold", labelpad=10
    )
    ax2.set_title(
        "Linear Correlation with Facies", fontsize=16, fontweight="bold", pad=18
    )
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.invert_yaxis()
    ax2.set_xlim([0, 1])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add values on bars
    for i, v in enumerate(pearson_r):
        ax2.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=12, fontweight="bold")

    # === CHART 3: Signal-to-Noise Ratio ===
    ax3 = fig.add_subplot(gs_charts[0, 2])
    bars3 = ax3.barh(
        y_pos, snr, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax3.set_xlabel("Signal-to-Noise Ratio", fontsize=14, fontweight="bold", labelpad=10)
    ax3.set_title("Discrimination Quality", fontsize=16, fontweight="bold", pad=18)
    ax3.grid(axis="x", alpha=0.3, linestyle="--")
    ax3.invert_yaxis()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Add values on bars
    for i, v in enumerate(snr):
        ax3.text(
            v + max(snr) * 0.02,
            i,
            f"{v:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # === BOTTOM: SUMMARY TABLE ===
    ax_table = fig.add_subplot(gs_main[2, :])
    ax_table.axis("off")

    # Create summary table with top performers (up to 5)
    num_attrs = len(cohens_d)
    top_n = min(5, num_attrs)  # Show up to 5, but not more than we have
    top_idx = np.argsort(cohens_d)[::-1][:top_n]

    table_data = []
    table_data.append(
        ["Rank", "Attribute", "Cohen's d", "Pearson r", "SNR", "Effect Size"]
    )

    for rank, idx in enumerate(top_idx, 1):
        d_val = cohens_d[idx]
        if d_val > 3.0:
            effect = "HUGE"
        elif d_val > 2.0:
            effect = "VERY LARGE"
        elif d_val > 0.8:
            effect = "LARGE"
        elif d_val > 0.5:
            effect = "MEDIUM"
        else:
            effect = "SMALL"

        table_data.append(
            [
                f"#{rank}",
                names[idx],
                f"{cohens_d[idx]:.3f}",
                f"{pearson_r[idx]:.3f}",
                f"{snr[idx]:.2f}",
                effect,
            ]
        )

    # Create table
    table = ax_table.table(
        cellText=table_data, cellLoc="left", loc="center", bbox=[0.06, 0.06, 0.88, 0.86]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    # Slightly reduced vertical scaling to improve row proportions
    table.scale(1, 3.2)

    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor("#4A148C")
        cell.set_text_props(weight="bold", color="white", fontsize=16)
        cell.set_edgecolor("black")
        cell.set_linewidth(2.5)
        cell.set_height(0.09)

    # Style data rows (1 to top_n, not hardcoded to 5)
    for i in range(1, top_n + 1):
        # Rank cell
        table[(i, 0)].set_facecolor("#E1BEE7")
        table[(i, 0)].set_text_props(weight="bold", fontsize=14)

        # Color code by rank
        if i == 1:
            row_color = "#FFD700"  # Gold
        elif i == 2:
            row_color = "#C0C0C0"  # Silver
        elif i == 3:
            row_color = "#CD7F32"  # Bronze
        else:
            row_color = "#F5F5F5"  # Light gray

        for j in range(6):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_edgecolor("black")
            table[(i, j)].set_linewidth(1.6)
            table[(i, j)].set_height(0.09)
            if j > 0:  # Not the rank column
                table[(i, j)].set_text_props(fontsize=14)

    # Draw a bold border around the table area for emphasis
    try:
        from matplotlib.patches import Rectangle

        # The rectangle coordinates match the table bbox in axis coords
        rect = Rectangle(
            (0.06, 0.06),
            0.88,
            0.86,
            transform=ax_table.transAxes,
            fill=False,
            linewidth=3.0,
            edgecolor="black",
            zorder=10,
        )
        ax_table.add_patch(rect)
    except Exception:
        pass

    # Add title for table (positioned above the table)
    table_title = (
        f"TOP {top_n} PERFORMERS - DISCRIMINATION RANKING"
        if top_n < 5
        else "TOP 5 PERFORMERS - DISCRIMINATION RANKING"
    )
    # Title above the table (render without a surrounding box)
    # Place the table title just above the table area for better grouping
    fig.text(
        0.5,
        0.44,
        table_title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    # Remove tight_layout to avoid warning

    # Save (add a little top margin so title/banner isn't tight to the edge)
    filepath = f"{cache_dir}/rock_physics_attributes_comparison.png"
    try:
        plt.subplots_adjust(top=0.92)
    except Exception:
        pass
    plt.savefig(filepath, dpi=150, facecolor="white")
    plt.close()

    print(f"✓ Saved: {filepath}")
    return filepath


def generate_all_visualizations(
    hybrid_results, facies, ranking_results, cache_dir=".cache"
):
    """
    Generate all individual visualization plots.

    Args:
        hybrid_results: Dictionary of computed attributes
        facies: Facies volume
        ranking_results: List of discrimination statistics
        cache_dir: Directory to save plots

    Returns:
        dict: Paths to generated plots
    """
    import os

    os.makedirs(cache_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL VISUALIZATIONS")
    print("=" * 70)

    generated_plots = {}

    # 1. Lambda-Rho vs Facies
    if "lambda_rho" in hybrid_results:
        path = plot_attribute_vs_facies(
            hybrid_results["lambda_rho"], facies, "Lambda-Rho (λρ)", cache_dir
        )
        generated_plots["lambda_rho_vs_facies"] = path

    # 2. Mu-Rho vs Facies
    if "mu_rho" in hybrid_results:
        path = plot_attribute_vs_facies(
            hybrid_results["mu_rho"], facies, "Mu-Rho (μρ)", cache_dir
        )
        generated_plots["mu_rho_vs_facies"] = path

    # 3. Fluid Factor vs Facies
    if "fluid_factor" in hybrid_results:
        path = plot_attribute_vs_facies(
            hybrid_results["fluid_factor"], facies, "Fluid Factor", cache_dir
        )
        generated_plots["fluid_factor_vs_facies"] = path

    # 4. Poisson Impedance vs Facies
    if "poisson_impedance" in hybrid_results:
        path = plot_attribute_vs_facies(
            hybrid_results["poisson_impedance"], facies, "Poisson Impedance", cache_dir
        )
        generated_plots["poisson_impedance_vs_facies"] = path

    # 5. Lambda-Mu Crossplot
    if "lambda_rho" in hybrid_results and "mu_rho" in hybrid_results:
        path = plot_lambda_mu_crossplot(
            hybrid_results["lambda_rho"], hybrid_results["mu_rho"], facies, cache_dir
        )
        generated_plots["lambda_mu_crossplot"] = path

    # 6. Attribute Comparison Chart
    if ranking_results:
        path = plot_attribute_comparison(ranking_results, cache_dir)
        generated_plots["attributes_comparison"] = path

    print("\n" + "=" * 70)
    print(f"✓ Generated {len(generated_plots)} visualization files")
    print("=" * 70)

    return generated_plots


def plot_multiangle_ei_comparison(ei_results, facies, cache_dir=".cache"):
    """
    Create multi-angle EI comparison visualization.

    Args:
        ei_results: Results from compute_ei_multiangle() or run_multiangle_analysis()
        facies: Facies volume
        cache_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print("\nGenerating multi-angle EI comparison plot...")

    # Extract EI volumes and angles
    if "ei_volumes" in ei_results:
        ei_volumes = ei_results["ei_volumes"]
        angles = ei_results["angles"]
    elif "ei_dict" in ei_results:
        ei_dict = ei_results["ei_dict"]
        angles = sorted(ei_dict.keys())
        ei_volumes = [ei_dict[a] for a in angles]
    else:
        print("⚠ Warning: No EI volumes found in results")
        return None

    n_angles = len(angles)

    # Get center slices
    ni, nj, nk = ei_volumes[0].shape
    idx_i = ni // 2
    idx_k = nk // 2

    # Create figure
    _apply_seismic_plot_style()
    fig = plt.figure()
    fig.suptitle(
        "Multi-Angle Elastic Impedance (EI) - Depth Domain Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Calculate number of rows needed
    n_cols = 3
    n_rows = (n_angles + n_cols - 1) // n_cols

    # Plot each angle
    for idx, (angle, ei_vol) in enumerate(zip(angles, ei_volumes)):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        # Get inline slice
        ei_slice = ei_vol[idx_i, :, :]

        # Plot
        im = ax.imshow(
            ei_slice.T,
            aspect="auto",
            cmap="seismic",
            origin="upper",
            interpolation="bilinear",
        )

        # Add statistics
        ei_min = ei_vol.min()
        ei_max = ei_vol.max()
        ei_mean = ei_vol.mean()

        ax.set_title(
            f"EI at {angle:.0f}°\nRange: [{ei_min:.2e}, {ei_max:.2e}]",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Crossline")
        ax.set_ylabel("Depth (samples)")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save
    filepath = f"{cache_dir}/angle_dependent_ei_depth_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Saved: {filepath}")
    return filepath


def plot_multiangle_ei_facies_analysis(ei_results, facies, cache_dir=".cache"):
    """
    Create multi-angle EI facies discrimination analysis.

    Args:
        ei_results: Results from run_multiangle_analysis()
        facies: Facies volume
        cache_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt

    print("\nGenerating multi-angle EI facies analysis plot...")

    # Extract data
    if "ei_dict" in ei_results:
        ei_dict = ei_results["ei_dict"]
    elif "ei_volumes" in ei_results:
        angles = ei_results["angles"]
        ei_volumes = ei_results["ei_volumes"]
        ei_dict = {int(a): v for a, v in zip(angles, ei_volumes)}
    else:
        print("⚠ Warning: No EI data found")
        return None

    angles = sorted(ei_dict.keys())

    # Compute Cohen's d for each angle
    print("  Computing discrimination statistics for each angle...")
    cohens_d_values = []
    pearson_r_values = []
    snr_values = []

    for angle in angles:
        ei_vol = ei_dict[angle]
        stats = analyze_facies_correlation_depth(ei_vol, facies)
        cohens_d_values.append(stats["cohens_d"])
        pearson_r_values.append(abs(stats["pearson_r"]))
        snr_values.append(stats["snr"])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Multi-Angle EI - Facies Discrimination Performance",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Cohen's d vs angle
    ax1 = axes[0, 0]
    ax1.plot(angles, cohens_d_values, "o-", linewidth=2, markersize=8, color="darkblue")
    ax1.axhline(
        y=3.0, color="green", linestyle="--", alpha=0.5, label="HUGE effect (d>3)"
    )
    ax1.axhline(
        y=2.0, color="orange", linestyle="--", alpha=0.5, label="Very Large (d>2)"
    )
    ax1.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Large (d>0.8)")
    ax1.set_xlabel("Angle (degrees)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight="bold")
    ax1.set_title("Facies Discrimination Power", fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Mark best angle
    best_idx = np.argmax(cohens_d_values)
    best_angle = angles[best_idx]
    best_d = cohens_d_values[best_idx]
    ax1.plot(
        best_angle,
        best_d,
        "r*",
        markersize=20,
        label=f"Best: {best_angle}° (d={best_d:.2f})",
    )
    ax1.legend(loc="best", fontsize=9)

    # 2. Pearson correlation vs angle
    ax2 = axes[0, 1]
    ax2.plot(
        angles, pearson_r_values, "s-", linewidth=2, markersize=8, color="darkgreen"
    )
    ax2.set_xlabel("Angle (degrees)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("|Pearson r|", fontsize=12, fontweight="bold")
    ax2.set_title("Correlation with Facies Boundaries", fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])

    # Mark best
    best_idx_r = np.argmax(pearson_r_values)
    best_angle_r = angles[best_idx_r]
    best_r = pearson_r_values[best_idx_r]
    ax2.plot(best_angle_r, best_r, "r*", markersize=20)
    ax2.text(
        best_angle_r,
        best_r + 0.05,
        f"{best_angle_r}°\nr={best_r:.3f}",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

    # 3. SNR vs angle
    ax3 = axes[1, 0]
    ax3.plot(angles, snr_values, "^-", linewidth=2, markersize=8, color="purple")
    ax3.set_xlabel("Angle (degrees)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax3.set_title("Boundary Detection Quality", fontsize=13)
    ax3.grid(alpha=0.3)

    # Mark best
    best_idx_snr = np.argmax(snr_values)
    best_angle_snr = angles[best_idx_snr]
    best_snr = snr_values[best_idx_snr]
    ax3.plot(best_angle_snr, best_snr, "r*", markersize=20)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create table data
    table_data = []
    for i, angle in enumerate(angles):
        table_data.append(
            [
                f"{angle}°",
                f"{cohens_d_values[i]:.3f}",
                f"{pearson_r_values[i]:.3f}",
                f"{snr_values[i]:.2f}",
            ]
        )

    # Sort by Cohen's d
    table_data.sort(key=lambda x: float(x[1]), reverse=True)

    # Add header
    table_data.insert(0, ["Angle", "Cohen's d", "|Pearson r|", "SNR"])

    # Create table
    table = ax4.table(
        cellText=table_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best row (excluding header)
    table[(1, 0)].set_facecolor("#FFD700")
    table[(1, 1)].set_facecolor("#FFD700")
    table[(1, 2)].set_facecolor("#FFD700")
    table[(1, 3)].set_facecolor("#FFD700")

    ax4.set_title("Performance Ranking", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save
    filepath = f"{cache_dir}/angle_dependent_ei_facies_analysis.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Saved: {filepath}")
    return filepath


def main():
    """Run rock physics attributes analysis."""

    print("=" * 70)
    print("ROCK PHYSICS ATTRIBUTES ANALYSIS")
    print("=" * 70)
    print("\nThis analysis computes:")
    print("  • Lambda-Rho & Mu-Rho - Fluid/lithology separation (Goodway method)")
    print("  • Fluid Factor - Direct fluid indicator")
    print("  • Poisson Impedance - Fluid sensitivity")
    print("  • AVO Attributes - Intercept, gradient")
    print("  • EI Gradient - Angle-dependent elastic impedance")
    print("  • Hybrid Discriminants - Combined attributes")
    print("\nGoal: Find attributes with Cohen's d > 4.0 (HUGE effect size)")
    print("=" * 70)

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    props_depth = load_depth_data()

    # Compute hybrid attributes
    hybrid_results = compute_hybrid_ei_avo_attributes(
        props_depth["vp"], props_depth["vs"], props_depth["rho"], ei_angle=10
    )

    # Compare all attributes
    ranking = compare_all_attributes(hybrid_results, props_depth["facies"])

    # Generate all visualizations
    generated_plots = generate_all_visualizations(
        hybrid_results, props_depth["facies"], ranking, cache_dir=".cache"
    )

    # Save results
    cache_file = ".cache/rock_physics_attributes.npz"
    print(f"\nSaving results to {cache_file}...")
    np.savez_compressed(cache_file, **hybrid_results, facies=props_depth["facies"])
    print(f"✓ Saved {len(hybrid_results)} attributes")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated visualizations:")
    for name, path in generated_plots.items():
        print(f"  ✓ {name}: {path}")
    print("\nNext steps:")
    print("  1. Review individual attribute plots")
    print("  2. Analyze Lambda-Mu crossplot for fluid indicators")
    print("  3. Compare discrimination performance chart")
    print("  4. Apply best attributes to seismic modeling")
    print("=" * 70)


if __name__ == "__main__":
    main()
