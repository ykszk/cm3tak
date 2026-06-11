"""
DWI Structural Connectome Analysis Pipeline
============================================
For use with Connectome Mapper 3 (CMP3) outputs (.gpickle files).
Computes graph metrics suitable for academic publication.

Requirements:
    pip install networkx bctpy numpy scipy matplotlib seaborn nibabel

Usage:
    python connectome_analysis.py [options]
    Run with --help to see all available arguments.
"""

import argparse
import os
import pickle
import warnings
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

try:
    import bct
except ImportError:
    raise ImportError("Install bctpy: pip install bctpy")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="DWI Structural Connectome Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gpickle_dir",
        help="Path to folder containing subject subfolders with .gpickle files",
    )
    parser.add_argument(
        "--edge-weight",
        default="fiber_density",
        choices=["fiber_density", "fiber_number", "fiber_length", "FA"],
        help="Edge weight attribute to use from the connectome graph",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.15,
        metavar="FLOAT",
        help="Proportional threshold: keep top fraction of connections (0.0–1.0). Pass 0 to skip.",
    )
    parser.add_argument(
        "--n-rand",
        type=int,
        default=100,
        help="Number of random null networks per subject",
    )
    parser.add_argument(
        "--rand-itr",
        type=int,
        default=10,
        help="Number of rewiring iterations per edge (for randmio_und)",
    )
    parser.add_argument(
        "--cache-dir",
        default="./data/null_cache",
        help="Directory for null network cache files",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-plot-matrices",
        action="store_true",
        help="Disable plotting of adjacency matrices",
    )
    parser.add_argument(
        "--scale-label",
        default="scale3",
        help="Parcellation scale label used in filenames and logs",
    )

    args = parser.parse_args()
    return {
        "gpickle_dir": args.gpickle_dir,
        "edge_weight": args.edge_weight,
        "threshold_pct": args.threshold_pct if args.threshold_pct > 0 else None,
        "n_rand": args.n_rand,
        "rand_itr": args.rand_itr,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "plot_matrices": not args.no_plot_matrices,
        "scale_label": args.scale_label,
    }

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────

def load_gpickle(path: str, weight: str) -> np.ndarray:
    """Load a CMP3 .gpickle file and return an adjacency matrix.
    nx.read_gpickle was removed in NetworkX 3.0 — load manually with pickle.
    """
    with open(path, "rb") as f:
        G = pickle.load(f)
    A = nx.to_numpy_array(G, weight=weight)
    np.fill_diagonal(A, 0)          # remove self-connections
    A = np.nan_to_num(A, nan=0.0)  # replace NaN with 0
    A = (A + A.T) / 2               # symmetrize (should already be, but safety check)
    return A


def log_transform(A: np.ndarray) -> np.ndarray:
    """Log-transform edge weights (handles skewed streamline distributions)."""
    return np.log1p(A)


def proportional_threshold(A: np.ndarray, keep_pct: float) -> np.ndarray:
    """
    Keep the top `keep_pct` fraction of edges (proportional thresholding).
    Preserves equal density across subjects — preferred over absolute thresholding.
    """
    A_thresh = A.copy()
    # Work on upper triangle only
    triu_vals = A_thresh[np.triu_indices_from(A_thresh, k=1)]
    nonzero = triu_vals[triu_vals > 0]
    if len(nonzero) == 0:
        return A_thresh
    cutoff = np.percentile(nonzero, (1 - keep_pct) * 100)
    A_thresh[A_thresh < cutoff] = 0
    return A_thresh


def consistency_threshold(matrices: list, percentile: float = 75) -> np.ndarray:
    """
    Group-consistency thresholding (Roberts et al., 2017).
    Removes edges with high coefficient of variation (CV) across subjects.
    Returns a binary mask to apply to each subject's matrix.
    """
    stack = np.array(matrices)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean, np.inf)
    threshold_val = np.percentile(cv[np.isfinite(cv)], percentile)
    mask = cv <= threshold_val
    return mask


# ─────────────────────────────────────────────
# 2. NULL RANDOM NETWORKS (CACHED)
# ─────────────────────────────────────────────

def _compute_one_null(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    A, itr = args
    A_rand, _ = bct.randmio_und(A, itr)
    c = bct.clustering_coef_wu(A_rand).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        W_inv = np.where(A_rand > 0, 1.0 / A_rand, 0)
    D_rand, _ = bct.distance_wei(W_inv)
    l, _, _ecc, _rad, _diam = bct.charpath(D_rand, include_infinite=False)
    return c, l

def compute_null_metrics(
    A: np.ndarray,
    subject_id: str,
    n_rand: int = 100,
    itr: int = 10,
    cache_dir: str = "./null_cache",
) -> dict:
    """
    Compute null random network metrics for small-worldness.
    Results are cached to disk — recalculated only if matrix changes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{subject_id}_null_n{n_rand}.pkl")

    if os.path.exists(cache_file):
        log.info(f"  Loading cached null networks: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    n_workers = min(n_rand, os.cpu_count() or 1)
    log.info(f"  Computing {n_rand} null networks for {subject_id} with {n_workers} workers...")

    C_rand_list, L_rand_list = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_compute_one_null, (A, itr)) for _ in range(n_rand)]
        for fut in as_completed(futures):
            c, l = fut.result()
            C_rand_list.append(c)
            L_rand_list.append(l)

    results = {
        "C_rand": np.array(C_rand_list),
        "L_rand": np.array(L_rand_list),
        "n_rand": n_rand,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    log.info(f"  Saved null cache: {cache_file}")

    return results


# ─────────────────────────────────────────────
# 3. GRAPH METRICS
# ─────────────────────────────────────────────

def compute_global_metrics(A: np.ndarray, null: dict) -> dict:
    """
    Compute global graph theory metrics.
    All metrics are standard for structural connectome publications.
    """
    metrics = {}

    # --- Clustering coefficient (weighted) ---
    cc = bct.clustering_coef_wu(A)
    metrics["clustering_coef"] = cc.mean()

    # --- Distance matrix (inverse weight) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        W_inv = np.where(A > 0, 1.0 / A, 0)
    D, _ = bct.distance_wei(W_inv)

    # --- Characteristic path length & global efficiency ---
    L, Eglob, _ecc, _rad, _diam = bct.charpath(D, include_infinite=False)
    metrics["char_path_length"] = L
    metrics["global_efficiency"] = Eglob

    # --- Local efficiency ---
    Eloc = bct.efficiency_wei(A, local=True)
    metrics["local_efficiency"] = Eloc.mean()

    # --- Modularity (Louvain; run multiple times for stability) ---
    Q_vals, ci_vals = [], []
    for seed in range(10):
        np.random.seed(seed)
        ci, Q = bct.modularity_louvain_und(A)
        Q_vals.append(Q)
        ci_vals.append(ci)
    best_idx = int(np.argmax(Q_vals))
    metrics["modularity_Q"] = Q_vals[best_idx]
    metrics["community_assignments"] = ci_vals[best_idx]

    # --- Small-worldness (sigma) ---
    C_rand_mean = null["C_rand"].mean()
    L_rand_mean = null["L_rand"].mean()
    gamma = metrics["clustering_coef"] / C_rand_mean   # normalized clustering (>1 = small-world)
    lambda_ = L / L_rand_mean                          # normalized path length (≈1 = small-world)
    metrics["gamma"] = gamma
    metrics["lambda"] = lambda_
    metrics["sigma"] = gamma / lambda_                 # small-world coefficient (>1 = small-world)

    # --- Rich-club coefficient (normalized) ---
    rc = bct.rich_club_wu(A)
    metrics["rich_club_coef"] = rc

    return metrics


def compute_node_metrics(A: np.ndarray, ci: np.ndarray) -> dict:
    """
    Compute node-level graph metrics (returned as arrays, length = n_nodes).
    """
    node = {}

    # Strength (weighted degree)
    node["strength"] = A.sum(axis=1)

    # Weighted clustering coefficient
    node["clustering_coef"] = bct.clustering_coef_wu(A)

    # Betweenness centrality
    with np.errstate(divide="ignore", invalid="ignore"):
        W_inv = np.where(A > 0, 1.0 / A, 0)
    D, _ = bct.distance_wei(W_inv)
    node["betweenness"] = bct.betweenness_wei(D)

    # Within-module degree z-score & participation coefficient
    node["within_module_z"] = bct.module_degree_zscore(A, ci)
    node["participation_coef"] = bct.participation_coef(A, ci)

    # Local efficiency
    node["local_efficiency"] = bct.efficiency_wei(A, local=True)

    # Eigenvector centrality (via networkx — bct doesn't have it)
    # Use the largest connected component; isolated nodes get 0.
    G_nx = nx.from_numpy_array(A)
    ec_values = np.zeros(A.shape[0])
    lcc = max(nx.connected_components(G_nx), key=len)
    G_lcc = G_nx.subgraph(lcc)
    ec_lcc = nx.eigenvector_centrality_numpy(G_lcc, weight="weight")
    for node_idx, val in ec_lcc.items():
        ec_values[node_idx] = val
    node["eigenvector_centrality"] = ec_values

    return node


def classify_hubs(within_module_z: np.ndarray, participation_coef: np.ndarray) -> np.ndarray:
    """
    Classify nodes into hub types (Guimerà & Amaral, 2005):
      - Connector hub:  Z > 2.5 AND P > 0.3
      - Provincial hub: Z > 2.5 AND P <= 0.3
      - Non-hub:        Z <= 2.5
    Returns array of strings.
    """
    labels = np.array(["non-hub"] * len(within_module_z), dtype=object)
    is_hub = within_module_z > 2.5
    labels[is_hub & (participation_coef > 0.3)] = "connector-hub"
    labels[is_hub & (participation_coef <= 0.3)] = "provincial-hub"
    return labels


# ─────────────────────────────────────────────
# 4. GROUP STATISTICS
# ─────────────────────────────────────────────

def compare_groups(
    metrics_group1: list,
    metrics_group2: list,
    metric_name: str,
    n_permutations: int = 5000,
) -> dict:
    """
    Non-parametric permutation test for group differences in a scalar metric.
    Preferred over t-tests for graph metrics (non-normal distributions).
    """
    g1 = np.array([m[metric_name] for m in metrics_group1])
    g2 = np.array([m[metric_name] for m in metrics_group2])

    observed_diff = g1.mean() - g2.mean()
    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    perm_diffs = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diffs.append(perm[:n1].mean() - perm[n1:].mean())

    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else np.nan

    return {
        "metric": metric_name,
        "group1_mean": g1.mean(),
        "group1_std": g1.std(),
        "group2_mean": g2.mean(),
        "group2_std": g2.std(),
        "observed_diff": observed_diff,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_permutations": n_permutations,
    }


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple:
    """Benjamini-Hochberg FDR correction for multiple comparisons."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    reject = sorted_p <= thresholds
    # Find largest k where p[k] <= threshold[k]
    if reject.any():
        max_k = np.where(reject)[0].max()
        reject_final = np.zeros(n, dtype=bool)
        reject_final[sorted_idx[: max_k + 1]] = True
    else:
        reject_final = np.zeros(n, dtype=bool)
    return reject_final, thresholds[sorted_idx]


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

def plot_connectivity_matrix(A: np.ndarray, subject_id: str, output_dir: str):
    """Plot adjacency matrix with log-scale colormap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = "inferno"

    # Raw matrix
    im0 = axes[0].imshow(A, cmap=cmap, aspect="auto")
    axes[0].set_title(f"{subject_id}\nRaw Connectivity Matrix")
    axes[0].set_xlabel("ROI index")
    axes[0].set_ylabel("ROI index")
    plt.colorbar(im0, ax=axes[0], label="Edge weight")

    # Log-transformed
    A_log = np.log1p(A)
    im1 = axes[1].imshow(A_log, cmap=cmap, aspect="auto")
    axes[1].set_title(f"{subject_id}\nLog-transformed Matrix")
    axes[1].set_xlabel("ROI index")
    axes[1].set_ylabel("ROI index")
    plt.colorbar(im1, ax=axes[1], label="log(1 + edge weight)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{subject_id}_connectivity_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved matrix plot: {out_path}")


def plot_group_metrics(all_metrics: list, subject_ids: list, output_dir: str):
    """Violin/strip plot of global metrics across all subjects."""
    scalar_keys = [
        "clustering_coef", "char_path_length", "global_efficiency",
        "local_efficiency", "modularity_Q", "rich_club_coef", "sigma", "gamma", "lambda",
    ]

    data = {k: [m[k] for m in all_metrics] for k in scalar_keys if k in all_metrics[0]}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (key, vals) in zip(axes, data.items()):
        ax.violinplot(vals, positions=[0], showmedians=True)
        ax.scatter([0] * len(vals), vals, color="black", s=20, zorder=3)
        ax.set_xticks([])
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylabel("Value")

    plt.suptitle("Group-Level Graph Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "group_graph_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved group metrics plot: {out_path}")


def plot_hub_map(node_metrics: dict, subject_id: str, output_dir: str):
    """Scatter plot of within-module Z vs participation coefficient (hub map)."""
    Z = node_metrics["within_module_z"]
    P = node_metrics["participation_coef"]
    hub_labels = classify_hubs(Z, P)

    colors = {"non-hub": "lightgray", "provincial-hub": "steelblue", "connector-hub": "crimson"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in colors.items():
        mask = hub_labels == label
        ax.scatter(P[mask], Z[mask], c=color, label=label, alpha=0.7, edgecolors="k", linewidths=0.3)

    ax.axhline(2.5, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0.3, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Participation Coefficient (P)")
    ax.set_ylabel("Within-Module Degree Z-Score (Z)")
    ax.set_title(f"{subject_id} — Hub Classification")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{subject_id}_hub_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved hub map: {out_path}")


# ─────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(all_results: dict, output_dir: str):
    """Save all results as numpy .npz and a human-readable summary TSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full results as pickle
    results_pkl = os.path.join(output_dir, "all_results.pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump(all_results, f)
    log.info(f"Saved full results: {results_pkl}")

    # Save global metrics summary as TSV
    scalar_keys = [
        "clustering_coef", "char_path_length", "global_efficiency",
        "local_efficiency", "modularity_Q", "sigma", "gamma", "lambda",
    ]
    tsv_path = os.path.join(output_dir, "global_metrics_summary.tsv")
    with open(tsv_path, "w") as f:
        header = ["subject_id"] + scalar_keys
        f.write("\t".join(header) + "\n")
        for subj_id, res in all_results.items():
            gm = res["global_metrics"]
            row = [subj_id] + [str(round(gm.get(k, np.nan), 6)) for k in scalar_keys]
            f.write("\t".join(row) + "\n")
    log.info(f"Saved global metrics TSV: {tsv_path}")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def find_gpickle_files(directory: str) -> list:
    """Recursively find all .gpickle files in a directory."""
    return sorted(Path(directory).rglob("*conndata-network_connectivity.gpickle"))


def _process_one_subject(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    gpickle_path, config = args
    subject_id = gpickle_path.stem.split("_conndata")[0]
    log.info(f"\nProcessing: {subject_id}")

    # --- Load ---
    try:
        A_raw = load_gpickle(str(gpickle_path), config["edge_weight"])
    except Exception as e:
        log.error(f"  Failed to load {gpickle_path}: {e}")
        return subject_id, None

    log.info(f"  Matrix shape: {A_raw.shape}, density: {(A_raw > 0).mean():.3f}")

    # --- Log-transform ---
    A = log_transform(A_raw)

    # --- Threshold ---
    if config["threshold_pct"] is not None:
        A = proportional_threshold(A, keep_pct=config["threshold_pct"])
        density = (A > 0).mean()
        log.info(f"  Post-threshold density: {density:.3f} (top {config['threshold_pct']*100:.0f}%)")

    # --- Plot matrix ---
    if config["plot_matrices"]:
        plot_connectivity_matrix(A_raw, subject_id, config["output_dir"])

    # --- Null networks (cached) ---
    null = compute_null_metrics(
        A,
        subject_id=subject_id,
        n_rand=config["n_rand"],
        itr=config["rand_itr"],
        cache_dir=config["cache_dir"],
    )

    # --- Global metrics ---
    log.info("  Computing global metrics...")
    global_metrics = compute_global_metrics(A, null)
    ci = global_metrics.pop("community_assignments")
    rc = global_metrics.pop("rich_club_coef")

    log.info(f"  σ (small-world) = {global_metrics['sigma']:.3f}  "
             f"(>1 = small-world; γ={global_metrics['gamma']:.3f}, λ={global_metrics['lambda']:.3f})")
    log.info(f"  Q (modularity)  = {global_metrics['modularity_Q']:.3f}")
    log.info(f"  Eglob           = {global_metrics['global_efficiency']:.3f}")

    # --- Node metrics ---
    log.info("  Computing node-level metrics...")
    node_metrics = compute_node_metrics(A, ci)
    hub_labels = classify_hubs(
        node_metrics["within_module_z"],
        node_metrics["participation_coef"]
    )
    n_connector = (hub_labels == "connector-hub").sum()
    n_provincial = (hub_labels == "provincial-hub").sum()
    log.info(f"  Hubs: {n_connector} connector, {n_provincial} provincial")

    # --- Hub map plot ---
    plot_hub_map(node_metrics, subject_id, config["output_dir"])

    return subject_id, {
        "matrix_raw": A_raw,
        "matrix_processed": A,
        "global_metrics": global_metrics,
        "node_metrics": node_metrics,
        "community_assignments": ci,
        "rich_club_coef": rc,
        "hub_labels": hub_labels,
    }


def run_pipeline(config: dict):
    log.info("=" * 60)
    log.info("DWI Connectome Analysis Pipeline")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)

    # Find input files
    gpickle_files = find_gpickle_files(config["gpickle_dir"])
    if not gpickle_files:
        log.error(f"No .gpickle files found in: {config['gpickle_dir']}")
        return

    log.info(f"Found {len(gpickle_files)} connectome file(s)")

    all_results = {}

    n_workers = min(len(gpickle_files), os.cpu_count() or 1)
    log.info(f"Processing {len(gpickle_files)} subject(s) in parallel with {n_workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_one_subject, (p, config)): p
            for p in gpickle_files
        }
        for fut in as_completed(futures):
            subject_id, result = fut.result()
            if result is not None:
                all_results[subject_id] = result

    # --- Group-level plot ---
    if len(all_results) > 1:
        all_global = [v["global_metrics"] for v in all_results.values()]
        subj_ids = list(all_results.keys())
        plot_group_metrics(all_global, subj_ids, config["output_dir"])

    # --- Save ---
    save_results(all_results, config["output_dir"])

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info(f"Results saved to: {config['output_dir']}")
    log.info("=" * 60)

    return all_results


# ─────────────────────────────────────────────
# OPTIONAL: GROUP COMPARISON EXAMPLE
# ─────────────────────────────────────────────

def example_group_comparison(all_results: dict, group1_ids: list, group2_ids: list):
    """
    Example: compare global metrics between two groups.
    Replace group1_ids / group2_ids with your actual subject ID lists.
    """
    g1_metrics = [all_results[s]["global_metrics"] for s in group1_ids if s in all_results]
    g2_metrics = [all_results[s]["global_metrics"] for s in group2_ids if s in all_results]

    metric_names = ["clustering_coef", "char_path_length", "global_efficiency",
                    "local_efficiency", "modularity_Q", "sigma"]

    print("\n── Group Comparison (Permutation Test) ──")
    results_list = []
    p_values = []

    for metric in metric_names:
        res = compare_groups(g1_metrics, g2_metrics, metric, n_permutations=5000)
        results_list.append(res)
        p_values.append(res["p_value"])

    # FDR correction
    reject, _ = fdr_correct(np.array(p_values))

    for res, rej in zip(results_list, reject):
        sig = "* FDR-sig" if rej else ""
        print(
            f"  {res['metric']:25s}  "
            f"G1={res['group1_mean']:.4f}±{res['group1_std']:.4f}  "
            f"G2={res['group2_mean']:.4f}±{res['group2_std']:.4f}  "
            f"p={res['p_value']:.4f}  d={res['cohens_d']:.3f}  {sig}"
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = parse_args()
    results = run_pipeline(config)

    # Uncomment and fill in subject IDs to run group comparison:
    # example_group_comparison(
    #     results,
    #     group1_ids=["sub-01", "sub-02"],
    #     group2_ids=["sub-03", "sub-04"],
    # )