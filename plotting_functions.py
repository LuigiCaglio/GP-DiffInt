import numpy as np
import matplotlib.pyplot as plt



def plot_states_with_zoom(
    t,
    states,                 # shape (N, >=1): true displacement, optionally more
    derivatives,            # shape (N, 2): [dx/dt, d2x/dt2]
    displacement_GP,        # shape (N): smoothed displacement
    velocity_GP,            # shape (N): smoothed velocity
    acceleration_GP,        # shape (N): smoothed acceleration
    displacement_variance,  # shape (N,)
    velocity_variance,      # shape (N,)
    acceleration_variance,  # shape (N,) — only used if n_panels=3
    t_full_min=0.0,
    t_full_max=None,
    t_zoom_min=None,
    t_zoom_max=None,
    fontsize_axes=16,
    fontsize_legends=12,
    save_path=None,
    figsize=(12, 10),
    n_panels=3,           
):
    """
    Plot full-range states with optional zoom, with either:
       - 3 panels: x(t), dx/dt, d²x/dt²
       - 2 panels: x(t), dx/dt
    """

    assert n_panels in (2, 3), "n_panels must be 2 or 3"

    if t_full_max is None:
        t_full_max = float(t[-1])

    has_zoom = (t_zoom_min is not None) and (t_zoom_max is not None)
    if has_zoom:
        zoom_mask = (t >= t_zoom_min) & (t <= t_zoom_max)
        t_zoom = t[zoom_mask]

    # Uncertainty bands
    unc0 = 2 * np.sqrt(displacement_variance)
    unc1 = 2 * np.sqrt(velocity_variance)
    if n_panels == 3:
        unc2 = 2 * np.sqrt(acceleration_variance)

    fig = plt.figure(figsize=figsize)

    # Grid: number of rows = n_panels
    gs_main = fig.add_gridspec(
        n_panels, 2,
        width_ratios=[1.5, 1],
        hspace=0.15, wspace=0.1,
        left=0.08, right=0.98, top=0.98, bottom=0.08
    )

    # Left column
    axs = [fig.add_subplot(gs_main[i, 0]) for i in range(n_panels)]
    # Right column zoom
    axs_zoom = [fig.add_subplot(gs_main[i, 1]) for i in range(n_panels)] if has_zoom else None

    # -----------------------------  
    # Panel 1: displacement
    # -----------------------------
    axs[0].plot(t, states[:, 0], "k", lw=2, label="True displacement/state")
    axs[0].plot(t, displacement_GP, "--", lw=1.5, color="tab:blue", label="Estimated")
    axs[0].fill_between(t, displacement_GP-unc0, displacement_GP+unc0, alpha=0.2, color="tab:blue")

    if has_zoom:
        axs[0].axvspan(t_zoom_min, t_zoom_max, alpha=0.15, color="gray")
        axs_zoom[0].plot(t_zoom, states[zoom_mask, 0], "k", lw=2)
        axs_zoom[0].plot(t_zoom,displacement_GP [zoom_mask], "--", lw=1.5, color="tab:blue")
        axs_zoom[0].fill_between(t_zoom,
                                 displacement_GP[zoom_mask]-unc0[zoom_mask],
                                 displacement_GP[zoom_mask]+unc0[zoom_mask],
                                 alpha=0.2, color="tab:blue")

    axs[0].set_ylabel("x(t)", fontsize=fontsize_axes)

    # -----------------------------
    # Panel 2: velocity
    # -----------------------------
    axs[1].plot(t, derivatives[:, 0], "k", lw=2, label="True velocity/1st derivative")
    axs[1].plot(t, velocity_GP, "--", lw=1.5, color="tab:orange", label="Estimated")
    axs[1].fill_between(t, velocity_GP-unc1, velocity_GP+unc1, alpha=0.2, color="tab:orange")

    if has_zoom:
        axs[1].axvspan(t_zoom_min, t_zoom_max, alpha=0.15, color="gray")
        axs_zoom[1].plot(t_zoom, derivatives[zoom_mask, 0], "k", lw=2)
        axs_zoom[1].plot(t_zoom, velocity_GP[zoom_mask], "--", lw=1.5, color="tab:orange")
        axs_zoom[1].fill_between(t_zoom,
                                 velocity_GP[zoom_mask]-unc1[zoom_mask],
                                 velocity_GP[zoom_mask]+unc1[zoom_mask],
                                 alpha=0.2, color="tab:orange")

    axs[1].set_ylabel("dx/dt", fontsize=fontsize_axes)

    # -----------------------------
    # Panel 3: acceleration (ONLY if n_panels = 3)
    # -----------------------------
    if n_panels == 3:
        axs[2].plot(t, derivatives[:, 1], "k", lw=2, label="True acceleration/2nd derivative")
        axs[2].plot(t, acceleration_GP, "--", lw=1.5, color="tab:green", label="Estimated")
        axs[2].fill_between(t, acceleration_GP-unc2, acceleration_GP+unc2, alpha=0.2, color="tab:green")

        if has_zoom:
            axs[2].axvspan(t_zoom_min, t_zoom_max, alpha=0.15, color="gray")
            axs_zoom[2].plot(t_zoom, derivatives[zoom_mask, 1], "k", lw=2)
            axs_zoom[2].plot(t_zoom, acceleration_GP[zoom_mask], "--", lw=1.5, color="tab:green")
            axs_zoom[2].fill_between(t_zoom,
                                     acceleration_GP[zoom_mask]-unc2[zoom_mask],
                                     acceleration_GP[zoom_mask]+unc2[zoom_mask],
                                     alpha=0.2, color="tab:green")

        axs[2].set_ylabel("d²x/dt²", fontsize=fontsize_axes)
        axs[2].set_xlabel("Time [s]", fontsize=fontsize_axes)

    else:
        # 2‑panel version: put xlabel on row 1 (velocity)
        axs[1].set_xlabel("Time [s]", fontsize=fontsize_axes)

    if has_zoom:
        axs_zoom[-1].set_xlabel("Time [s]", fontsize=fontsize_axes)

    # X limits
    for ax in axs:
        ax.set_xlim(t_full_min, t_full_max)
    if has_zoom:
        for ax in axs_zoom:
            ax.set_xlim(t_zoom_min, t_zoom_max)

    # Legends + grid
    for ax in axs:
        ax.legend(loc="best", fontsize=fontsize_legends)
        ax.grid(True, ls="--", alpha=0.4)

    if has_zoom:
        for ax in axs_zoom:
            ax.legend(loc="best", fontsize=fontsize_legends)
            ax.grid(True, ls="--", alpha=0.4)

    # Remove xticks from upper rows
    for i in range(n_panels - 1):
        axs[i].set_xticklabels([])
        if has_zoom:
            axs_zoom[i].set_xticklabels([])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axs, (axs_zoom if has_zoom else None)

 

def plot_derivative_comparison(
    t,
    derivatives,       # shape (N, 2): [dx/dt, d2x/dt2]
    zs,                # shape (3, N)
    fd_data,           # [x_fd, xd_fd, xdd_fd]
    x_sg, xd_sg, xdd_sg,
    xd_tik, xdd_tik,
    t_min=50, t_max=70,
    figsize=(14, 8)
):
    """
    Compare derivative estimation methods (GP, FFT, SG, Tikhonov)
    using ONLY dx/dt and d2x/dt2.
    """

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle("Derivative Comparison (GP vs Classical Methods)", fontsize=18, y=0.98)

    methods = {
        "GP":       [zs[1, :], zs[2, :], ["tab:blue", "--"]],
        "FFT":      [fd_data[1], fd_data[2], ["tab:red", ":"]],
        "SG":       [xd_sg, xdd_sg, ["purple", "-."]],
        "Tikhonov": [xd_tik, xdd_tik, ["gray", "-."]],
    }

    true_vel = derivatives[:, 0]
    true_acc = derivatives[:, 1]

    # ---- Build zoom mask ----
    zoom_mask = (t >= t_min) & (t <= t_max)

    # Utility: compute y-limits inside zoom region
    def get_ylim(true_data, methods_dict, idx):
        """
        idx = 0 for velocity, idx = 1 for acceleration
        """
        vals = [true_data[zoom_mask]]  # include true data
        for vel, acc, _ in methods_dict.values():
            vals.append((vel if idx == 0 else acc)[zoom_mask])
        arr = np.concatenate(vals)
        pad = 0.1 * (arr.max() - arr.min() + 1e-12)
        return arr.min() - pad, arr.max() + pad

    # Compute y-limits for velocity and acceleration
    ylo_vel, yhi_vel = get_ylim(true_vel, methods, idx=0)
    ylo_acc, yhi_acc = get_ylim(true_acc, methods, idx=1)

    # -------- Velocity plot --------
    axs[0].plot(t, true_vel, 'k', lw=1.5, label="True velocity")
    for name, (vel, acc, style) in methods.items():
        axs[0].plot(t, vel, color=style[0], ls=style[1], lw=1.2, label=name)

    axs[0].set_ylabel(r"$\dot{x}(t)$", fontsize=14)
    axs[0].set_xlim(t_min, t_max)
    axs[0].set_ylim(ylo_vel, yhi_vel)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, ls="--", alpha=0.4)

    # -------- Acceleration plot --------
    axs[1].plot(t, true_acc, 'k', lw=1.5, label="True acceleration")
    for name, (vel, acc, style) in methods.items():
        axs[1].plot(t, acc, color=style[0], ls=style[1], lw=1.2, label=name)

    axs[1].set_ylabel(r"$\ddot{x}(t)$", fontsize=14)
    axs[1].set_xlabel("Time (s)", fontsize=14)
    axs[1].set_xlim(t_min, t_max)
    axs[1].set_ylim(ylo_acc, yhi_acc)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout() 
    plt.show()





def plot_derivative_errors(
    t,
    derivatives,     # shape (N, 2)
    zs,              # shape (3, N)
    fd_data,         # [x_fd, xd_fd, xdd_fd]
    x_sg, xd_sg, xdd_sg,
    xd_tik, xdd_tik,
    t_min=50, t_max=70,
    figsize=(14, 8)
):
    """
    Plot error over time for derivative estimation methods:
    GP, FFT, SG, Tikhonov.
    Only dx/dt and d2x/dt2 are shown.
    """

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle("Derivative Estimation Squared Errors Over Time", fontsize=18, y=0.98)

    # Methods: (velocity, acceleration, color)
    methods = {
        "GP":       [zs[1, :], zs[2, :], "tab:blue"],
        "FFT":      [fd_data[1], fd_data[2], "tab:red"],
        "SG":       [xd_sg, xdd_sg, "purple"],
        "Tikhonov": [xd_tik, xdd_tik, "gray"],
    }

    true_vel = derivatives[:, 0]
    true_acc = derivatives[:, 1]

    # ---- Build zoom mask ----
    zoom_mask = (t >= t_min) & (t <= t_max)

    # Utility: automatic y-limits inside zoom region
    def get_ylim_error(true_data, methods_dict, idx):
        vals = []
        for vel, acc, color in methods_dict.values():
            err = (vel - true_data) if idx == 0 else (acc - true_data)
            vals.append(err[zoom_mask])
        arr = np.concatenate(vals)
        pad = 0.1 * (arr.max() - arr.min() + 1e-12)
        return arr.min() - pad, arr.max() + pad

    # Compute y-limits for velocity and acceleration error
    ylo_vel, yhi_vel = get_ylim_error(true_vel, methods, idx=0)
    ylo_acc, yhi_acc = get_ylim_error(true_acc, methods, idx=1)

    # -------- Velocity error --------
    for name, (vel, acc, color) in methods.items():
        axs[0].semilogy(t, (vel - true_vel)**2, lw=1.2, color=color, label=name)

    axs[0].set_ylabel(r"$e^2_{\dot{x}}(t)$", fontsize=14)
    axs[0].set_xlim(t_min, t_max)
    axs[0].set_ylim(ylo_vel, yhi_vel)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, ls="--", alpha=0.4)

    # -------- Acceleration error --------
    for name, (vel, acc, color) in methods.items():
        axs[1].semilogy(t, (acc - true_acc)**2, lw=1.2, color=color, label=name)

    axs[1].set_ylabel(r"$e^2_{\ddot{x}}(t)$", fontsize=14)
    axs[1].set_xlabel("Time (s)", fontsize=14)
    axs[1].set_xlim(t_min, t_max)
    axs[1].set_ylim(ylo_acc, yhi_acc)
    axs[1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()






def plot_integration_method_comparison(
    t,
    u,
    displacement_GP,
    displacement_detrended,
    displacement_fourier,
    u_mean=0.0,
    t_full_min=0.0,
    t_full_max=None,
    t_zoom_min=240.0,
    t_zoom_max=310.0,
    y_zoom_lim=(-0.2, 0.2),
    fontsize_axes=13,
    fontsize_legends=11,
    figsize=(12, 11),
    save_path=None,
):
    """
    Compare displacement estimates from three integration methods against the
    true displacement signal, with a full-range view, a zoomed view, and a
    log-scale squared-error panel.

    Layout (top to bottom as displayed):
        Row 0 – Zoomed view
        Row 1 – Full time range
        Row 2 – Squared error (log scale)

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time vector.
    u : ndarray, shape (N,)
        True displacement signal.
    displacement_GP : ndarray, shape (N,)
        GP-KF smoothed position state (z_ks[0, :]).
    displacement_detrended : ndarray, shape (N,)
        Displacement from double integration + detrending (mean-subtracted).
    displacement_fourier : ndarray, shape (N,)
        Displacement from frequency-domain integration + highpass (mean-subtracted).
    u_mean : float
        Mean of the true displacement, added back to the mean-subtracted estimates.
    t_full_min : float
        Left limit of the full-range x-axis.
    t_full_max : float or None
        Right limit of the full-range x-axis. Defaults to t[-1].
    t_zoom_min : float
        Left limit of the zoomed x-axis.
    t_zoom_max : float
        Right limit of the zoomed x-axis.
    y_zoom_lim : tuple of (float, float) or None
        y-axis limits for the zoomed panel. Pass None to use auto-scaling.
    fontsize_axes : int
    fontsize_legends : int
    figsize : tuple
    save_path : str or None
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : Figure
    axs : tuple of Axes
        (ax_full, ax_zoom, ax_error) — in logical order, matching the
        parameter names above regardless of the row ordering.
    """
    if t_full_max is None:
        t_full_max = float(t[-1])

    # Internal ordering: [zoom, full, error] → displayed rows 0, 1, 2
    fig, axs_rows = plt.subplots(3, 1, figsize=figsize, sharex=False)
    fig.subplots_adjust(hspace=0.15)

    ax_zoom, ax_full, ax_error = axs_rows  # display rows
    axs = (ax_full, ax_zoom, ax_error)     # logical handles returned to caller

    detrend_label   = "Double int. + detrending"
    fourier_label   = "Freq. domain int. + highpass"
    gp_label        = "GP-KF"

    detrend_col  = "tab:green"
    fourier_col  = "tab:orange"
    gp_col       = "tab:blue"
    true_col     = "k"

    # ------------------------------------------------------------------
    # Full-range panel
    # ------------------------------------------------------------------
    ax_full.plot(t, u,
                 label="True", color=true_col, linewidth=2.5)
    ax_full.plot(t, displacement_detrended + u_mean,
                 label=detrend_label, color=detrend_col, linewidth=2)
    ax_full.plot(t, displacement_fourier + u_mean,
                 label=fourier_label, color=fourier_col, linewidth=2)
    ax_full.plot(t, displacement_GP + u_mean,
                 "--", label=gp_label, color=gp_col, linewidth=2)
    ax_full.axvspan(t_zoom_min, t_zoom_max, alpha=0.15, color="gray", zorder=0)
    ax_full.set_xlim(t_full_min, t_full_max)
    ax_full.set_ylabel("Displacement [m]", fontsize=fontsize_axes)
    ax_full.set_xlabel("Time [s]", fontsize=fontsize_axes)
    ax_full.set_title("Full time range", fontsize=14, loc="left")
    ax_full.legend(frameon=True, loc="lower left", fancybox=True,
                   shadow=True, fontsize=fontsize_legends)

    # ------------------------------------------------------------------
    # Zoomed panel
    # ------------------------------------------------------------------
    ax_zoom.plot(t, u,
                 label="True", color=true_col, linewidth=2.5)
    ax_zoom.plot(t, displacement_fourier + u_mean,
                 label=fourier_label, color=fourier_col, linewidth=2)
    ax_zoom.plot(t, displacement_detrended + u_mean,
                 label=detrend_label, color=detrend_col, linewidth=2)
    ax_zoom.plot(t, displacement_GP + u_mean,
                 "--", label=gp_label, color=gp_col, linewidth=2)
    ax_zoom.set_xlim(t_zoom_min, t_zoom_max)
    if y_zoom_lim is not None:
        ax_zoom.set_ylim(*y_zoom_lim)
    ax_zoom.set_ylabel("Displacement [m]", fontsize=fontsize_axes)
    ax_zoom.set_xlabel("Time [s]", fontsize=fontsize_axes)
    ax_zoom.set_title("Zoomed view", fontsize=14, loc="left")
    ax_zoom.legend(frameon=True, loc="best", fancybox=True,
                   shadow=True, fontsize=fontsize_legends)

    # ------------------------------------------------------------------
    # Squared-error panel (log scale)
    # ------------------------------------------------------------------
    ax_error.semilogy(t, (displacement_fourier + u_mean - u) ** 2,
                      label=fourier_label, color=fourier_col, linewidth=2)
    ax_error.semilogy(t, (displacement_detrended + u_mean - u) ** 2,
                      label=detrend_label, color=detrend_col, linewidth=2)
    ax_error.semilogy(t, (displacement_GP + u_mean - u) ** 2,
                      "--", label=gp_label, color=gp_col, linewidth=2)
    ax_error.set_xlim(t_full_min, t_full_max)
    ax_error.set_ylabel("Squared error [m²]", fontsize=fontsize_axes)
    ax_error.set_xlabel("Time [s]", fontsize=fontsize_axes)
    ax_error.set_title("Method comparison (log scale)", fontsize=14, loc="left")
    ax_error.legend(frameon=True, loc="best", fancybox=True,
                    shadow=True, fontsize=fontsize_legends)

    # ------------------------------------------------------------------
    # Shared formatting
    # ------------------------------------------------------------------
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axs















