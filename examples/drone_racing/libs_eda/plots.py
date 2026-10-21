from matplotlib import pyplot as plt
import numpy as np


def _get_metric(result_log, axis_name):
    """Return (values, std) for a metric, preferring its *_mean field."""
    if axis_name + "_mean" in result_log:
        mean = np.asarray(result_log[axis_name + "_mean"], dtype=float)
    elif axis_name in result_log:
        mean = np.asarray(result_log[axis_name], dtype=float)
    else:
        return None, None

    std = None
    if axis_name + "_std" in result_log:
        std = np.asarray(result_log[axis_name + "_std"], dtype=float)
    return mean, std


def plot_value_line(ax, result_log, axis_name, color="blue", label=None):
    """Plot a metric and its mean +/- standard deviation when available."""
    mean, std = _get_metric(result_log, axis_name)
    if mean is None:
        return ax

    steps = np.asarray(result_log.get("step", np.arange(len(mean))), dtype=float)
    n = min(len(steps), len(mean))
    steps, mean = steps[:n], mean[:n]
    ax.plot(steps, mean, color=color, label=label or axis_name)

    if std is not None:
        std = std[:n]
        valid = np.isfinite(steps) & np.isfinite(mean) & np.isfinite(std)
        ax.fill_between(steps, mean - std, mean + std,
                        where=valid, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("training step")
    ax.set_ylabel(axis_name.replace("_", " "))
    ax.grid(True, alpha=0.25)
    return ax


def save_plot(fig, path):
    """Save a plot at presentation-quality resolution and close its figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
