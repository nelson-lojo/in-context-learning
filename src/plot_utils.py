
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Dict#, Callable

# from eval import get_model_from_run, get_err_from_run

def plot_model_errs(errors: Dict[str, Tensor], baseline: int, no_std_dev: bool = False):
    for label, err in errors.items():
        loss_means = err.mean(axis=0)
        label_suffix = "" if no_std_dev else "($\mu \; \pm \; 1\sigma$)"

        plt.plot(loss_means, lw=2, label=f"{label} {label_suffix}")

        if not no_std_dev:
            loss_stds  = err.std(axis=0)
            plt.fill_between(list(range(loss_means.shape[0])), loss_means-loss_stds, loss_means+loss_stds, alpha=0.2, linewidth=0, antialiased=True)

    plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()


# def plot_errs_from_run(paths: Dict[str, str], baseline: int, mutate_xs: Callable = (lambda xs: xs), mutate_ys: Callable = (lambda ys: ys), 
#                          mutate_bs: Callable = (lambda bs: bs), no_std_dev: bool=False):
#     """Plot losses for models provided in { "path/to/run" : "plot label" } format"""

#     plot_model_errs(
#         { label : get_err_from_run(path, mutate_xs, mutate_ys, mutate_bs)
#           for path, label in paths.items() },
#         baseline, no_std_dev
#     )


