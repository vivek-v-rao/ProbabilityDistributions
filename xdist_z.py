#!/usr/bin/env python3
# plot standardized PDFs (optional) and print right-tail probabilities at 1..10 sigmas

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, logistic, hypsecant, t

PLOT = True  # <-- set to False to disable plotting

# import matplotlib only when plotting to keep headless runs lighter
def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

# ---------- utilities ----------

def std_pdf(dist, *params):
    """
    Return a function pdf_std(z) for the standardized variable Z = (X - mu)/sigma,
    where X ~ dist(*params, loc=0, scale=1).
    Uses f_Z(z) = sigma * f_X(mu + sigma*z).
    """
    mean, var = dist.stats(*params, moments="mv")
    mu = float(mean)
    sigma = float(math.sqrt(var))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Non-finite or non-positive std for {dist.name} with params {params}: sigma={sigma}")
    def pdf_z(z):
        return sigma * dist.pdf(mu + sigma * z, *params)
    return pdf_z, mu, sigma

def tail_probs_at_k_sigmas(dist, *params, k_max=10):
    """
    Return [ sf(k * sigma) for k=1..k_max ], where sigma is the
    std dev of dist(*params, loc=0, scale=1).
    """
    _, var = dist.stats(*params, moments="mv")
    sigma = float(math.sqrt(var))
    ks = np.arange(1, k_max + 1, dtype=float)
    return [float(dist.sf(k * sigma, *params)) for k in ks]

# ---------- main ----------

def main(plot: bool = PLOT):
    # distributions: (scipy dist, shape-params tuple, label)
    dists = [
        (norm,      tuple(), "normal"),
        (laplace,   tuple(), "laplace"),
        (logistic,  tuple(), "logistic"),
        (hypsecant, tuple(), "hyperbolic secant"),
        (t,         (10,),   "student t (df=10)"),
        (t,         (5,),    "student t (df=5)"),
        (t,         (3,),    "student t (df=3)"),
    ]

    native_meta = []

    # ---- (A) Plot standardized PDFs (optional) ----
    if plot:
        plt = _lazy_import_matplotlib()
        z = np.linspace(-5, 5, 4001)
        pdf_cols = {}
        for dist, params, name in dists:
            pdf_fn, mu_native, sigma_native = std_pdf(dist, *params)
            pdf_cols[name] = pdf_fn(z)
            native_meta.append((name, mu_native, sigma_native))
        df_pdf = pd.DataFrame(pdf_cols, index=z)
        df_pdf.index.name = "z (standardized)"

        plt.figure(figsize=(9, 6))
        for name in df_pdf.columns:
            plt.plot(df_pdf.index, df_pdf[name], label=name, linewidth=1.5)
        plt.xlabel("z (mean 0, std 1)")
        plt.ylabel("density")
        plt.title("Standardized PDFs (mean 0, unit variance)")
        plt.legend(loc="upper right")
        plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        plt.savefig("standardized_pdfs.png", dpi=150)
        plt.show()
    else:
        # still compute native_meta for transparency (no plotting needed)
        for dist, params, name in dists:
            _, mu_native, sigma_native = (lambda p=dist, q=params: (None,)+std_pdf(p,*q)[1:])()
            native_meta.append((name, mu_native, sigma_native))

    # ---- (B) Right-tail probabilities at 1..10 sigmas ----
    k_vals = list(range(1, 11))
    data = {}
    for dist, params, name in dists:
        data[name] = tail_probs_at_k_sigmas(dist, *params, k_max=10)

    df_tail = pd.DataFrame(data, index=k_vals)
    df_tail.index.name = "k (sigmas)"

    # scientific notation with exactly two digits after the decimal
    pd.set_option("display.float_format", "{:.2e}".format)

    print("\nRight-tail probabilities P(X >= mu + k*sigma):")
    print(df_tail.T.to_string())

    # native (canonical) mean and sigma used for standardization
    meta_df = pd.DataFrame(native_meta, columns=["distribution", "native_mean", "native_sigma"])
    pd.set_option("display.float_format", "{:.6f}".format)
    print("\nNative (canonical) mean and sigma used for standardization:")
    print(meta_df.to_string(index=False))

if __name__ == "__main__":
    main()
