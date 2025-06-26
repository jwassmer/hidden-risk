# %%
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from pathlib import Path
from scipy.stats import weibull_min
import scipy.stats as stats
import hashlib

PATH_TO_GMP = "data/GMP/mobility"


def read_P(year, path=PATH_TO_GMP):
    yearp1 = str(year + 1)
    year = str(year)
    path = f"{path}/" + year + "-" + yearp1[2:] + "/CSV-Daten"
    P = pd.read_csv(os.path.join(path, f"P{year[2:]}.csv"), sep=";")
    return P


def read_PT(year, path=PATH_TO_GMP):
    yearp1 = str(year + 1)
    year = str(year)
    path = f"{path}/" + year + "-" + yearp1[2:] + "/CSV-Daten"
    PT = pd.read_csv(os.path.join(path, f"PT{year[2:]}.csv"), sep=";")
    HH = pd.read_csv(os.path.join(path, f"HH{year[2:]}.csv"), sep=";")

    HH = HH.set_index("ID")
    PT["SIEDDICHTE"] = [HH.loc[ID]["SIEDDICHTE"] for ID in PT["ID"]]
    return PT


def read_W(year, path=PATH_TO_GMP):
    yearp1 = str(year + 1)
    year = str(year)
    path = f"{path}/" + year + "-" + yearp1[2:] + "/CSV-Daten"
    W = pd.read_csv(os.path.join(path, f"W{year[2:]}.csv"), sep=";")

    return W


def read_HH(year, path=PATH_TO_GMP):
    yearp1 = str(year + 1)
    year = str(year)
    path = f"{path}/" + year + "-" + yearp1[2:] + "/CSV-Daten"
    HH = pd.read_csv(os.path.join(path, f"HH{year[2:]}.csv"), sep=";")
    HH = HH.set_index("ID")
    return HH


def read_PT_kmPKW_files(path, attr):
    data_points = np.array([])
    frames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if ("PT" in file) and (".csv" in file):
                PT = pd.read_csv(os.path.join(subdir, file), sep=";")
                PT_clean = PT[
                    (PT["AnzPKW"] != 0) & (PT["dau_PKW"] > 0) & (PT["km_PKW"] > 0)
                ]
                frames.append(PT_clean)

    PTS = pd.concat(frames)

    if "travel_time" in attr:
        attr = "dau_PKW"
    elif "length" in attr:
        attr = "km_PKW"
    q = PTS[attr].quantile(0.75)
    filtered_PT = PTS[PTS[attr] < q]
    data_points = (filtered_PT[attr] / filtered_PT["AnzPKW"]).to_numpy()
    return data_points


# %%
def exp_func(x, a, k):
    return a * np.exp(-k * x)


def lin_func(x, m, b):
    return m * x + b


def mobility_fit_params(
    path=PATH_TO_GMP, weight="travel_time", bincount=250, cache=True
):
    """
    Compute the fit parameters for the mobility distribution.

    Parameters:
        path (str): Path to the GMP data directory.
        weight (str): Weight attribute to use for the fit ('length' or 'travel_time').
        bincount (int): Number of bins for the histogram.
        cache (bool): Whether to cache the results for future use.

    Returns:
        tuple: Tuple containing the maximum bin value, exponential fit parameters, and linear fit parameters.
    """
    first_compute = False
    if cache:
        if Path(f"{path}/mobility_fit_{weight}_{bincount}").is_dir():
            maxbin = np.load(f"{path}/mobility_fit_{weight}_{bincount}/maxbin.npy")[0]
            popt_lin = np.load(
                f"{path}/mobility_fit_{weight}_{bincount}/lin_coeffs.npy"
            )
            popt_exp = np.load(
                f"{path}/mobility_fit_{weight}_{bincount}/exp_coeffs.npy"
            )
        else:
            first_compute = True
    if not cache or first_compute:
        if "length" in weight:
            multiplier = 1000  # get m from km
        elif "travel_time" in weight:
            multiplier = 60  # get sec from min
        data_points = read_PT_kmPKW_files(path, weight) * multiplier
        data_pkw, bins = np.histogram(data_points, bins=bincount, density=True)
        binscenters = np.array(
            [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
        )

        max_ind = np.argmax(data_pkw)
        popt_exp, pcov_exp = curve_fit(
            exp_func, binscenters[max_ind:], data_pkw[max_ind:], p0=[0.05, 0.001]
        )
        popt_lin, pcov_lin = curve_fit(
            lin_func, binscenters[:max_ind], data_pkw[:max_ind], p0=[0.01, 0.01]
        )
        maxbin = binscenters[max_ind]

        if cache:
            os.mkdir(f"{path}/mobility_fit_{weight}_{bincount}")
            np.save(f"{path}/mobility_fit_{weight}_{bincount}/maxbin.npy", [maxbin])
            np.save(f"{path}/mobility_fit_{weight}_{bincount}/lin_coeffs.npy", popt_lin)
            np.save(f"{path}/mobility_fit_{weight}_{bincount}/exp_coeffs.npy", popt_exp)

    return maxbin, popt_exp, popt_lin


def weibull_pdf(x, c, loc, lambda_):
    """
    Computes the Weibull PDF with location parameter.

    Parameters:
    - x: value for which the PDF is computed
    - c: shape parameter
    - loc: location parameter
    - lambda_: scale parameter

    Returns:
    - value of the PDF at x
    """
    y = x - loc
    if y < 0:
        return 0
    else:
        return (c / lambda_) * (y / lambda_) ** (c - 1) * np.exp(-((y / lambda_) ** c))


def weibull_fit(path=PATH_TO_GMP, weight="travel_time"):
    if "length" in weight:
        multiplier = 1000  # get m from km
    elif "travel_time" in weight:
        multiplier = 60  # get sec from min

    try:
        params = np.load(Path(f"{path}/weibull_fit_{weight}.npy"))
    except:
        data_points = read_PT_kmPKW_files(path, weight) * multiplier
        params = weibull_min.fit(data_points)
        np.save(f"{path}/weibull_fit_{weight}.npy", params)
    return params


def lognorm_pdf(x, mu, sigma):
    if x <= 0:
        return 0
    else:
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(x) - mu) ** 2) / (2 * sigma**2)
        )


def lognorm_fit(path=PATH_TO_GMP, weight="travel_time"):
    if "length" in weight:
        multiplier = 1000  # get m from km
    elif "travel_time" in weight:
        multiplier = 60  # get sec from min

    try:
        params = np.load(Path(f"{path}/lognorm_fit_{weight}.npy"))
        # fitted_mu, fitted_sigma = params
    except:
        data_points = read_PT_kmPKW_files(path, weight) * multiplier
        params = stats.norm.fit(np.log(data_points))
        np.save(f"{path}/lognorm_fit_{weight}.npy", params)
    return params
    # return lambda x: stats.lognorm.pdf(x, fitted_sigma, scale=np.exp(fitted_mu))


def mobility_func(xs, max_bin, popt_exp, popt_lin):
    """
    Calculate the mobility function, a linear increased until maximum, follwed by exponential decrease, for the given input values.

    Parameters:
        xs (float or list): Input value(s) for which to calculate the mobility function.
        max_bin (float): Maximum bin value used in the fit.
        popt_exp (array-like): Exponential fit parameters.
        popt_lin (array-like): Linear fit parameters.

    Returns:
        float or list: Calculated mobility function value(s) corresponding to the input value(s).
    """
    if hasattr(xs, "__len__"):
        return [
            exp_func(x, *popt_exp) if x > max_bin else lin_func(x, *popt_lin)
            for x in xs
        ]
    elif xs > max_bin:
        return exp_func(xs, *popt_exp)
    else:
        return lin_func(xs, *popt_lin)


# %%
