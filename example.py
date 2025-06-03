"""
An example of how the Hino detection method can be used under real-life
conditions.

Written by Jessy Colonval.
"""
import time
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from src.hino import Hino

if __name__ == "__main__":
    # Loads the Iris dataset and converts it into a DataFrame.
    iris = datasets.load_iris()
    df = DataFrame(data=np.c_[iris["data"], iris["target"]],
                   columns=iris["feature_names"] + ["target"])

    # Initializes a Hino's model.
    model = Hino(df, iris["feature_names"], "target")

    # Runs the outliers detection with the default configuration.
    start = time.time()
    is_outliers = model.fit()
    end = time.time() - start

    n_outliers = sum(is_outliers)
    print("Default configuration:")
    print(f"\t{n_outliers:d} outliers found in {end:.5f} seconds!")
    print(f"\tWith {model.n_quantiles:d} quantiles and a tolerance limit",
          f"at {model.limit:d}.")

    # Runs the outliers detection with a custom configuration.
    start = time.time()
    model.set_n_quantiles(20)
    model.set_limit(1)
    is_outliers = model.fit()
    end = time.time() - start

    n_outliers = sum(is_outliers)
    print("\nCustom configuration:")
    print(f"\t{n_outliers:d} outliers found in {end:.5f} seconds!")
    print(f"\tWith {model.n_quantiles:d} quantiles and a tolerance limit",
          f"at {model.limit:d}.")
