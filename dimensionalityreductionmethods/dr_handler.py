import sys, time, random, warnings, pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding
import umap

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from .dr_methods import (
    run_pca,
    run_isomap,
    run_tsne,
    run_umap,
    run_autoencoder,
    run_kpca,
    run_lle,
    apply_autoencoder,
)


class DimensionalityReductionHandler:
    """
    Given data and a list of methods, this class provides numerical and graphical representations of reconstruction error (difference between original data and
    its reconstruction after dimensionality reduction) and trustworthiness (how well local relationships are preserved in low dimensions vs high dimensions) to
    determine the intrinsic dimensionality of the data.

    Attributes:
        data: a numpy array represending the data to perform dimensionality reduction on
        results: a dicionary of the methods and their associated reconstruction error and trustworthiness
        methods: the methods to use to perform dimensionality reduction
    """

    def __init__(self, data):
        """
        Initializes the class with the provided data. Also initializes the results to be None

        Parameters:
            data: a numpy array representing the data to perform dimensionality reduction on TODO: numpy array?
        """
        self.data = data
        self.results = None

    def analyze_dimensionality_reduction(
        self, methods, autoencoder_max_dim=sys.maxsize
    ):
        """
        Performs dimensionality reduction using the provided methods on the data from the initialization. The results are printed out.

        Supported methods: PCA, KPCA, Isomap, UMAP, TSNE, Autoencoder.

        Parameters:
            methods (list of str): Dimensionality reduction methods to apply.
            autoencoder_max_dim (int, optional): Maximum dimension for Autoencoder to reduce computational stress. Defaults to sys.maxsize.
        """
        self.methods = [method.strip().lower() for method in methods]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        results = {}

        method_funcs = {
            "pca": run_pca,
            "isomap": run_isomap,
            "tsne": run_tsne,
            "umap": run_umap,
            "autoencoder": lambda data: run_autoencoder(data, autoencoder_max_dim),
            "kpca": run_kpca,
            "lle": run_lle,
        }

        valid_methods = [method for method in self.methods if method in method_funcs]
        invalid_methods = [
            method for method in self.methods if method not in method_funcs
        ]

        if invalid_methods:
            warnings.warn(
                f"The following methods are not recognized and will be ignored: {', '.join(invalid_methods)}",
                category=UserWarning,
            )

        self.methods = valid_methods

        results_list = Parallel(n_jobs=-1, timeout=7200)(
            delayed(method_funcs[method])(scaled_data) for method in valid_methods
        )

        results = dict(zip(self.methods, results_list))
        self.results = results
        # pprint.pprint(results)

    def plot_results(self):
        """
        Plots the results from the analyze_dimensionality_reduction method on a single figure.
        The `analyze_dimensionality_reduction` method must be called before using this function.
        """
        if self.results == None:
            warnings.warn(
                "Please call the analyze_dimensionality_reduction method first before calling this one.",
                category=UserWarning,
            )
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel("Components")
        ax1.set_ylabel("Reconstruction Error (%)")
        ax1.set_ylim(-0.5, 101.0)
        colors = {}
        trustworthiness_min = 100

        for method in self.methods:
            if method in self.results:
                color = f"#{random.randint(0, 0xFFFFFF):06x}"
                colors[method] = color

                method_data = self.results[method]
                components = method_data["components"]
                reconstruction_error = method_data["reconstruction_error"]

                if reconstruction_error is not None:
                    if len(reconstruction_error) == 1:
                        reconstruction_error = [reconstruction_error[0]] * len(
                            components
                        )

                    ax1.plot(
                        components,
                        reconstruction_error,
                        marker="o",
                        color=color,
                        label=f"{method} R_E",
                    )

        ax2 = ax1.twinx()
        ax2.set_ylabel("Trustworthiness (%)", labelpad=15)
        ax2.set_ylim(-0.5, 101.0)

        for method in self.methods:
            if method in self.results:
                method_data = self.results[method]
                components = method_data["components"]
                trustworthiness = method_data["trustworthiness"]

                if trustworthiness is not None:
                    if len(trustworthiness) == 1:
                        trustworthiness = [trustworthiness[0]] * len(components)

                    trustworthiness_min = min(trustworthiness_min, min(trustworthiness))

                    ax2.plot(
                        components,
                        trustworthiness,
                        marker="x",
                        linestyle="--",
                        color=colors[method],
                        label=f"{method} T",
                    )

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + lines2,
            labels + labels2,
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            title="Metrics",
        )

        plt.title("Reconstruction Error & Trustworthiness for Selected Methods")
        plt.tight_layout()
        plt.show()

        # Separate zoomed-in plot
        fig_zoom, ax_zoom = plt.subplots(figsize=(10, 6))
        ax_zoom.set_title("Close Up Trustworthiness")
        ax_zoom.set_xlabel("Components")
        ax_zoom.set_ylabel("Trust (%)")

        for method in self.methods:
            if method in self.results:
                method_data = self.results[method]
                components = method_data["components"]
                trustworthiness = method_data["trustworthiness"]

                if trustworthiness is not None:
                    if len(trustworthiness) == 1:
                        trustworthiness = [trustworthiness[0]] * len(components)

                    ax_zoom.plot(
                        components,
                        trustworthiness,
                        marker="x",
                        linestyle="--",
                        color=colors[method],
                        label=f"{method} T",
                    )

        ax_zoom.legend(
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            title="Methods",
        )
        plt.tight_layout()
        plt.show()

    def table(self):
        """
        Generate a summary table of results from dimensionality reduction methods.
        Includes optimal trustworthiness and reconstruction error components, along with their corresponding values and computation time.
        """
        if self.results == None:
            print(
                "WARNING: Please call the analyze_dimensionality_reduction method first before calling this one."
            )
            return

        results_summary = []

        for method in self.methods:
            if method in self.results:
                method_data = self.results[method]
                components = method_data["components"]
                reconstruction_error = method_data["reconstruction_error"]
                trustworthiness = method_data["trustworthiness"]
                time = method_data["time"]

                if trustworthiness is not None and any(
                    t is not None for t in trustworthiness
                ):
                    max_trust = max(trustworthiness)
                    place_trust = (
                        trustworthiness.index(max_trust)
                        if isinstance(trustworthiness, list)
                        else np.argmax(trustworthiness)
                    )
                    trust_opt_component = components[place_trust]
                else:
                    max_trust, trust_opt_component = "-", "-"

                if reconstruction_error is not None and any(
                    e is not None for e in reconstruction_error
                ):
                    min_error = min(reconstruction_error)
                    place_error = (
                        reconstruction_error.index(min_error)
                        if isinstance(reconstruction_error, list)
                        else np.argmin(reconstruction_error)
                    )
                    error_opt_component = components[place_error]
                else:
                    min_error, error_opt_component = "-", "-"

                results_summary.append(
                    [
                        method,
                        trust_opt_component,
                        max_trust,
                        error_opt_component,
                        min_error,
                        time,
                    ]
                )

        df = pd.DataFrame(
            results_summary,
            columns=[
                "Method",
                "Opt. Trustworthiness Components",
                "Max Trustworthiness",
                "Opt. Error Components",
                "Min R. Error",
                "time",
            ],
        )
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        return df

    def visualization(self, labels=None):
        """
        Visualize the results of dimensionality reduction methods in 2D.

        Parameters:
            labels (array-like, optional): Labels for coloring the scatter plots.

        Each selected method will reduce the data to 2D and be plotted on a grid.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        plot_idx = 0

        for method in self.methods:
            if method not in self.results:
                continue
            embedding_2d = None

            if method == "tsne":
                tsne = TSNE(n_components=2, perplexity=4, random_state=42)
                embedding_2d = tsne.fit_transform(self.data)
            elif method == "isomap":
                isomap = Isomap(n_components=2)
                embedding_2d = isomap.fit_transform(self.data)
            elif method == "umap":
                reducer = umap.UMAP(n_components=2)
                embedding_2d = reducer.fit_transform(self.data)
            elif method == "autoencoder":
                embedding_2d = apply_autoencoder(
                    data=self.data, n_components=2, hidden_layer_neurons=4
                )
            elif method == "pca":
                pca = PCA(n_components=2)
                embedding_2d = pca.fit_transform(self.data)
            elif method == "kpca":
                kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
                embedding_2d = kpca.fit_transform(self.data)
            elif method == "lle":
                lle = LocallyLinearEmbedding(n_components=2, n_neighbors=3)
                embedding_2d = lle.fit_transform(self.data)

            if embedding_2d is not None:
                ax = axes[plot_idx]

                if labels is not None:
                    scatter = ax.scatter(
                        embedding_2d[:, 0],
                        embedding_2d[:, 1],
                        c=labels,
                        cmap="plasma",
                        alpha=0.4,
                    )
                    fig.colorbar(scatter, ax=ax, label="Labels")
                else:
                    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.4)

                ax.set_title(f"{method}")
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def visualization_3d(self, labels=None, plot_in_3d=False):
        """
        Visualizes the results of dimensionality reduction in 3D.

        Parameters:
            labels (array-like, optional): Class labels for coloring points. Defaults to None.
            plot_in_3d (bool, optional): ... Defaults to False.
        """
        n_components = 3 if plot_in_3d else 2
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(20, 10),
            subplot_kw={"projection": "3d"} if plot_in_3d else {},
        )
        axes = axes.flatten()

        plot_idx = 0
        for method in self.methods:
            if method in self.results:
                embedding = None

                if method == "tsne":
                    tsne = TSNE(
                        n_components=n_components, perplexity=4, random_state=42
                    )
                    embedding = tsne.fit_transform(self.data)
                elif method == "isomap":
                    isomap = Isomap(n_components=n_components)
                    embedding = isomap.fit_transform(self.data)
                elif method == "umap":
                    reducer = umap.UMAP(n_components=n_components)
                    embedding = reducer.fit_transform(self.data)
                elif method == "autoencoder":
                    embedding = apply_autoencoder(
                        data=self.data,
                        n_components=n_components,
                        hidden_layer_neurons=6,
                    )
                elif method == "pca":
                    pca = PCA(n_components=n_components)
                    embedding = pca.fit_transform(self.data)
                elif method == "kpca":
                    kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=0.1)
                    embedding = kpca.fit_transform(self.data)
                elif method == "lle":
                    lle = LocallyLinearEmbedding(
                        n_components=n_components, n_neighbors=3
                    )
                    embedding = lle.fit_transform(self.data)

                if embedding is None:
                    continue

                ax = axes[plot_idx]
                if plot_in_3d:
                    if labels is not None:
                        scatter = ax.scatter(
                            embedding[:, 0],
                            embedding[:, 1],
                            embedding[:, 2],
                            c=labels,
                            cmap="plasma",
                            alpha=0.4,
                        )
                        fig.colorbar(scatter, ax=ax, label="Labels")
                    else:
                        ax.scatter(
                            embedding[:, 0],
                            embedding[:, 1],
                            embedding[:, 2],
                            alpha=0.4,
                        )
                else:
                    if labels is not None:
                        scatter = ax.scatter(
                            embedding[:, 0],
                            embedding[:, 1],
                            c=labels,
                            cmap="plasma",
                            alpha=0.4,
                        )
                        fig.colorbar(scatter, ax=ax, label="Labels")
                    else:
                        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)

                ax.set_title(f"{method}")
                plot_idx += 1

        plt.tight_layout()
        plt.show()
