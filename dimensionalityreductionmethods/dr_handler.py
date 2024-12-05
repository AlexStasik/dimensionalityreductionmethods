import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import sys, random, warnings, time, pprint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding
import umap

from .dr_methods import (
    run_pca,
    run_isomap,
    run_tsne,
    run_umap,
    run_autoencoder,
    run_kpca,
    run_lle,
    get_autoencoder_embedding,
)


class DimensionalityReductionHandler:
    """
    This class applies various dimensionality reduction methods to the initial dataset and provides numerical and graphical representations of below key metrics:

    - Reconstruction error: the difference between the original data and its reconstruction after dimensionality reduction.
    - Trustworthiness: How well local relationships are preserved when reducing the data to lower dimensions.
    - Total time: The time taken to run each method.

    These metrics help assess the performance of each method and determine the intrinsic dimensionality of the data.

    Attributes:
        data (numpy array): the dataset on which dimensionality reduction methods will be applied.
        results: a tabulate containing the results of each method, including reconstruction error, trustworthiness and the total running time of each method.
        methods (list of strings): A list of methods to be used for dimensionality reduction.
    """

    def __init__(self, data):
        """
        Initializes the class with the provided data.

        Parameters:
            data (numpy array): The dataset to perform dimensionality reduction on.
        """
        self.data = data
        self.results = None

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

    def analyze_dimensionality_reduction(
        self, methods, autoencoder_max_dim=sys.maxsize
    ):
        """
        Performs dimensionality reduction techniques on the initial dataset and computes performance metrics.

        Supported methods: PCA, KPCA, LLE, ISOMAP, UMAP, TSNE, AUTOENCODER.

        Parameters:
            methods (list of str): Dimensionality reduction methods to apply.
            autoencoder_max_dim (int, optional): maximum dimension for Autoencoder to reduce computational stress --> defaults to sys.maxsize.
        """
        self.methods = [method.strip().lower() for method in methods]
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
            delayed(method_funcs[method])(self.scaled_data) for method in valid_methods
        )

        results = dict(zip(self.methods, results_list))
        self.results = results
        # pprint.pprint(results)

    def plot_results(self):
        """
        Visualizes the results of the dimensionality reduction analysis.

        Generates a comprehensive plot showcasing reconstruction error and trustworthiness across the components for each applied dimensionality
        reduction method. A separate zoomed-in plot highlights trustworthiness metrics in detail.

        Requirements:
            - The `analyze_dimensionality_reduction` method must be called prior to use.

        Raises:
            UserWarning: If `analyze_dimensionality_reduction` has not been called.
        """
        if self.results == None:
            warnings.warn(
                "Please call the `analyze_dimensionality_reduction` method first before calling the `plot_results` method.",
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

        This method summarizes key metrics for each dimensionality reduction technique, including:
        - Optimal trustworthiness component and its value.
        - Optimal reconstruction error component and its value.
        - Computation time for each method.

        Requirements:
            - The `analyze_dimensionality_reduction` method must be called prior to use.

        Returns:
            pandas.DataFrame: A table summarizing the performance metrics of each method.

        Raises:
            UserWarning: If `analyze_dimensionality_reduction` has not been called.
        """
        if self.results == None:
            warnings.warn(
                "Please call the `analyze_dimensionality_reduction` method first before calling the `table` method.",
                category=UserWarning,
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

    def visualization(self, labels=None, plot_in_3d=False):
        """
        Visualize the results of dimensionality reduction in 2D or 3D.

        This method projects the data into 2D or 3D space using selected dimensionality reduction methods and creates scatter plots for visualization.

        Parameters:
            labels (array-like, optional): Labels for data points, used to color the scatter plots. Defaults to None.
            plot_in_3d (bool, optional): If True, visualizes the embeddings in 3D; otherwise, 2D plots are generated. Defaults to False.

        Requirements:
            - The `analyze_dimensionality_reduction` method must be called prior to use.

        Output:
            Displays scatter plots for each dimensionality reduction method, either in 2D or 3D based on the `plot_in_3d` parameter.

        Raises:
            UserWarning: If `analyze_dimensionality_reduction` has not been called.
        """
        if self.results == None:
            warnings.warn(
                "Please call the `analyze_dimensionality_reduction` method first before calling the `visualization` method.",
                category=UserWarning,
            )
            return

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
                    embedding = get_autoencoder_embedding(
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
