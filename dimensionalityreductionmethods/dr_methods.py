import numpy as np
from matplotlib import pyplot as plt
import random, umap, time, sys, pprint
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, TSNE, trustworthiness, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import pairwise_distances
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from joblib import Parallel, delayed
import warnings


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

        def run_pca():
            start_pca = time.time()

            pca = PCA()
            reduced_data = pca.fit_transform(scaled_data)
            reconstructed_data = pca.inverse_transform(reduced_data)
            rmse = root_mean_squared_error(scaled_data, reconstructed_data)

            explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
            reconstruction_errors = [
                (1.0 - evr) * 100 for evr in explained_variance_cumsum
            ]

            runtime = time.time() - start_pca

            return {
                "components": np.arange(1, explained_variance_cumsum.shape[0] + 1),
                "trustworthiness": None,
                "reconstruction_error": reconstruction_errors,
                "time": runtime,
            }

        def run_isomap():
            start_isomap = time.time()

            n_components = np.arange(1, scaled_data.shape[1] + 1)
            reconstruction_errors = []
            trustworthiness_scores = []

            for n in n_components:
                isomap_embedding = Isomap(n_components=n)
                transformed_data = isomap_embedding.fit_transform(scaled_data)

                reconstruction_error = isomap_embedding.reconstruction_error()
                reconstruction_errors.append(reconstruction_error)

                trustworthiness_score = trustworthiness(
                    scaled_data, transformed_data, n_neighbors=5
                )
                trustworthiness_scores.append(trustworthiness_score)

            trustworthiness_scores_isomap_percent = [
                score * 100 for score in trustworthiness_scores
            ]
            max_error = max(reconstruction_errors) if reconstruction_errors else 1
            reconstruction_errors_isomap_percent = [
                (value / max_error) * 100 for value in reconstruction_errors
            ]

            runtime = time.time() - start_isomap

            return {
                "components": n_components,
                "trustworthiness": trustworthiness_scores_isomap_percent,
                "reconstruction_error": reconstruction_errors_isomap_percent,
                "time": runtime,
            }

        def run_tsne():
            start_tsne = time.time()

            n_components_range = np.arange(1, scaled_data.shape[1] + 1)
            trustworthiness_scores = []

            for n in n_components_range:
                tsne = TSNE(
                    n_components=n, perplexity=20, random_state=42, method="exact"
                )
                transformed_data = tsne.fit_transform(scaled_data)
                trustworthiness_score = trustworthiness(
                    scaled_data, transformed_data, n_neighbors=5
                )
                trustworthiness_scores.append(trustworthiness_score)

            trustworthiness_scores_tsne_percent = [
                score * 100 for score in trustworthiness_scores
            ]

            runtime = time.time() - start_tsne

            return {
                "components": n_components_range,
                "trustworthiness": trustworthiness_scores_tsne_percent,
                "reconstruction_error": None,
                "time": runtime,
            }

        def run_umap():
            start_umap = time.time()

            n_components = np.arange(1, scaled_data.shape[1] + 1)
            trustworthiness_scores = []

            for n in n_components:
                umap_embedding = umap.UMAP(n_components=n)
                transformed_data = umap_embedding.fit_transform(scaled_data)
                trustworthiness_score = trustworthiness(
                    scaled_data, transformed_data, n_neighbors=5
                )
                trustworthiness_scores.append(trustworthiness_score)

            trustworthiness_scores_umap_percent = [
                score * 100 for score in trustworthiness_scores
            ]

            runtime = time.time() - start_umap

            return {
                "components": n_components,
                "trustworthiness": trustworthiness_scores_umap_percent,
                "reconstruction_error": None,
                "time": runtime,
            }

        def run_autoencoder():
            start_autoencoder = time.time()

            input_dim = scaled_data.shape[1]
            max_dim = min(input_dim, autoencoder_max_dim)
            encoding_dims = np.arange(1, max_dim + 1)

            trustworthiness_scores = []
            reconstruction_errors = []

            for dim in encoding_dims:
                input_layer = Input(shape=(input_dim,))

                hidden1 = Dense(input_dim, activation="relu")(input_layer)
                hidden2 = Dense(input_dim, activation="relu")(hidden1)
                hidden3 = Dense(input_dim, activation="relu")(hidden2)
                hidden4 = Dense(input_dim, activation="relu")(hidden3)

                encoded = Dense(dim, activation="relu")(hidden4)

                hidden5 = Dense(input_dim, activation="relu")(encoded)
                hidden6 = Dense(input_dim, activation="relu")(hidden5)
                hidden7 = Dense(input_dim, activation="relu")(hidden6)
                hidden8 = Dense(input_dim, activation="relu")(hidden7)

                decoded = Dense(input_dim, activation="sigmoid")(hidden8)

                autoencoder = Model(input_layer, decoded)
                encoder = Model(input_layer, encoded)
                autoencoder.compile(optimizer="adam", loss="mse")
                autoencoder.fit(
                    scaled_data,
                    scaled_data,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                )

                encoded_data = encoder.predict(scaled_data)
                decoded_data = autoencoder.predict(scaled_data)

                reconstruction_error = root_mean_squared_error(
                    scaled_data, decoded_data
                )
                reconstruction_errors.append(reconstruction_error)

                trustworthiness_score = trustworthiness(
                    scaled_data, encoded_data, n_neighbors=5
                )
                trustworthiness_scores.append(trustworthiness_score)

            trustworthiness_scores_autoenoder_percent = [
                score * 100 for score in trustworthiness_scores
            ]
            max_error = max(reconstruction_errors) if reconstruction_errors else 1
            reconstruction_errors_autoencoder_percent = [
                (score / max_error) * 100 for score in reconstruction_errors
            ]

            runtime = time.time() - start_autoencoder

            return {
                "components": encoding_dims,
                "trustworthiness": trustworthiness_scores_autoenoder_percent,
                "reconstruction_error": reconstruction_errors_autoencoder_percent,
                "time": runtime,
            }

        def run_kpca():
            start_kpca = time.time()

            reconstruction_errors = []
            n_components = np.arange(1, scaled_data.shape[1] + 1)

            for n in n_components:
                kpca = KernelPCA(
                    kernel="rbf", n_components=n, fit_inverse_transform=True, gamma=0.1
                )
                reduced_data = kpca.fit_transform(scaled_data)
                reconstructed_data = kpca.inverse_transform(reduced_data)

                rmse = root_mean_squared_error(scaled_data, reconstructed_data)
                reconstruction_errors.append(rmse)

            reconstruction_errors_percent = (
                1 - (np.cumsum(reconstruction_errors) / np.sum(reconstruction_errors))
            ) * 100

            runtime = time.time() - start_kpca

            return {
                "components": np.arange(1, reduced_data.shape[1] + 1),
                "trustworthiness": None,
                "reconstruction_error": reconstruction_errors_percent,
                "time": runtime,
            }

        def run_lle():
            start_lle = time.time()

            reconstruction_errors = []
            n_components = np.arange(1, scaled_data.shape[1] + 1)

            for n in n_components:
                lle = LocallyLinearEmbedding(
                    n_components=n, n_neighbors=10, method="standard"
                )
                reduced_data = lle.fit_transform(scaled_data)
                batch_size = 1000
                rmse_sum = 0
                num_batches = 0

                for i in range(0, scaled_data.shape[0], batch_size):
                    end = min(i + batch_size, scaled_data.shape[0])
                    original_distances = pairwise_distances(
                        scaled_data[i:end], scaled_data
                    )
                    reduced_distances = pairwise_distances(
                        reduced_data[i:end], reduced_data
                    )
                    batch_rmse = np.sqrt(
                        np.mean((original_distances - reduced_distances) ** 2)
                    )
                    rmse_sum += batch_rmse
                    num_batches += 1

                rmse = rmse_sum / num_batches
                reconstruction_errors.append(rmse)

            reconstruction_errors_percent = (
                1 - np.cumsum(reconstruction_errors) / np.sum(reconstruction_errors)
            ) * 100

            runtime = time.time() - start_lle

            return {
                "components": n_components,
                "trustworthiness": None,
                "reconstruction_error": reconstruction_errors_percent,
                "time": runtime,
            }

        method_funcs = {
            "pca": run_pca,
            "isomap": run_isomap,
            "tsne": run_tsne,
            "umap": run_umap,
            "autoencoder": run_autoencoder,
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
            delayed(method_funcs[method])() for method in valid_methods
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
                embedding_2d = self._apply_autoencoder(
                    n_components=2, hidden_layer_neurons=4
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

    def _apply_autoencoder(self, n_components, hidden_layer_neurons):
        """
        Helper function to apply an autoencoder for dimensionality reduction.

        Parameters:
            data (array-like): Input data for dimensionality reduction.

        Returns:
            embedding_2d (array-like): 2D representation of the input data.
        """
        input_layer = Input(shape=(self.data.shape[1],))
        encoded = Dense(hidden_layer_neurons, activation="relu")(input_layer)
        encoded = Dense(n_components, activation="relu")(encoded)
        decoded = Dense(hidden_layer_neurons, activation="relu")(encoded)
        decoded = Dense(self.data.shape[1], activation="sigmoid")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)

        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(
            self.data, self.data, epochs=50, batch_size=32, shuffle=True, verbose=0
        )

        return encoder.predict(self.data)

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
                    embedding = self._apply_autoencoder(
                        n_components=n_components, hidden_layer_neurons=6
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
