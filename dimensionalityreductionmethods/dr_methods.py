import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import root_mean_squared_error, pairwise_distances
import umap

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def run_pca(scaled_data):
    start_pca = time.time()

    pca = PCA()
    reduced_data = pca.fit_transform(scaled_data)
    reconstructed_data = pca.inverse_transform(reduced_data)
    rmse = root_mean_squared_error(scaled_data, reconstructed_data)

    explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
    reconstruction_errors = [(1.0 - evr) * 100 for evr in explained_variance_cumsum]

    runtime = time.time() - start_pca

    return {
        "components": np.arange(1, explained_variance_cumsum.shape[0] + 1),
        "trustworthiness": None,
        "reconstruction_error": reconstruction_errors,
        "time": runtime,
    }


def run_isomap(scaled_data):
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


def run_tsne(scaled_data):
    start_tsne = time.time()

    n_components_range = np.arange(1, scaled_data.shape[1] + 1)
    trustworthiness_scores = []

    for n in n_components_range:
        tsne = TSNE(n_components=n, perplexity=20, random_state=42, method="exact")
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


def run_umap(scaled_data):
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


def run_autoencoder(scaled_data, autoencoder_max_dim):
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

        reconstruction_error = root_mean_squared_error(scaled_data, decoded_data)
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


def run_kpca(scaled_data):
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


def run_lle(scaled_data):
    start_lle = time.time()

    reconstruction_errors = []
    n_components = np.arange(1, scaled_data.shape[1] + 1)

    for n in n_components:
        lle = LocallyLinearEmbedding(n_components=n, n_neighbors=10, method="standard")
        reduced_data = lle.fit_transform(scaled_data)
        batch_size = 1000
        rmse_sum = 0
        num_batches = 0

        for i in range(0, scaled_data.shape[0], batch_size):
            end = min(i + batch_size, scaled_data.shape[0])
            original_distances = pairwise_distances(scaled_data[i:end], scaled_data)
            reduced_distances = pairwise_distances(reduced_data[i:end], reduced_data)
            batch_rmse = np.sqrt(np.mean((original_distances - reduced_distances) ** 2))
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


def apply_autoencoder(data, n_components, hidden_layer_neurons):
    """
    Helper function to apply an autoencoder for dimensionality reduction.

    Parameters:
        data (array-like): Input data for dimensionality reduction.

    Returns:
        embedding_2d (array-like): 2D representation of the input data.
    """
    input_layer = Input(shape=(data.shape[1],))
    encoded = Dense(hidden_layer_neurons, activation="relu")(input_layer)
    encoded = Dense(n_components, activation="relu")(encoded)
    decoded = Dense(hidden_layer_neurons, activation="relu")(encoded)
    decoded = Dense(data.shape[1], activation="sigmoid")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(data, data, epochs=50, batch_size=32, shuffle=True, verbose=0)

    return encoder.predict(data)
