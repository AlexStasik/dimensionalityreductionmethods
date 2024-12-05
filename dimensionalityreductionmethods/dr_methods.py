import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import root_mean_squared_error, pairwise_distances
import umap

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def run_pca(scaled_data):
    """
    Perform Principal Component Analysis (PCA) on the provided scaled data.

    This method applies PCA to reduce the dimensionality of the data and calculates the reconstruction error and explained variance
    for each principal component. It also returns the runtime of the PCA process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the PCA analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (None): Trustworthiness is not calculated for PCA.
            - "reconstruction_error" (list): A list of reconstruction errors (as percentages) for each number of components.
            - "time" (float): The time taken to run the PCA analysis.

    Notes:
        The reconstruction error is calculated as the percentage of variance not explained by each principal component. It is based on
        the cumulative explained variance ratio of PCA.
    """
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
    """
    Perform Isomap dimensionality reduction on the provided scaled data.

    This method applies Isomap to reduce the dimensionality of the data for different numbers of components (from 1 to n) and calculates
    the reconstruction error and trustworthiness for each transformation. It also returns the runtime of the Isomap process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the Isomap analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (list): A list of trustworthiness scores (as percentages) for each number of components.
            - "reconstruction_error" (list): A list of reconstruction errors (as percentages) for each number of components.
            - "time" (float): The time taken to run the Isomap analysis.
    """
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
    """
    Perform t-SNE (t-Distributed Stochastic Neighbor Embedding) on the provided scaled data.

    This method applies t-SNE to reduce the dimensionality of the data for different numbers of components (from 1 to n) and calculates
    the trustworthiness for each transformation. It also returns the runtime of the t-SNE process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the t-SNE analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (list): A list of trustworthiness scores (as percentages) for each number of components.
            - "reconstruction_error" (None): Reconstruction error is not calculated for t-SNE.
            - "time" (float): The time taken to run the t-SNE analysis.
    """
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
    """
    Perform UMAP (Uniform Manifold Approximation and Projection) on the provided scaled data.

    This method applies UMAP to reduce the dimensionality of the data for different numbers of components (from 1 to n) and calculates
    the trustworthiness for each transformation. It also returns the runtime of the UMAP process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the UMAP analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (list): A list of trustworthiness scores (as percentages) for each number of components.
            - "reconstruction_error" (None): Reconstruction error is not calculated for UMAP.
            - "time" (float): The time taken to run the UMAP analysis.
    """
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
    """
    Perform Autoencoder-based dimensionality reduction on the provided scaled data.

    This method trains an autoencoder for different numbers of encoding dimensions (from 1 to the specified maximum dimension) and
    calculates both trustworthiness and reconstruction error for each transformation. It also returns the runtime of the process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.
        autoencoder_max_dim (int): The maximum number of encoding dimensions to consider for the autoencoder.

    Returns:
        dict: A dictionary containing the results of the Autoencoder analysis:
            - "components" (numpy array): The number of encoding dimensions considered (from 1 to n).
            - "trustworthiness" (list): A list of trustworthiness scores (as percentages) for each number of components.
            - "reconstruction_error" (list): A list of reconstruction errors (as percentages) for each number of components.
            - "time" (float): The time taken to run the Autoencoder analysis.
    """
    start_autoencoder = time.time()

    input_dim = scaled_data.shape[1]
    max_dim = min(input_dim, autoencoder_max_dim)
    encoding_dims = np.arange(1, max_dim + 1)

    trustworthiness_scores = []
    reconstruction_errors = []

    for dim in encoding_dims:
        input_layer = Input(shape=(input_dim,))

        encoder_1 = Dense(input_dim, activation="relu")(input_layer)
        encoder_2 = Dense(input_dim, activation="relu")(encoder_1)
        encoder_3 = Dense(input_dim, activation="relu")(encoder_2)
        encoder_4 = Dense(input_dim, activation="relu")(encoder_3)

        encoded = Dense(dim, activation="relu")(encoder_4)

        decoder_1 = Dense(input_dim, activation="relu")(encoded)
        decoder_2 = Dense(input_dim, activation="relu")(decoder_1)
        decoder_3 = Dense(input_dim, activation="relu")(decoder_2)
        decoder_4 = Dense(input_dim, activation="relu")(decoder_3)

        decoded = Dense(input_dim, activation="sigmoid")(decoder_4)

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
    """
    Perform Kernel Principal Component Analysis (KPCA) on the provided scaled data.

    This method applies KPCA using a Radial Basis Function (RBF) kernel to reduce the dimensionality of the data for different
    numbers of components. It calculates the reconstruction error for each transformation and also returns the runtime of the process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the KPCA analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (None): Trustworthiness is not calculated for KPCA.
            - "reconstruction_error" (list): A list of reconstruction errors (as percentages) for each number of components.
            - "time" (float): The time taken to run the KPCA analysis.
    """
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
    """
    Perform Locally Linear Embedding (LLE) on the provided scaled data.

    This method applies LLE to reduce the dimensionality of the data for different numbers of components. It calculates the
    reconstruction error for each transformation using pairwise distances between the original and the reduced data.
    It also returns the runtime of the process.

    Parameters:
        scaled_data (numpy array): The input data that has been preprocessed.

    Returns:
        dict: A dictionary containing the results of the LLE analysis:
            - "components" (numpy array): The number of components considered (from 1 to n).
            - "trustworthiness" (None): Trustworthiness is not calculated for LLE.
            - "reconstruction_error" (list): A list of reconstruction errors (as percentages) for each number of components.
            - "time" (float): The time taken to run the LLE analysis.
    """
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


def get_autoencoder_embedding(data, n_components, hidden_layer_neurons):
    """
    Helper function to apply an autoencoder for dimensionality reduction and return the lower-dimensional embedding.

    Parameters:
        data (numpy array): The input data to reduce.
        n_components (int): The desired number of components (dimensionality) for the reduced representation.
        hidden_layer_neurons (int): The number of neurons in the hidden layers of the autoencoder.

    Returns:
        numpy array: The reduced-dimensional representation of the input data.

    Notes:
        The autoencoder is trained using mean squared error (MSE) loss, and the encoder is used to obtain the embedding.
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
