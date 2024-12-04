# DimensionalityReductionMethods

Dimensionality reduction is an essential aspect of data analysis and machine learning. It allows for the transformation of high-dimensional data into more manageable, interpretable forms while also preserving the core structure of the data. This package aims to simplify the application of various dimensionality reduction techniques, including methods like PCA, t-SNE, and UMAP, to datasets across a wide range of target dimensions.

With this package, users can:
- Perform dimensionality reduction using multiple methods with minimal setup and compare methods using side-by-side evaluations.
- Analyze results quantitatively, measuring trustworthiness (how well local relationships are preserved during reduction) and reconstruction error (the discrepancy between original and reconstructed data) to assess the performance of methods and determine the intrinsic dimensionality of the data.
- Visualize lower dimensional projections for further insights into the data structure and relationships.

By combining automation, visualization, and flexibility, this package simplifies the exploration of high-dimensional datasets and guides users in choosing the best methods for their applications.

## Installation
DimensionalityReductionMethods can be easily installed via pip from the [PyPI repository](https://test.pypi.org/project/dimensionalityreductionmethods/). The below command will install the DimensionalityReductionMethods package and its dependencies.
NOTE: currently links to testpypi, same with below code

```console
pip install -i https://test.pypi.org/simple/ dimensionalityreductionmethods
```

## Getting Started
Below is a step-by-step guide on how to use the package.

### 1. Import the package.
```python
import dimensionalityreductionmethods as drm
```
### 2. Intialize the `DimensionalityReductionHandler` with your dataset. Ensure the dataset is a Numpy array.
```python
drh = drm.DimensionalityReductionHandler(data)
```
### 3. Provide a list of dimensionality reduction methods to apply to the data. 
The supported methods are: PCA, KPCA, Isomap, UMAP, t-SNE, Autoencoder, LLE.
```python
drh.analyze_dimensionality_reduction(
    [
        "isomap",
        "PCA",
        "tSNE",
        "Umap",
        "kpca",
        "autoencoder",
        "lle",
    ]
)
```
This method applies the specified dimensionality reduction techniques to the dataset. The data is reduced to 1 to n dimensions, where n is the original dimensionality of the dataset. It computes performance metrics such as trustworthiness and reconstruction error for each method (if applicable), helping to evaluate how well each method preserves the data's structure in lower dimensions.
### 4. Plot the results of the dimensionality reduction methods. 
The plot summarizes the performance of each method, such as trustworthiness and reconstruction error across dimensions.
```python
drh.plot_results()
```
### 5. Display a summary table of the results.
The table shows the optimal components, maximum trustworthiness, minimum reconstruction error, and computation time for each dimensionality reduction method.
```python
drh.table()
```
### 6. Visualize low-dimensional projections of the data.
```python
drh.visualization()
???drh.visualization_3d(plot_in_3d=True)???
```

### All steps together:
```python
import dimensionalityreductionmethods as drm

# Initialize the handler with your data
drh = drm.DimensionalityReductionHandler(data)

# Analyze dimensionality reduction using selected methods
drh.analyze_dimensionality_reduction(
    [
        "isomap",
        "PCA",
        "tSNE",
        "Umap",
        "kpca",
        "autoencoder",
        "lle",
    ]
)

# Visualize and summarize the results
drh.plot_results()
drh.table()
drh.visualization()
???drh.visualization_3d(plot_in_3d=True)???
```

The examples folder includes sample notebooks featuring toy datasets that serve as helpful references.