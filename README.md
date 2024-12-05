# DimensionalityReductionMethods

Dimensionality reduction is an essential aspect of data analysis and machine learning. It allows for the transformation of high-dimensional data into more manageable, interpretable forms while also preserving the core structure of the data. This package aims to simplify the application of various dimensionality reduction techniques, including methods like PCA, t-SNE, and UMAP, to datasets across a wide range of target dimensions.

With this package, users can:
- Perform dimensionality reduction using multiple methods with minimal setup and compare methods using side-by-side evaluations.
- Analyze results quantitatively, measuring trustworthiness (how well local relationships are preserved during reduction) and reconstruction error (the discrepancy between original and reconstructed data) to assess the performance of methods and determine the intrinsic dimensionality of the data.
- Visualize lower dimensional projections for further insights into the data structure and relationships.

By combining automation, visualization, and flexibility, this package simplifies the exploration of high-dimensional datasets and guides users in choosing the best methods for their applications.

## Installation
DimensionalityReductionMethods can be easily installed via pip from the [PyPI repository](https://pypi.org/project/dimensionalityreductionmethods/). The below command will install the DimensionalityReductionMethods package and its dependencies.

```console
pip install dimensionalityreductionmethods
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
By default, the data is visualized in 2D. Set `plot_in_3d=True` to generate a 3D visualization.
```python
drh.visualization()
drh.visualization(plot_in_3d=True)
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
drh.visualization(plot_in_3d=True)
```

The examples folder includes sample notebooks featuring toy datasets that serve as helpful references.


## Methods Overview

This section outlines key dimensionality reduction techniques, highlighting their functionality, benefits, and limitations to help users choose the best method for their data.

### **PCA** : Principal Component Analysis
 
This method aims to preserve the maximum variance of the high dimensional dataset while reduces the number of components to the smallest possible.
 
|Pros|Cons|
|:-----|:----|
| Simple method | Assumes linearity|
| Works well for linear data | Sensitive to scaled data|
| Computationally efficient | Sensitive to noisy data|
| Easy to interpret| Cannot capture nonlinear relationships |
 

### **KPCA** : Kernel Principal Component Analysis
 
Maximizes the variance between the high dimensional data within a nonlinear feature space, effectively capturing dataset's complex/nonlinear relationships using kernel functions to minimize the principal components. The principal components can represent the dataset's number of dimensions.

|Pros|Cons|
|:-----|:----|
| Nonlinear method | Kernel choice can be tricky|
| Flexible with different kernels| Higher computational cost|
| Works well for complex datasets| Sensitive to parameters like kernel width|
| Computationally expensive| May not perform well |

### **LLE** : Locally Linear Embedding

The main idea is to use k-neighbors of each point included in the dataset and tranform it as a combination of them. It computes the weights that best reconstruct each vector from its neighbors and then generates low-dimensional representations that can be reconstructed using these weights. In general, preserves local relationships by minimizing reconstruction error of each point.

|Pros|Cons|
|:-----|:----|   
| Effective for manifold learning | Sensitive to noise and number of neighbors|
| Captures intrinsic geometry| Computationally expensive for large datasets|
| Preserves local structures||
 

### **t-SNE** : t-distributed Stochastic Neighbor Embedding
Models the distribution of each point's closest neighbors -perplexity- and maps them onto a lower-dimensional space while maintaining local relationships. This allows dimensionality reduction while simultaniously clustering the data. The clusters present the relationship between the high dimensional data, visualized in the dimensionality reduced space.

|Pros|Cons|
|:-----|:----|
| Excels at visualizing high-dimensional data| Computationally intensive|
| Captures local clusters well| Hard to interpret quantitatively|
| Widely adopted in exploratory analysis| Does not preserve global structure|

### **ISOMAP** : Isometric Feature Mapping

Preserves the geodesic stucture between all data points of the high dimensional dataset while maximizing the variance. The lower dimensions, represented as principal components, summarize the intrinsic structure of the high dimensional data.

|Pros|Cons|
|:-----|:----|
| Good for manifold learning| Sensitive to noise|
| Preserves global structures| Requires good connectivity of the graph|
| Effective for datasets with intrinsic geometry| Computationally expensive for large datasets|


### **UMAP** : Uniform Manifold Approximation and Projection

Minimizes a cross-entropy loss between high-dimensional and low-dimensional fuzzy topological structures.
 
|Pros|Cons|
|:-----|:----|
| Fast and scalable| Hyperparameters tuning can affect results|
| Preserves both local and global structures| Interpretation is not as straightforward as PCA|
| Works well with noisy data||
| Versatile for visualization and clustering||


### **Autoencoder**

Autoencoders belong to the N.N. category and their structure helps the method to reduce the number of variables in a dataset. An autoecoder structure consists of an input layer, an output layer and various hidden layers, with the structure depending on the desired complexity for the N.N. In order to achieve dimensionality reduction, the dataset enters the input layer, passes through the hidden layers and reaches a bottleneck which reducts the dimension of the dataset. Then the decoders -the layers after the botttleneck- reconstruct the high dimensional dataset we previously entered as input.

|Pros|Cons|
|:-----|:----|
| Handles nonlinear data| Requires careful architecture tuning|
| Can be customized for different tasks| Training can be computationally expensive|
| Scalable to large datasets| Prone to overfitting|

### Dimensionality Reduction Performance
 
In order to determine the optimal method and the appropriate number of dimensions, we must evaluate the reconstruction error and the trustworthiness of each method -if they exist-. We consider one method as optimal if the trustworthiness is close to 100% and/or its reconstruction error is low. However, not all methods provide both metrics. The table below outlines which metrics are available for each method.

| Method |  Trustworthiness  | Reconstruction Error |
|:-----|:--------:|:------:|
| PCA   |_No_|_Yes_|
| KPCA  |_No_|_Yes_|
| LLE   |_Yes_|_Yes_|
| ISOMAP|_Yes_|_Yes_|
| UMAP  |_Yes_|_No_|
| tSNE  |_Yes_|_No_|
| AUTOENCODER |_Yes_|_Yes_|
