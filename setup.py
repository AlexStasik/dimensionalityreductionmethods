from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy>=1.19.2",
    "matplotlib>=3.3.4",
    "scikit-learn>=0.24.2",
    "tensorflow>=2.4.0",
    "pandas>=1.2.3",
    "umap-learn>=0.5.0",
    "tabulate>=0.8.9",
    "joblib>=1.0.1",
    "torch>=1.9.0",
    "ipykernel",
    "ipywidgets>=7.6.0",
]

setup(
    name="dimensionalityreductionmethods",
    version="0.1",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
