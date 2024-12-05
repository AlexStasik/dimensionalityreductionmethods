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
    version="0.1.2",
    description="A package for applying, comparing, and visualizing dimensionality reduction methods across various target dimensions.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DarianSINTEF/dimensionalityreductionmethods",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Afroditi Natsaridou <asikonats@gmail.com>, Darian Zhang <darianyzhang@gmail.com>",
)
