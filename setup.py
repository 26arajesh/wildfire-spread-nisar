
from setuptools import setup, find_packages

setup(
    name='wildfire-spread-nisar',
    version='0.1.0',
    description='Wildfire Spread Prediction with NISAR Data',
    author='Aditya Rajesh',
    author_email='the.aditya.rajesh@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'torch',
        'rasterio',
        'geopandas',
        'shapely',
        'matplotlib',
        'jupyterlab',
        'dvc',
        'mlflow',
        'pytest',
        'black',
        'flake8',
        'ipykernel',
    ],
    python_requires='>=3.8',
)
