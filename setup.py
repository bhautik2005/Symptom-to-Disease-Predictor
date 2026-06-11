from setuptools import setup, find_packages

setup(
    name="symptom-disease-predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.3",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "joblib>=1.4.2",
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "pyyaml>=6.0.1",
        "jinja2>=3.1.4",
        "imbalanced-learn",
        "tabulate",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "symptom-predictor=main:main",
        ],
    },
)
