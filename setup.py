from setuptools import setup, find_packages

setup(
    name="smart_traffic_management",
    version="1.0.0",
    description="AI-Based Smart Traffic Management and Prediction Platform",
    author="Urban Analytics Engineer",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "plotly>=5.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.22.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
        "watchdog>=3.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
