"""
Setup script for Bakken BNN package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="bakken-bnn",
    version="1.0.0",
    author="Jackson R. Tomski",
    author_email="",
    description="Bayesian Neural Network for Bakken Reservoir Characterization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="rockPropCalculator"),
    package_dir={"": "rockPropCalculator"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scipy>=1.7.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "segyio>=1.9.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "ipython>=7.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
)