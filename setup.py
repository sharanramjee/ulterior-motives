"""
Setup script for the ulterior-motives package.
"""

from setuptools import setup, find_packages

setup(
    name="ulterior-motives",
    version="0.1.0",
    description="Detecting Misaligned Reasoning in Continuous Thought Models",
    author="Anonymous",
    author_email="anonymous@example.com",
    url="https://anonymous.4open.science/r/ulterior-motives-anon/",
    packages=find_packages(),
    package_data={"moralchain": ["data/*.json"]},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
