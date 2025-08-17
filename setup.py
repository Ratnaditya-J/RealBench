from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="realbench",
    version="0.1.0",
    author="RealBench Team",
    author_email="contact@realbench.org",
    description="Real-world benchmark for Generative AI evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RealBench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
        "jsonschema>=4.0.0",
        "tabulate>=0.8.9",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8",
        "spacy>=3.4.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "requests>=2.25.0",
        ],
        "web": [
            "streamlit>=1.10.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "realbench=cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "realbench": ["data/**/*.json", "templates/*.html"],
    },
)
