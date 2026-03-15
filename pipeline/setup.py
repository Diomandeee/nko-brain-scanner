"""
Setup configuration for N'Ko Training Pipeline and Benchmark Suite.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="nko-training",
    version="1.0.0",
    description="N'Ko Language Training Pipeline and AI Model Benchmark Suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="N'Ko Learning Platform",
    author_email="",
    url="https://github.com/learnnko/learnnko",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "benchmark": [
            "anthropic>=0.40.0",
            "openai>=1.50.0",
            "sacrebleu>=2.4.0",
            "google-genai>=0.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nko-benchmark=training.benchmarks.nko_benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)

