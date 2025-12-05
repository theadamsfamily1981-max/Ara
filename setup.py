"""
Legacy setup.py for backwards compatibility.
Prefer using pyproject.toml for modern Python packaging.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements (optional - core deps are in pyproject.toml)
requirements = []
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    requirements = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ara",
    version="0.2.0",
    author="ARA Framework Team",
    description="Ara: Autonomous Research Agent - A unified AI system with meta-learning, embodiment, and self-improvement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "scipy>=1.10.0",
            "transformers>=4.30.0",
            "geoopt>=0.5.0",
            "gudhi>=3.8.0",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest>=7.4.0",
            "pytest-benchmark>=4.0.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ara=ara.main:main",
            "ara-chat=ara.cli:main",
            "ara-meta=ara.meta.cli:main",
        ],
    },
)
