"""
Setup script for brainwave analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="brainwave-analysis",
    version="1.0.0",
    author="Brainwave Analysis Team",
    author_email="team@brainwave-analysis.com",
    description="A comprehensive toolkit for EEG/EOG data analysis and gaze direction prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/brainwave-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "deep": [
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brainwave-analysis=brainwave_analysis.src.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brainwave_analysis": ["configs/*.yaml"],
    },
)
