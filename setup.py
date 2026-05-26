from setuptools import find_packages, setup
from pathlib import Path
import os

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Version resolution order:
#   1. DEEPSLICE_VERSION environment variable (CI tag injection)
#   2. VERSION file at repo root
#   3. Fallback to 0.0.0+local (never call git at install time; breaks inside sdists)
version = os.getenv("DEEPSLICE_VERSION", "").strip()
if not version or version.startswith("{{"):
    version_file = this_directory / "VERSION"
    if version_file.exists():
        version = version_file.read_text(encoding="utf-8").strip() or "0.0.0+local"
    else:
        version = "0.0.0+local"

setup(
    name="DeepSlice",
    python_requires=">=3.9,<3.13",
    packages=find_packages(),
    version=version,
    license="GPL-3.0",
    description="A package to align histology to 3D brain atlases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepSlice Team",
    package_data={
        "DeepSlice": [
            "metadata/volumes/placeholder.txt",
            "metadata/config.json",
            "metadata/weights/*.txt",
        ]
    },
    include_package_data=True,
    author_email="harry.carey@medisin.uio.no",
    url="https://github.com/PolarBean/DeepSlice",
    download_url=f"https://github.com/PolarBean/DeepSlice/archive/refs/tags/{version}.tar.gz",
    keywords=["histology", "brain", "atlas", "alignment"],
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-image>=0.22",
        "scipy>=1.10",
        # Keras 3 (TF 2.16+) changed callback APIs we depend on; pin until tested.
        "tensorflow>=2.13,<2.16",
        "h5py>=3.9",
        "requests>=2.31",
        "protobuf>=4.21",
        "lxml>=4.9",
        "Pillow>=10.0",
        "matplotlib>=3.8",
        "PySide6>=6.6",
    ],
    extras_require={
        "atlas": ["nibabel>=5.2"],
        "pdf": ["reportlab>=4.0"],
        "dev": [
            "pytest>=8.0",
            "pytest-qt>=4.4",
            "coverage>=7.4",
        ],
        "all": ["nibabel>=5.2", "reportlab>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "deepslice-gui=DeepSlice.gui.app:main",
            "deepslice-train=DeepSlice.training.train_runner:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
    ],
)
