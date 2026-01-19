"""
Setup script for Semi-UNet3+-CBAM IC SEM Defect Segmentation package
"""

from setuptools import setup, find_packages


setup(
    name="semi-unet3p-cbam",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for IC SEM defect detection using Semi-UNet3+ with CBAM",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semi-unet3p-cbam",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.17.0",
        "Pillow>=8.0.0",
        "tensorboard>=2.5.0",
        "tqdm>=4.50.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "PyYAML>=5.4.0",
        "albumentations>=1.1.0"
    ],
    entry_points={
        "console_scripts": [
            "semi-unet3p-train=cli.train:main",
            "semi-unet3p-preprocess=cli.preprocess_data:main",
        ],
    },
)