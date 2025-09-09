"""Setup script for hookedllm."""

from setuptools import find_packages, setup


setup(
    name="hookedllm",
    version="0.1.0",
    description="A lightweight library for extracting and steering activations from LLMs",
    author="Steven Abreu",
    author_email="s.abreu@rug.nl",
    url="https://github.com/stevenabreu7/conceptor-llm",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "safetensors>=0.3.0",
        "transformers>=4.20.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "black>=22.0.0", "isort>=5.0.0", "flake8>=4.0.0", "debugpy>=1.6.0"]
    },
)
