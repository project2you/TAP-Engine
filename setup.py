from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [r for r in requirements if not r.startswith("#") and r.strip()]

setup(
    name="tap-engine",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Token-Adaptive Precision Core Engine for efficient LLM inference and training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tap-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[r for r in requirements if not r.endswith('; platform_system != "Windows"')],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "sphinx>=6.2.0",
            "black>=23.3.0",
        ],
        "triton": ["triton>=2.0.0"],
        "viz": ["matplotlib>=3.7.0"],
        "full": [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
            "bitsandbytes>=0.39.0",
            "peft>=0.4.0",
            "matplotlib>=3.7.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tap-engine=tap_engine.cli:main",
        ],
    },
)
