from setuptools import setup, find_packages

setup(
    name="ngn",
    version="0.1.0",
    description="Neural Graph Network - Learning dynamic inter-layer communication",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.10.0",
        "wandb>=0.13.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)