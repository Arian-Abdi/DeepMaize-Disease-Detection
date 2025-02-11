from setuptools import setup, find_packages

setup(
    name="deepmaize",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'timm>=0.9.0',
        'pillow>=9.0.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.2.0',
        'tqdm>=4.65.0',
        'pytest>=8.0.0',
    ],
    author="Arian Abdipor",
    author_email="arian.abdipour9@gmail.com",
    description="Deep learning-powered maize leaf disease detection using Swin Transformer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Arian-Abdi/DeepMaize-Disease_Detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)