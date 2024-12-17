from setuptools import setup, find_packages

setup(
    name='NASearch_lib',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'calflops==0.2.9',
        'matplotlib==3.8.2',
        'networkx==3.2.1',
        'numpy==1.26.4',
        'psutil==5.9.8',
        'PyYAML==6.0.1',
        'scikit_learn==1.4.0',
        'scipy==1.13.0',
        'setuptools==68.2.2',
        'torch==2.1.2',
        'torchvision==0.16.2',
        'transformers==4.40.1'
    ],
    # other metadata
)
