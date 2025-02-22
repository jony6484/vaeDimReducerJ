from setuptools import find_packages, setup

VERSION = '0.2.0'
DESCRIPTION = 'Vae Dim Reducer'
setup(
    name='VaeDimReducerJ',
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    author='Jonathan Fuchs',
    author_email="<jony6484@gmail.com>",
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
    
)
