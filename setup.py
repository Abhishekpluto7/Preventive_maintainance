from setuptools import setup, find_packages

requirements = [
    'sklearn>=0.22.1',
    'argparse>=1.1',
    'pandas>=1.0.1',
    'numpy>=1.18.1',
    'tensorflow>=1.14.0',
    
]
setup(
    author = "Abhishek R",
    author_email = 'abhishekraju@pluto7.com',
    python_requires = '>=3.6.9',
    description = 'Package for predicting Remaining Useful Lifecycle(RUL) for \'Turbofan engine degradation simulation dataset\' from NASA\'s open data portal usning RANDOM FOREST REGRESSION Algorithm',
    name = 'randomforestregression',
    version = '0.1.0',
    install_requires = requirements,
    packages = find_packages(),
    include_package_data = True
)