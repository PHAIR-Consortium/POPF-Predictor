from setuptools import setup

setup(
    name='POPF-Predictor',
    version='1.1',
    url=' https://github.com/PHAIR-Amsterdam/POPF-Predictor.git',
    install_requires=[
        'imbalanced_learn>=0.9.0',
        'imblearn>=0.0',
        'matplotlib>=3.5.3',
        'numpy>=1.21.6',
        'pandas>=1.1.5',
        'scikit_learn>=1.0.2',
        'scipy>=1.7.3',
        'sdv>=0.17.1',
        'openpyxl'
    ],
    keywords=['deep learning', 'radiomics', 'medical prediction models', 'medical image analysis','medical image segmentation', 'POPF-Predictor'],
    author='PHAIR-Amsterdam',
    author_email='j.i.bereska@amsterdamumc.nl',
    description=' POPF predictor - Framework for predicting POPF from CT radiomic features'
)
