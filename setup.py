from setuptools import setup, find_packages

setup(
    name='nirdizati_light',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dice_ml @ git+https://github.com/abuliga/DiCE.git@origin/main',
        'declare4py @ git+https://github.com/abuliga/declare4py.git@main',
        'pymining==0.2',
        'pandas~=1.5.3',
        'pm4py~=2.7.10.3',
        'scikit-learn~=1.3.0',
        'shap~=0.41.0',
        'numpy~=1.23.3',
        'hyperopt~=0.2.7',
        'dateparser~=1.1.8',
        'holidays~=0.28',
        'funcy~=2.0.0',
        'xgboost~=2.0.3',
        'pymoo~=0.6.0.1',
        'torch~=2.2.1',
        'PDPbox~=0.3.0',
        'seaborn~=0.13.2',
        'simpy~=4.1.0',
    ]
)
