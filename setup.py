from setuptools import setup

setup(
    name='LRP_CMF_integration',
    version='0.2',
    packages=['src',
              'src.log',
              'src.encoding',
              'src.labeling',
              'src.evaluation',
              'src.explanation',
              'src.explanation.wrappers',
              'src.confusion_matrix_feedback',
              'src.predictive_model',
              'src.hyperparameter_optimisation'],
    install_requires=[
        'pymining',
        'logging',
        'pandas',
        'pm4py',
        'functools',
        'sklearn',
        'shap',
        'numpy',
        'enum',
        'hyperopt',
        'pathlib',
        'tensorflow'
    ],
    url='https://github.com/PDI-FBK/LRP_CMF_integration',
    license='',
    author=['Williams Rizzi', 'Sven Weinzierl', 'Sandra Zilker'],
    author_email='wrizzi@fbk.eu',
    description='LRP CMF integration'
)
