import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='jigsaw',
    version='0.2',
    packages=['jigsaw'],
    include_package_data=True,
    license='BSD License',
    description='A jigsaw puzzle solver.',
    long_description=README,
    author='Matthijs Moed',
    author_email='mhmoed@gmail.com',
    install_requires=[
        'click==6.2',
        'numpy==1.10.1',
        'scikit-image==0.11.3',
        'scipy==0.16.0'
    ],
    scripts=[
        'bin/shuffle-image',
        'bin/solve-jigsaw-lp'
    ]
)
