from setuptools import setup, find_packages

setup(name='fattah',
      packages=find_packages(), 
      install_requires=[
            'torch',
            'transformers',
            'scattermoe @ git+https://github.com/shawntan/scattermoe@main#egg=scattermoe'
      ])