from setuptools import setup, find_packages
from typing import List

HYPERLINKS = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This functions returns the  list of requirements mentioned in the requirements.txt file
    '''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPERLINKS in requirements:
            requirements.remove(HYPERLINKS)
        return requirements


setup(
    name='mlpackage',
    version='0.1.0',
    author='Venkata Girish',
    author_email='21AG1A6645@gmail.com',
    description='A machine learning package for data preprocessing, model training, and evaluation.',
    packages=find_packages(exclude=("Venkat", "Venkat.*")),
    install_requires=get_requirements('requirements.txt')
)