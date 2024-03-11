from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function return the contents of the requirements file as a list of strings
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

setup(
    name="mlproject",
    version="0.0.1",
    author="Alex Joseph",
    author_email="alex.joseph.2207@gmail.com",
    packages=find_packages(),
    requires_install=get_requirements('requirements.txt')
)