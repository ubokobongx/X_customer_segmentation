from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements if req.strip() and not req.startswith('#')] 
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name = 'CustomerSegmentation',
version = '0.01',
author = 'Ubokobong',
author_email = 'ubokobong@oxygenx.africa',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)