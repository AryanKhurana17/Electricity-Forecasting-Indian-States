from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filename: str = "requirements.txt") -> List[str]:
    """
    Reads a requirements.txt file and returns a list of required packages.
    Ignores lines like '-e .'
    """
    requirements = []
    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if line and line != HYPHEN_E_DOT:
                    requirements.append(line)
    except FileNotFoundError:
        print(f"{filename} file not found.")
    return requirements
print(get_requirements())
setup(
    name="electricityforecasting", 
    version="0.0.1",
    author="Aryan Khurana",
    author_email="aryankhurana1701@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
