from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """ 
    This function will return list of requirements
    """

    requirement_list: List[str] = []

    try:
        with open("requirements.txt", "r") as requirement_file:
            lines = requirement_file.readlines()

            for line in lines:
                requirement = line.strip()

                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")

    return requirement_list


setup(
    name="agentic_trip_planner",
    version="0.1.0",
    author="Rohit More",
    author_email="rohit4690@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)