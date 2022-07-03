from setuptools import setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open("requirements-dev.txt") as dev_requirements_file:
    dev_requirements = dev_requirements_file.read().splitlines()

setup(
    name="vs-split",
    version="0.1.0",
    author="Lj V. Miranda",
    author_email="ljvmiranda@gmail.com",
    packages=["vs_split"],
    url="https://github.com/ljvmiranda921/vs-split",
    license="MIT LICENSE",
    description="A Python library for creating adversarial splits",
    long_description=readme,
    install_requires=requirements,
    test_requires=dev_requirements,
    keywords=["machine learning", "nlp", "statistics"],
)
