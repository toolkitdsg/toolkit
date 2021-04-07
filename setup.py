from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


setup(
    name="ceg",
    version="0.0.1",
    author="Bruno Gonzales",
    author_email="brunocgf@gmail.com",
    description="A package for quality and presicion of data",
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pandas-gbq'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
