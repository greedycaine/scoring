# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scoring",
    version="0.0.1",
    author="Pan Fu",
    author_email="panfu0207@gmail.com",
    description="Build scorecard for credit risk analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greedycaine/scoring",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)