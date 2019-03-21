# -*- coding: utf-8 -*-

import setuptools
from os import path
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the version from __init__
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'scoring/__init__.py'), encoding='utf-8') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)
# with open('/scoring/__init__.py') as fh2:
#     __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fh2.read()).group(1)

setuptools.setup(
    name="scoring",
    version=__version__,
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