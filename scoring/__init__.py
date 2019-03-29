# -*- coding:utf-8 -*-

from scoring.cleanning import delFromVardict,renameCols,getVarTypes,fillMissing,describe
from scoring.bivariate import *
from scoring.discretization import *
from scoring.evaluation import *
from scoring.scoring import *


# python setup.py sdist bdist_wheel
# twine upload dist/*

name='scoring'
__version__='0.0.8'
