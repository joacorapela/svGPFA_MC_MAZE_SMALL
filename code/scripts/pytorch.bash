#!/bin/bash

SVGPFA_SRC=~/svGPFA/repos/svGPFA/
GCNU_COMMON_REPO=~/gatsby-swc/gatsby/code/gcnu_common_repo
CURRENT_DIR=`pwd`

source ~/.condaInit
conda activate svGPFA

cd $SVGPFA_SRC
git checkout optimAllParamsSimultaneouslyVariableNIndPointsPerLatent

cd $GCNU_COMMON_REPO
git checkout pytorch1

cd $CURRENT_DIR
