#!/bin/sh
# Usage:
#    launch_training.sh <ID>

helpstring="Usage:
launch_training.sh [ID]

[ID]: ID of the fold
"
ID=$1

# Check inputs
if [ -z ${1+x} ]
then
echo ${helpstring}
return
fi


source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv
python3 train_BDT.py --config CXN --index $ID --condor 
