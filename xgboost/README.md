Code taken from Bmm5 analysis
https://github.com/drkovalskyi/Bmm5/blob/master/MVA/ModelHandler.py 

ModelHandler.py is a wrapper around XGBoost. All that you need to do is to create a new class that inherits from ModelHandler, provide feature names, input files and tune the parameters.

To setup the environment, use Run3 CMSSW release:

```
scram p CMSSW CMSSW\_13\_0\_13
cd CMSSW\_13\_0\_13/src/
cmsenv
```


See a working example running on DiMuon BPH MC nanoAODs

`python train_muon_mva.py` 
