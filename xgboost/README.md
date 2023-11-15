# XGBoost - Boosted Decision Trees
Note: The code is a modified version of the one used in the [Bmm5 analysis](https://github.com/drkovalskyi/Bmm5/blob/master/MVA/ModelHandler.py)

## Setting the environment

```
cmsrel CMSSW_13_0_13
cd CMSSW_13_0_13/src
cmsenv
git clone https://github.com/Ma128-bit/common_tools/ .
scram b -j20
```

## Configuration file 
ModelHandler.py is a wrapper around XGBoost. All you need to do is create a json configuration file that contains the following information:
* `feature_names` : name of the branches of the tree used for the BDT training. No feature wit name that contains "fold" or "bdt" or "bdt_cv"
* `other_branches` : other branches that you keed (for Y set, selections, weight, etc...). No branch wit name that contains "fold" or "bdt" or "bdt_cv"
* `Y_column` : The name of the branch used to create the Y set
* `BDT_0` : is the value of Y_column associated to 0 in the Y set, values of Y_column different from BDT_0 are set to 1 in the Y set
* `selections_*` : List of pre-selections (* = 1,2,3,...)
* `xgb_parameters` : Parameter of the XGBoost model
* `tree_name` : Name of the input tree in the root files
* `output_path` : Output path for saving results
* `event_fraction` : Fraction of root files used (usefull in case of big datasets)
* `validation_plots` : Boolean option to save validation plots
* `number_of_splits` : Number of kfoler
* `do_weight` : Boolean option to use weights
* `weight_column` : Tree branch used as weight
* `index_branch`: The branch used as index in fold splitting 
* `prediction_category_lable`: A branch that identifies the categories (0, 1, 2, 3, ...; in the same order in which the input categories are given in predict_BDT.py
* `out_tree_name`: Name of the TTree in the output file
* `Name` : Customizable name that will be inserted into the output files
* `data_path` : Path where the datasets are saved
* `files` : List of datasets in data_path

An example of configuration file is **`config/config.json`**

### Instructions for selections_ :
The code automatically run all the pre-selections `selections_*` listed in the configuration file.

Each selection have this shape:
```python=
"selections_1": [
    {"highPurity": [">", 0]},
    {"pt": [">", 2]},
    {"eta": ["<", 2.5]},
    {"eta": [">", -2.5]},
    {"isGlobal": ["==", true], "isTracker": ["!=", true]}
]
```
Notes: 
* At this moment, only >, <, == and != operators are supported. 
* Different rows (example `{"highPurity": [">", 0]}` and `{"pt": [">", 2]}`) are in AND
* Element on the same row (like: `{"isGlobal": ["==", true], "isTracker": ["!=", true]}`) are in OR.
    * **ATTENTION!** To use the OR **all** keys in the row (in the example `isGlobal` and `isTracker`) must be different otherwise the code won't work!
* It is also possible to implement the NOT of an entire selection writing "!" at the end of the name: `selections_1!`

### Instructions for splitting in categories :
Categories are treated as selections. For example, if you want a category called "Cat_A", you have to add selections like:
```python=
"Cat_A_1": [
    (like selections_1 in example before)
]
```
All the selections that starts with `Cat_A` will be applied.

There are also special selections: 
* If a selection, that starts with `Cat_A`, contains `_sig_` (like `Cat_A_sig_2`) it is applied only on signal data
* If a selection, that starts with `Cat_A`, contains `_bkg_` (like `Cat_A_bkg_3`) it is applied only on bkg data

NOTE: Please in category names do not use strange characters (especially '/' and '#') or spaces, possibly only use letters and '_'

## Training
The script **`train_BDT.py`** allows for the training of the model. It has 3 inputs:

`python3 train_BDT.py --config [config_file] --index [kfold_number] --category [category_name]`

* [config_file]: **Mandatory**  Name of the configuration file (for example **`config/config.json`**)
* [kfold_number]: **Optional**  ID of the fold (for example if `number_of_splits` in the configuration file is 5, kfold_number can be chosen between 0 and 4). If the parameter is not inserted the training will be performed on all folds.
* [category_name]: **Optional** Is used to train the mode on a category. When you pass this all the selections that starts with [category_name] will be applied.

NOTE: To train the model locally on different categories you must run the above command several times (equal to the number of categories) with the following precautions: 
1° Category: `python3 train_BDT.py --config [config_file] --category "Lable_for_Cat_1"
2° Category: `python3 train_BDT.py --config "copy_of_the_config_file_crated_in_previuous_step" --category "Lable_for_Cat_2" --condor
etc ...

### Output of **`train_BDT.py`** 
In the folder `output_path` it create a directory having the date in %Y%m%d-%H%M%S format as its name. In this directory there is: 
* A copy of the configuration file
* All the model trained (saved as .pkl)
* The validation plots

## Training on the batch system 
First use **`prepare_condor.sh`** to create the files for condor submission
```
source prepare_condor.sh [config_file] [categories_names]
```
* [config_file]: **Mandatory**  Name of the configuration file (for example **`config/config.json`**)
* [category_name]: **Mandatory** List of categories (like "Cat_A Cat_B Cat_C"). If there are no categories you have to pass ""

Example: `source prepare_condor.sh config/config_tau3mu.json "Cat_A Cat_B Cat_C"`

### Submission
**`prepare_condor.sh`** creates the directory **`output_path`/date** (in %Y%m%d-%H%M%S format) with a copy of the configuration file and files for submission (submit.condor, launch_training.sh one per categoy) 

To submit the jobs, form the main directory, run the submission of **`output_path`/date/submit.condor** (or submit_*.condor with * = categoy name)

## Predictions
To add the predictions of each model and the overall `bdt` score and `bdt_cv` score to the data you have to run:
```
python3 predict_BDT.py [copy_of_config_file] [categories_names]
```
Whare [copy_of_config_file] is the one descibed in **Output of `train_BDT.py`** and [categories_names] is like "Cat_A Cat_B Cat_C".

### Output of **`predict_BDT.py`** 
It saves original data (`feature_names` + `other_branches`) adding the branches: `fold_i_` (i=0, ..., `number_of_splits`-1), `bdt` and `bdt_cv`. Data are saved in .csv and .root (TTree) format.
It also saves feature importance plots




