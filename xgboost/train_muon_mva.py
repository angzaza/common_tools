from ModelHandler import *

feature_names = [
    ### Good
    "trkValidFrac",
    "glbTrackProbability",
    "nLostHitsInner",
    "nLostHitsOuter",
    "trkKink",
    "chi2LocalPosition",
    "match2_dX",
    "match2_pullX",
    "match1_dX",
    "match1_pullX",

    ### Weak but useful
    "nPixels",
    "nValidHits",
    "nLostHitsOn",
    "match2_dY",
    "eta",
    "match1_dY",
    "match2_pullY",
    "match1_pullY",
    "match2_pullDyDz",
    "match1_pullDyDz",
    "match2_pullDxDz",
    "match1_pullDxDz",
    "pt",
]

class MuonMVA(ModelHandler):

    def get_true_classification(self, data):
        return (data["sim_type"] != 1)

    def get_parameters(self):
        parameters = super(MuonMVA, self).get_parameters()
        parameters['min_child_weight'] = 0.01
        parameters['max_depth'] = 5
        parameters['eta'] = 0.3
        parameters['tree_method'] = 'hist' 
        return parameters

def train_model(files, name):
    
    model = MuonMVA("results/muon_mva", feature_names, "muons", "evt")
    #model.do_weight=True
    model.load_datasets(files)    
    model.apply_selection(model.data["pt"]>2)
    #model.apply_selection(model.data["pt"]<6)
    #model.apply_selection(abs(model.data["eta"])<1.4)

    number_of_splits = 3
    train_model_for_all_splits = False
    for event_index in range(number_of_splits):
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        print("Number of signal/background events in training sample: %u/%u (%0.3f)" \
            % (sum(model.y_train == True),
               sum(model.y_train == False),
               sum(model.y_train == True) / float(sum(model.y_train == False))))
        model_name = "%s-%s-Event%u" % (name, date, event_index)
        #model.add_weight()  #old reweight
        model.train(model_name)
        if not train_model_for_all_splits:
            break

if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d-%H%M")

    data_path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/525/muon_mva/"

    files_Run2022 = [
        data_path + "InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
    ]
    train_model(files_Run2022, "Run2022")

