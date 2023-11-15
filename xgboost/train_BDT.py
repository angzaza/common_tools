import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
from ModelHandler import *

class MuonMVA(ModelHandler):
    pass
warnings.filterwarnings("default", category=UserWarning, module="numpy")

def train_model(files, name, config, index=None, category=None, condor=False):
    model = MuonMVA(config)
    model.load_datasets(files, config)

    with open(config, 'r') as file:
        json_file = json.load(file)
    number_of_splits = json_file['number_of_splits']
    output_path = json_file['output_path']
    selections_keys = [key for key in json_file.keys() if key.startswith("selections_")]
    print("pre-selections: ",selections_keys)

    for key in selections_keys:
        model.apply_selection(config, key)
    
    if category!=None:
        category_key = [key for key in json_file.keys() if key.startswith(category)]
        print(category, " selections: ", category_key)
        for key in category_key:
            if "_sig_" in key:
                mask2 = model.data[model.Y_column]
                mask2 = mask2.apply(lambda x: x != model.BDT_0) #sig is true and bkg is false
                mask2 = ~mask2
                model.apply_selection(config, key, mask2)
            elif "_bkg_" in key:
                mask2 = model.data[model.Y_column]
                mask2 = mask2.apply(lambda x: x != model.BDT_0) #sig is true and bkg is false
                model.apply_selection(config, key, mask2)
            else:
                model.apply_selection(config, key)
                
    output_path_new = output_path + "/" + date
    if not os.path.exists(output_path_new):
        subprocess.call("mkdir -p %s" % output_path_new, shell=True)

    if not condor:
        config_out = "%s/%s_config.json" % (output_path_new, name)
        subprocess.run(["cp", config, config_out])
        with open(config_out, 'r') as file:
            dati = json.load(file)
        new_key = 'date'
        dati[new_key] = date
        with open(config_out, 'w') as file:
            json.dump(dati, file, indent=2)
    
    if index==None:
        for event_index in range(number_of_splits):
            model.prepare_train_and_test_datasets(config, number_of_splits, event_index)
            print("Number of signal/background events in training sample: %u/%u (%0.3f)" \
                % ((model.y_train == True).sum(),
                   (model.y_train == False).sum(),
                   (model.y_train == True).sum() / float((model.y_train == False).sum())))
            if(category!=None):
                model_name = "%s/%s-Event%u-Category_%s" % (date, name, event_index, category)
            else:
                model_name = "%s/%s-Event%u" % (date, name, event_index)
            #model.add_weight()  #old reweight
            model.train(config, model_name)
    else:
            model.prepare_train_and_test_datasets(config, number_of_splits, index)
            print("Number of signal/background events in training sample: %u/%u (%0.3f)" \
                % ((model.y_train == True).sum(),
                   (model.y_train == False).sum(),
                   (model.y_train == True).sum() / float((model.y_train == False).sum())))
            if(category!=None):
                model_name = "%s/%s-Event%u-Category_%s" % (date, name, index, category)
            else:
                model_name = "%s/%s-Event%u" % (date, name, index)
            #model.add_weight()  #old reweight
            model.train(config, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BDT training")

    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--category", type=str, help="Name of the category")
    parser.add_argument("--index", type=int, help="Index number")
    parser.add_argument("--condor", action="store_true", help="Use of condor")

    
    args = parser.parse_args()
    
    config = args.config
    category = args.category
    index = args.index
    condor = args.condor

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with open(config, 'r') as file:
        json_file = json.load(file)
    data_path = json_file['data_path']
    files = json_file['files']
    name = json_file['Name']
    if condor:
        date = json_file['date']
        
    files_Run2022 = [data_path + i for i in files]
    
    train_model(files_Run2022, name, config, index, category, condor)
    #Fa il train sui number_of_splits fold ma poi manca la parte dove vengono aggrgati (?)
    #Quindi bisogna creare un nuovo file .root con il nuovo branch
    #Implementare la divisione in categorie per tau3mu

