import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
from ModelHandler import *

class MuonMVA(ModelHandler):
    pass
warnings.filterwarnings("default", category=UserWarning, module="numpy")

def predict_BDT_model(files, name, config, date, categories=None):
    model = MuonMVA(config)
    model.load_datasets(files, config)

    with open(config, 'r') as file:
        json_file = json.load(file)
    number_of_splits = json_file['number_of_splits']
    output_path = json_file['output_path']
    category_lable = json_file['prediction_category_lable']
    out_tree_name = json_file['out_tree_name']
    selections_keys = [key for key in json_file.keys() if key.startswith("selections_")]
    print("pre-selections: ",selections_keys)

    for key in selections_keys:
        model.apply_selection(config, key)

    if categories is None:
        models = model.load_models(name, date)
        features_imp = model.predict_models(models, name)
        model.plot_feature_importance(features_imp, "", name, date)
        model.mk_bdt_score(models)
        print(model.data)

    if categories is not None:
        models = model.load_models(name, date)
        category_list = categories.split(' ')
        models_per_cat = {k: {} for k in category_list}
        for key, value in models.items():
            for category in category_list:
                if category in key:
                    models_per_cat[category][key] = value
        
        data_copy = model.data.copy()
        N_cat = len(category_list)
        out_df = []
        df_index = []
        for i in range(N_cat):
            model.data = model.data[model.data[category_lable] == i]
            df_index.append(model.data.index.to_series())
            features_imp = model.predict_models(models_per_cat[category_list[i]], name)
            model.plot_feature_importance(features_imp, category_list[i], name, date)
            model.mk_bdt_score(models_per_cat[category_list[i]])
            out_df.append(model.data)
            print("End ", category_list[i])
            model.data = data_copy.copy()
        
        total_index = pd.concat(df_index, axis=0)
            
        model.data = out_df[0]
        branches_list = model.all_branches
        branches_list.append('bdt')
        branches_list.append('bdt_cv')

        for df in out_df[1:]: #MERGE DIFFERENT CATEGORIES
            model.data = pd.merge(model.data, df, on=list(branches_list), how='outer')

        model.data.index = total_index

        for i in range(number_of_splits): #MERGE SAME FOLD but DIFFERENT CATEGORIES branches 
            fold_name = f"fold_{i}_"
            model.data[fold_name] = model.data[fold_name+category_list[0]+'_']
            model.data = model.data.drop(fold_name+category_list[0]+'_', axis=1, errors='ignore')
            for j in range(1, N_cat):
                model.data[fold_name] = model.data[fold_name].combine_first(model.data[fold_name+category_list[j]+'_'])
                model.data = model.data.drop(fold_name+category_list[j]+'_', axis=1, errors='ignore')

        model.data = model.data.sort_index()
        print(model.data)

    fileName = output_path + "/" + date + "/"+name+"_minitree"
    model.data.to_csv(fileName+".csv", index=False)
    print("File CSV saved!")
    rdf = ROOT.RDF.MakeCsvDataFrame(fileName+".csv")
    rdf.Snapshot(out_tree_name, fileName+".root")
    print("File ROOT saved!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BDT training")

    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--categories", type=str, help="List of categories separated by ' ' ")

    args = parser.parse_args()
    
    config = args.config
    categories = args.categories
    
    with open(config, 'r') as file:
        json_file = json.load(file)
    data_path = json_file['data_path']
    files = json_file['files']
    name = json_file['Name']
    date = json_file['date']
        
    files_Run2022 = [data_path + i for i in files]
    
    predict_BDT_model(files_Run2022, name, config, date, categories)
    #Fa il train sui number_of_splits fold ma poi manca la parte dove vengono aggrgati (?)
    #Quindi bisogna creare un nuovo file .root con il nuovo branch
    #Implementare la divisione in categorie per tau3mu

