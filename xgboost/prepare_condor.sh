#!/bin/sh
# Usage:
#    prepare_condor.sh <Name_file> <N_files>

helpstring="Usage:
prepare_run.sh [config_file] [Categories]
\n-config_file: Name of the json configuration file
\n-Categories: List of categoreies, example: \"Cat_A Cat_B Cat_C\", or \"\" in case you don't need category
"
J_NAME=$1
CATEGORIES=$2

# Check inputs
if [ -z ${2+x} ]
then
echo -e ${helpstring}
return
fi

cmsenv

#!/bin/bash

if ! command -v jq &> /dev/null; then
    echo "jq non è installato. Installalo prima di continuare."
    exit 1
fi

file_json=$J_NAME

key="date"
date_value=$(date "+%Y%m%d-%H%M%S")
output_path=$(jq -r ".output_path" "$file_json")
name=$(jq -r ".Name" "$file_json")
file_json_new="${output_path}/${date_value}/${name}_config.json"
real_out="${output_path}/${date_value}"
mkdir -p "$real_out"
mkdir "${real_out}/log"
cp "$file_json" "$file_json_new"
jq --arg key "$key" --arg date_value "$date_value" '.[$key] = $date_value' "$file_json_new" > temp.json
mv temp.json "$file_json_new"

IFS=" " read -ra categ <<< "$CATEGORIES"

# Controlla se l'array categ è vuoto
if [ ${#categ[@]} -eq 0 ]; then
    cp ./templates/submit.condor ./$real_out
    sed -i "s#PATH#${real_out}#g" ./${real_out}/submit.condor
    number_of_splits=$(jq -r ".number_of_splits" "${file_json}")
    echo "queue ${number_of_splits}" >> "./${real_out}/submit.condor"
    chmod a+x ./${real_out}/submit.condor

    cp ./templates/launch_training.sh ./${real_out}
    sed -i "s#CXN#$file_json_new#g" ./${real_out}/launch_training.sh
    chmod a+x ./${real_out}/launch_training.sh
    echo "Files saved in ${real_out}"
else
    # Stampa gli elementi in un ciclo for
    for i in "${categ[@]}"; do
        index=${i}
        cp ./templates/submit.condor ./${real_out}/submit_${index}.condor
        sed -i "s/launch_training.sh/launch_${index}_training.sh/" ./${real_out}/submit_${index}.condor
        sed -i "s#PATH#${real_out}#g" ./${real_out}/submit_${index}.condor
        number_of_splits=$(jq -r ".number_of_splits" "${file_json}")
        echo "queue ${number_of_splits}" >> "./${real_out}/submit_${index}.condor"
        chmod a+x ./${real_out}/submit_${index}.condor

        cp ./templates/launch_training.sh ./${real_out}/launch_${index}_training.sh
        sed -i "s#CXN#$file_json_new#g" ./${real_out}/launch_${index}_training.sh
        sed -i "s/--condor/--condor --category ${index}/" ./${real_out}/launch_${index}_training.sh
        chmod a+x ./${real_out}/launch_${index}_training.sh
	echo "Files for category ${index} saved in ${real_out}"
    done
fi

echo "Completed successfully!"
