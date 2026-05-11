from topicgpt_python import *
import yaml
from dotenv import load_dotenv
from topicgpt_python.generation_1 import generate_topic_lvl1_resume
from topicgpt_python.assignment import assign_topics_resume
import time
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def assignment_thread(modelo, nombre, api, dataset, offset_num):
    inicio_time = time.time()
    assign_topics_resume(
        api,
        modelo,
        f"data/input/{dataset}/{dataset}_chunks/asig_set_"+str(offset_num)+".jsonl",
        "prompt/assignment alt.txt",
        f"data/output/{dataset}/chunks/R_assignment_"+nombre+"_N"+str(offset_num)+".jsonl",
        f"data/output/{dataset}/R_refinement_"+nombre+".md",
        verbose=config["verbose"],
        log_file=f"data/logs/{dataset}/R_log_assignment_{offset_num}.jsonl"
    )
    tiempo_ejecucion = time.time() - inicio_time
    print("tiempo: ", tiempo_ejecucion)

    archivo_tiempo = "data/tiempo.json"
    if os.path.exists(archivo_tiempo) and os.path.getsize(archivo_tiempo) > 0:
        with open(archivo_tiempo, "r") as archivo:
            datos = json.load(archivo)
    else:
        datos = {}
    if nombre + "_" + dataset + " asignacion" +str(offset_num) in datos:
        print("el modelo existe")
    else:
        print("el modelo no existe")
    datos[nombre + "_" + dataset + " asignacion"+str(offset_num)] = tiempo_ejecucion
    with open(archivo_tiempo, "w") as archivo:
        json.dump(datos, archivo, indent=4)
    return tiempo_ejecucion, offset_num

def data_to_chunks(input_file, num_chunks, dataset):
    df = pd.read_json(input_file, lines=True)
    largo = len(df)
    chunk_size = int(largo / num_chunks)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    print(len(chunks))
    print(len(chunks[0]))
    for i in range(len(chunks)):
        df_chunk = chunks[i]
        df_chunk.to_json(f"data/input/{dataset}/{dataset}_chunks/asig_set_"+str(i)+".jsonl", lines=True, orient="records")

def merge_chunks(input_directory, nombre, output_file):
    with open(output_file, "w") as archivo:
        for file_name in os.listdir(input_directory):
            if file_name.endswith(".jsonl") and nombre in file_name:
                file_path = os.path.join(input_directory, file_name)
                with open(file_path, "r") as archivo_input:
                    for line in archivo_input:
                        archivo.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, default="vllm_server", required=True)
    parser.add_argument("--dataset", type=str, default="wiki", required=True)
    parser.add_argument("--model", type=str, default="llama-3.1-405b", required=True)
    parser.add_argument("--nombre", type=str, default="llama405_wiki", required=True)
    args = parser.parse_args()
    load_dotenv()
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    # ahora se aplica lo de threads 
    api = args.api
    dataset = args.dataset
    modelo = args.model
    nombre = args.nombre
    data_to_chunks(
        f"data/input/{dataset}/asig_set.jsonl",
        14,
        dataset
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i in range(15):
            futures[executor.submit(assignment_thread, modelo, nombre, api, dataset, i)] = i
        tiempo_total = 0
        for future in as_completed(futures):
            data_temp = futures[future]
            try:
                data = future.result()
            except Exception as e:
                print(f'chunk {str(data_temp)} a generado un error: {e}')
            else:
                print(f'chunk {str(data_temp)} se genero en este tiempo: {str(data[0])}')
                tiempo_total += data[0]
    merge_chunks(
        f"data/output/{dataset}/chunks",
        nombre,
        f"data/output/{dataset}/R_asignacion_"+nombre+".jsonl"
    )