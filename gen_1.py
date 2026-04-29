from topicgpt_python import *
import yaml
from dotenv import load_dotenv
from topicgpt_python.generation_1 import generate_topic_lvl1_resume
from topicgpt_python.assignment import assign_topics_resume
import time
import json
import os

# Poner TODO el código de ejecución debajo de esta línea
if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    load_dotenv()

    modelo = "meta-llama/Llama-3.1-405B-Instruct"
    nombre = "vllm_llama405_wiki"
    inicio_time = time.time()

    generate_topic_lvl1_resume(
        "vllm", 
        modelo,
        "data/input/wiki_subset.jsonl",
        config["generation"]["prompt"],
        config["generation"]["seed"],
        "data/output/wiki/R_generation_"+nombre+".jsonl",
        "data/output/wiki/R_generation_"+nombre+".md",
        verbose=True
    )

    tiempo_ejecucion = time.time() - inicio_time
    print("tiempo: ", tiempo_ejecucion)

    archivo_tiempo = "tiempo.json"
    if os.path.exists(archivo_tiempo) and os.path.getsize(archivo_tiempo) > 0:
        with open(archivo_tiempo, "r") as archivo:
            datos = json.load(archivo)
    else:
        datos = {}

    # alterar el nombre segun etapa
    if nombre + " generacion" in datos:
        print("el modelo existe")
    else:
        print("el modelo no existe")

    # alterar el nombre segun etapa
    datos[nombre + " generacion"] = tiempo_ejecucion
    with open(archivo_tiempo, "w") as archivo:
        json.dump(datos, archivo, indent=4)