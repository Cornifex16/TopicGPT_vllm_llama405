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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, default="vllm_server", required=True)
    parser.add_argument("--model", type=str, default="llama-3.1-405b", required=True)
    parser.add_argument("--nombre", type=str, default="llama405_wiki", required=True)
    args = parser.parse_args()

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    load_dotenv()
    api = args.api
    modelo = args.model
    nombre = args.nombre
    inicio_time = time.time()

    generate_topic_lvl1_resume(
        "vllm_server", 
        modelo,
        "data/input/sample.jsonl",
        config["generation"]["prompt"],
        config["generation"]["seed"],
        "data/output/sample/R_generation_"+nombre+".jsonl",
        "data/output/sample/R_generation_"+nombre+".md",
        verbose=True,
        log_file=f"data/log_generation{nombre}.jsonl"
    )

    tiempo_ejecucion = time.time() - inicio_time
    print("tiempo: ", tiempo_ejecucion)

    inicio_time = time.time()

    refine_topics(
        "vllm_server",
        modelo,
        config["refinement"]["prompt"],
        "data/output/sample/R_generation_"+nombre+".jsonl",
        "data/output/sample/R_generation_"+nombre+".md",
        "data/output/sample/R_refinement_"+nombre+".md",
        "data/output/sample/R_refinement_"+nombre+".jsonl",
        verbose=True,
        remove=config["refinement"]["remove"],
        mapping_file="data/output/sample/R_mapping_"+nombre+".json",
        log_file=f"data/log_refinement{nombre}.jsonl"
    )
    tiempo_ejecucion = time.time() - inicio_time
    print("tiempo: ", tiempo_ejecucion)

    assign_topics_resume(
        "vllm_server",
        modelo,
        "data/input/sample.jsonl",
        "prompt/assignment alt.txt",
        "data/output/sample/chunks/R_assignment_"+nombre+".jsonl",
        "data/output/sample/R_refinement_"+nombre+".md",
        verbose=config["verbose"],
        log_file=f"data/log_assignment{nombre}.jsonl"
    )