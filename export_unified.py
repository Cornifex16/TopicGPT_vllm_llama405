"""
export_unified.py
-----------------
Autodiscovery de experimentos: lee output/ y logs/ sin configuracion manual.

Convenciones de nombres asumidas
----------------------------------
Output  (output/<dataset>/):
    R_{etapa}_{nombre_arbitrario}.jsonl
    etapa in {generation, refinement, asignacion}
    Ignora R_mapping_*, archivos .md y subcarpeta chunks/.

Logs  (logs/<dataset>/):
    generation / refinement:
        R_log_generation[vllm]_{nombre_arbitrario}.jsonl
        R_log_refinement[vllm]_{nombre_arbitrario}.jsonl
    assignment (multi-chunk):
        R_log_assignment_{nombre_arbitrario}{N}.jsonl
        donde N es el numero de chunk (opcional).

IMPORTANTE: "bills", "wiki" u otro nombre dentro del nombre de archivo
es parte del nombre arbitrario del usuario. El dataset lo determina
UNICAMENTE el nombre de la carpeta (output/bills, logs/wiki, etc.).

Uso:
    python export_unified.py
    python export_unified.py --base TopicGPT_vllm_llama405/data
    python export_unified.py --base data --out mis_resultados
"""

import re
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from topicgpt_python.metrics import metric_calc

# ---------------------------------------------------------------------------
# Columna ground-truth por dataset (agrega aqui nuevos datasets)
# ---------------------------------------------------------------------------
GT_COLS: dict[str, str] = {
    "bills": "topic",
    "wiki":  "category",
}
GT_DEFAULT = "topic"

# Columna de salida del modelo segun etapa
OUT_COL: dict[str, str] = {
    "generation":  "responses",
    "refinement":  "refined_responses",
    "asignacion":  "responses",
}

# Orden de presentacion de etapas en la tabla Markdown
ETAPA_ORDER = ["generation", "refinement", "asignacion"]

# ---------------------------------------------------------------------------
# Patrones de nombres de archivo
# ---------------------------------------------------------------------------

# output/<dataset>/R_{etapa}_{nombre_arbitrario}.jsonl
RE_OUTPUT = re.compile(
    r"^R_(generation|refinement|asignacion)_(.+)\.jsonl$",
    re.IGNORECASE,
)

# logs/<dataset>/R_log_generation[vllm]_{nombre_arbitrario}.jsonl
# logs/<dataset>/R_log_refinement[vllm]_{nombre_arbitrario}.jsonl
RE_LOG_GEN_REF = re.compile(
    r"^R_log_(generation|refinement)(?:vllm)?_(.+)\.jsonl$",
    re.IGNORECASE,
)

# logs/<dataset>/R_log_assignment_{nombre_arbitrario}{digitos_opcionales}.jsonl
RE_LOG_ASSIGN = re.compile(
    r"^R_log_assignment_(.+?)(\d+)?\.jsonl$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Autodiscovery
# ---------------------------------------------------------------------------

def discover_output_files(output_base: Path) -> list[dict]:
    """
    Escanea output/<dataset>/*.jsonl.
    Dataset   -> nombre de la carpeta (nunca del nombre del archivo).
    nombre_exp -> parte arbitraria tras la etapa en el nombre del archivo.
    Ignora R_mapping_*, archivos .md y subcarpetas (chunks/).
    """
    entries = []
    for dataset_dir in sorted(output_base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        col_gt  = GT_COLS.get(dataset.lower(), GT_DEFAULT)

        for f in sorted(dataset_dir.glob("*.jsonl")):
            if f.name.lower().startswith("r_mapping"):
                continue
            m = RE_OUTPUT.match(f.name)
            if not m:
                continue
            etapa      = m.group(1).lower()
            nombre_exp = m.group(2)  # nombre arbitrario completo, sin modificar

            entries.append({
                "dataset":    dataset,
                "etapa":      etapa,
                "nombre_exp": nombre_exp,
                "col_gt":     col_gt,
                "col_out":    OUT_COL.get(etapa, "responses"),
                "filepath":   f,
            })
    return entries


def discover_log_files(logs_base: Path) -> dict:
    """
    Escanea logs/<dataset>/*.jsonl y construye:
      (dataset, etapa_normalizada, nombre_exp_lower) -> [Path, ...]

    Dataset    -> nombre de la carpeta (nunca del nombre del archivo).
    nombre_exp -> parte arbitraria del nombre de log, sin numero de chunk.
    etapa_normalizada in {"generation", "refinement", "asignacion"}
    """
    index = {}

    for dataset_dir in sorted(logs_base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for f in sorted(dataset_dir.glob("*.jsonl")):
            name = f.name

            # generation / refinement
            m = RE_LOG_GEN_REF.match(name)
            if m:
                etapa_raw  = m.group(1).lower()
                nombre_exp = m.group(2)  # nombre arbitrario completo
                key = (dataset, etapa_raw, nombre_exp.lower())
                index.setdefault(key, []).append(f)
                continue

            # assignment (chunks)
            m = RE_LOG_ASSIGN.match(name)
            if m:
                nombre_exp = m.group(1)  # nombre sin numero de chunk
                key = (dataset, "asignacion", nombre_exp.lower())
                index.setdefault(key, []).append(f)
                continue

    return index


# ---------------------------------------------------------------------------
# Matching output <-> logs
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set:
    """Divide por '_' o '-', filtra tokens de mas de 2 caracteres."""
    return {t for t in re.split(r"[_\-]", s.lower()) if len(t) > 2}


def match_logs(entry: dict, log_index: dict) -> list:
    """
    Cruza un entry de output con los logs del mismo dataset y etapa.

    nombre_exp en output y logs son nombres arbitrarios del usuario;
    no se asume ninguna estructura interna. Matching en tres niveles:
      1. Exacto (lower)
      2. Uno es prefijo del otro
      3. Mayor solapamiento de tokens separados por '_' o '-' (> 2 chars)
    """
    dataset  = entry["dataset"]
    etapa    = entry["etapa"]
    out_name = entry["nombre_exp"].lower()

    candidates = {
        k: v for k, v in log_index.items()
        if k[0].lower() == dataset.lower() and k[1] == etapa
    }
    if not candidates:
        return []

    # 1. Exacto
    for (_, __, log_name), files in candidates.items():
        if log_name == out_name:
            return files

    # 2. Prefijo
    for (_, __, log_name), files in candidates.items():
        if out_name.startswith(log_name) or log_name.startswith(out_name):
            return files

    # 3. Mayor solapamiento de tokens
    out_tok = _tokens(out_name)
    best_files, best_n = [], 0
    for (_, __, log_name), files in candidates.items():
        n = len(out_tok & _tokens(log_name))
        if n > best_n:
            best_n, best_files = n, files

    return best_files if best_n > 0 else []


# ---------------------------------------------------------------------------
# Parseo de logs
# ---------------------------------------------------------------------------

TS_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]

def _parse_ts(s: str):
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def parse_log_files(log_files: list) -> dict:
    """
    Lee uno o mas archivos de log y devuelve estadisticas agregadas.
    Para asignacion, todos los chunks se tratan como una unica ejecucion.
    """
    all_ts  = []
    t_in    = 0
    t_out   = 0
    n_calls = 0
    models  = set()

    for lf in log_files:
        with open(lf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue

                usage = log.get("raw_completion", {}).get("usage", {})
                t_in  += usage.get("prompt_tokens",    0)
                t_out += usage.get("completion_tokens", 0)
                n_calls += 1

                ts = _parse_ts(log.get("timestamp", ""))
                if ts:
                    all_ts.append(ts)

                model = log.get("model")
                if model:
                    models.add(model)

    all_ts.sort()
    elapsed = (all_ts[-1] - all_ts[0]).total_seconds() if len(all_ts) > 1 else 0.0

    return {
        "total_time_seconds":  round(elapsed, 2),
        "total_input_tokens":  t_in,
        "total_output_tokens": t_out,
        "num_calls":           n_calls,
        "models":              ", ".join(sorted(models)),
        "log_files_found":     len(log_files),
    }


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def run_metrics(filepath: Path, col_gt: str, col_out: str):
    if not filepath.exists():
        print(f"    [SKIP] no existe: {filepath}")
        return None, None, None
    try:
        return metric_calc(str(filepath), col_gt, col_out)
    except Exception as e:
        print(f"    [ERROR] metric_calc: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Formato de salida
# ---------------------------------------------------------------------------

def fmt_f(v, d=8):
    return f"{v:.{d}f}" if v is not None else ""

def fmt_i(v):
    return f"{v:,}" if v is not None else ""

MD_HEADER = (
    "| Modelo | Etapa | P1 | ARI | MIS "
    "| Tiempo (s) | Tokens In | Tokens Out | Llamadas | Chunks log |"
)
MD_SEP = (
    "| ------ | ----- | -- | --- | --- "
    "| ---------- | --------- | ---------- | -------- | ---------- |"
)

def row_md(r: dict) -> str:
    return (
        f"| {r['nombre_exp']} | {r['etapa']} "
        f"| {fmt_f(r['p1'])} | {fmt_f(r['ari'])} | {fmt_f(r['mis'])} "
        f"| {r['total_time_seconds']} "
        f"| {fmt_i(r['total_input_tokens'])} "
        f"| {fmt_i(r['total_output_tokens'])} "
        f"| {r['num_calls']} | {r['log_files_found']} |"
    )

def make_md_table(rows: list) -> str:
    return "\n".join([MD_HEADER, MD_SEP] + [row_md(r) for r in rows])

def to_csv(all_rows: list) -> str:
    cols = [
        "dataset", "etapa", "nombre_exp",
        "p1", "ari", "mis",
        "total_time_seconds", "total_input_tokens", "total_output_tokens",
        "num_calls", "log_files_found", "models",
    ]
    lines = [",".join(cols)]
    for r in all_rows:
        def cell(c):
            v = r.get(c, "")
            if c in ("p1", "ari", "mis"):
                return fmt_f(v)
            if isinstance(v, str):
                return f'"{v}"'
            return str(v)
        lines.append(",".join(cell(c) for c in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base: str, out_stem: str):
    output_base = Path(base) / "output"
    logs_base   = Path(base) / "logs"

    print("Escaneando output/...")
    out_entries = discover_output_files(output_base)
    print(f"  {len(out_entries)} archivos de output encontrados.")

    print("Escaneando logs/...")
    log_index = discover_log_files(logs_base)
    print("Este es el log_index: ", log_index)
    total_log_keys = sum(len(v) for v in log_index.values())
    print(f"  {len(log_index)} claves de log, {total_log_keys} archivos en total.")

    # Agrupar por dataset para las secciones del Markdown
    sections = {}   # (dataset, etapa) -> [row, ...]
    all_rows = []

    for entry in out_entries:
        dataset    = entry["dataset"]
        etapa      = entry["etapa"]
        nombre_exp = entry["nombre_exp"]

        print(f"\n  [{dataset.upper()}] {etapa} | {nombre_exp}")

        # Metricas
        p1, ari, mis = run_metrics(entry["filepath"], entry["col_gt"], entry["col_out"])

        # Logs
        log_files = match_logs(entry, log_index)
        if not log_files:
            print(f"    [WARN] Sin logs encontrados.")
        else:
            print(f"    Logs ({len(log_files)}): {[f.name for f in log_files]}")
        log_stats = parse_log_files(log_files)

        row = {
            "dataset":    dataset,
            "etapa":      etapa,
            "nombre_exp": nombre_exp,
            "p1": p1, "ari": ari, "mis": mis,
            **log_stats,
        }
        key = (dataset, etapa)
        sections.setdefault(key, []).append(row)
        all_rows.append(row)

    # Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_parts = [f"# Resultados Unificados\n\n_Generado: {now}_\n"]

    for ds in sorted({k[0] for k in sections}):
        for etapa in ETAPA_ORDER:
            key = (ds, etapa)
            if key not in sections:
                continue
            titulo = f"{ds.capitalize()} - {etapa.capitalize()}"
            md_parts.append(f"## {titulo}\n\n{make_md_table(sections[key])}\n")

    md_path = f"{out_stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_parts))
    print(f"\nMarkdown -> {md_path}")

    # CSV
    csv_path = f"{out_stem}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(to_csv(all_rows))
    print(f"CSV      -> {csv_path}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Genera tabla unificada de metricas + logs (autodiscovery)."
    )
    parser.add_argument(
        "--base", default="data",
        help="Directorio raiz con output/ y logs/ (default: data)"
    )
    parser.add_argument(
        "--out", default="R_resultados_unificados",
        help="Nombre base de salida sin extension (default: resultados_unificados)"
    )
    args = parser.parse_args()
    main(args.base, args.out)