# TopicGPT Alterado

Este repositorio contiene codigo para la generación, refinamiento y asignación de tópicos utilizando Large Language Models (LLMs). Estos se ejecutan mediante dos scripts de terminal.

## Requisitos Previos

Antes de iniciar, el sistema requiere configurar las variables de entorno para las APIs. 

1. Crea un archivo `.env` en la raíz del proyecto.
2. Añade las siguientes variables para usar VLLM (ajusta los valores según corresponda):

```env
VLLM_BASE_URL="http://<ip-del-servidor>:8000/v1"
VLLM_API_KEY="tu-api-key"
```

**Opcionalmente:** en caso de querer usar otro proveedor de la siguiente lista usar su respectiva api key.

```env
OPENAI_API_KEY="tu-api-key"
OPENROUTER_API_KEY="tu-api-key"
DASHSCOPE_API_KEY="tu-api-key"
FIREWORKS_API_KEY="tu-api-key"
NVIDIA_API_KEY="tu-api-key"
```

## Configuración del Entorno (Setup)

Para instalar las dependencias, crear el entorno virtual y preparar los datos, ejecuta el script de setup. **Solo necesitas correr esto una vez.**

```bash
chmod +x setup.sh
./setup.sh
```

## Ejecución

El pipeline completo se ejecuta mediante `ejecucion.sh`. Este script es configurable y acepta 4 parámetros:
1. `MODELO`: El identificador exacto del modelo (ej. `meta-llama/Llama-3.1-405B-Instruct`).
2. `API`: El tipo de proveedor (ej. `vllm_server`, `openrouter`, `dashscope`).
3. `DATASET`: El dataset a usar para el script (ej. `wiki`, `bills`).
4. `NOMBRE`: Un identificador corto para nombrar los archivos de salida (ej. `llama405_wiki`).

Estos parametros siguen el siguiente listado de opciones disponibles:

`API`:
- `openai`
- `openrouter`
- `dashscope`
- `fireworks`
- `nvidia`
- `vllm_server`

`DATASET`:
- `wiki`
- `bills`

El usuario es libre de ingresar mas provedorees dentro del codigo y mas datasets, asegurar de mantener el orden de las carpetas y los nombres adecuados.
```text
TopicGPT_vllm_llama405/
└── data/
    ├── input/
    │   ├── bills/
    │   ├── wiki/
    │   └── dataset/
    │       ├── gen_set.jsonl
    │       └── asig_set.jsonl
    ├── logs/
    │   ├── bills/
    │   ├── wiki/
    │   └── dataset/
    └── output/
        ├── bills/
        ├── wiki/
        └── dataset/
```

## Reanudar desde cualquier etapa

Hay otros dos scripts `ejecucion_refinamiento.sh` y `ejecucion_asignacion.sh` que continuan el pipeline desde los procesos de refinamiento y asignacion en adelante. Usan los mismo parametros que `ejecucion.sh` asi que mantener la consistencia al ejecutar.

### Comando de Ejecución de Ejemplo:

```bash
chmod +x ejecucion.sh
./ejecucion.sh "vllm_server" "meta-llama/Llama-3.1-405B-Instruct" "wiki" "llama405"
```

## Estructura de Salida (Outputs)

Una vez que el proceso termine, los resultados se guardarán en la carpeta `data/output/<dataset>` en su respectivo dataset. hay que verificar la existencia de estos archivos para confirmar el éxito:
* `R_generation_<nombre>.jsonl` -> Salida de la generacion.
* `R_refinement_<nombre>.jsonl` -> Salida del refinamiento.
* `R_asignacion_<nombre>.jsonl` -> Salida de la asignacion.