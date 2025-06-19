"""Script para GERAÇÃO de respostas JSON de modelos MULTIMODAIS GEMINI.

Este script itera sobre um dataset de imóveis, processa as imagens associadas
a cada imóvel utilizando um modelo Gemini configurado, e envia prompts
específicos (sistema e humano) para o modelo.
O objetivo é que o modelo retorne uma análise estruturada em formato JSON.

Os resultados, incluindo a string JSON crua retornada pelo modelo, o objeto JSON
parseado (se o parsing for bem-sucedido), e metadados sobre o processamento
(latências, erros, etc.), são salvos em um arquivo JSON consolidado por configuração.

A avaliação das métricas é feita por um script separado.
"""

import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np # Usado na função lambda para json.dump default
import sys
import unicodedata # Para clean_json_string
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm # Para barra de progresso visual
from loguru import logger # Para logging flexível

# Tentativa de importar módulos utilitários e o conector Gemini
try:
    from utils.helpers import (
        get_images_from_directory,
    )
    from gemini_connector import (
        gemini_generate,
        list_available_gemini_models, # Pode ser uma lista estática ou heurística
        validate_gemini_model # Pode ser uma lista estática ou heurística
    )

except ImportError as e:
    logger.critical(f"Erro de importação: {e}. Verifique gemini_connector.py e utils/helpers.py.")
    raise SystemExit(1)

load_dotenv() # Carrega GOOGLE_API_KEY de .env se gemini_connector não o fizer explicitamente

# --- Configuração ---
# Lista de modelos Gemini multimodais a serem testados, separados por vírgula
MODEL_NAMES_TO_TEST = os.getenv("GEMINI_BENCHMARK_MODELS", "gemini-2.5-flash,gemini-2.5-pro").split(',')
# Temperaturas para testar com cada modelo
TEMPERATURES_TO_TEST = [float(t) for t in os.getenv("BENCHMARK_TEMPERATURES", "0.1,0.5,1.0").split(',')]

DATASET_PATH = Path(os.getenv("BENCHMARK_DATASET_PATH", "./data/benchmark_50_anotacao_gemini.csv"))
IMAGES_BASE_DIR = Path(os.getenv("BENCHMARK_IMAGES_DIR", "./data/images"))
OUTPUT_BASE_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "./resultados_geracao_gemini")) # Pasta de saída
PROMPTS_DIR = Path(os.getenv("BENCHMARK_PROMPTS_DIR", "./prompts"))
TIPO_PROMPT_LABEL = "analise_imovel_gemini_v1.0" # Novo rótulo de prompt se for diferente

SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt_gemini.txt" # Pode ser o mesmo system prompt
HUMAN_PROMPT_FILE = PROMPTS_DIR / "human_prompt_gemini.txt"   # Pode ser o mesmo human prompt

LOG_FILE_PATH = "logs/generation_gemini.log" # Novo arquivo de log

# --- Configurar Logging, Carregar Prompts, Funções Auxiliares ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH, rotation="10 MB", retention="7 days", level="DEBUG", encoding="utf-8")

SYSTEM_PROMPT_CONTENT = ""
HUMAN_PROMPT_CONTENT = ""
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f: SYSTEM_PROMPT_CONTENT = f.read()
    with open(HUMAN_PROMPT_FILE, 'r', encoding='utf-8') as f: HUMAN_PROMPT_CONTENT = f.read()
    logger.info(f"System prompt carregado de: {SYSTEM_PROMPT_FILE}")
    logger.info(f"Human prompt carregado de: {HUMAN_PROMPT_FILE}")
    if not SYSTEM_PROMPT_CONTENT or not HUMAN_PROMPT_CONTENT: raise ValueError("Prompts vazios.")
except Exception as e: logger.critical(f"Erro ao ler prompts: {e}"); raise SystemExit(f"Erro crítico ao ler prompts: {e}")

def clean_json_string(json_str: str) -> str:

    cleaned_chars = []
    for char_code in map(ord, json_str):
        if (32 <= char_code <= 126) or char_code in [9, 10, 13, 12, 8] or char_code > 127:
            cleaned_chars.append(chr(char_code))
        elif chr(char_code) == '\\': cleaned_chars.append('\\')
    return "".join(cleaned_chars)

def extract_json_from_response(raw_response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    if not raw_response_text: return None, "Resposta do modelo vazia."
    json_str_candidate = None
    match_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', raw_response_text, re.DOTALL)
    if match_block: json_str_candidate = match_block.group(1)
    else:
        first_brace = raw_response_text.find('{')
        last_brace = raw_response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str_candidate = raw_response_text[first_brace : last_brace+1]
        else: return None, "Nenhum padrão JSON (bloco ou chaves) encontrado."
    if not json_str_candidate: return None, "Nenhum candidato a string JSON."
    cleaned_json_str = clean_json_string(json_str_candidate)
    try:
        return json.loads(cleaned_json_str), None
    except json.JSONDecodeError as e:
        return None, f"Falha parse JSON (limpa): {e}. Str: '{cleaned_json_str[:200]}...'"

def find_property_image_dir(base_dir: Path, property_id_search_term: str) -> Optional[Path]:

    patterns_to_try = [property_id_search_term, f"{property_id_search_term}_*"]
    for pattern_str in patterns_to_try:
        try:
            matches = list(base_dir.glob(pattern_str))
            dir_matches = [m for m in matches if m.is_dir()]
            if dir_matches:
                if len(dir_matches) > 1: logger.warning(f"Múltiplos dirs para '{property_id_search_term}' '{pattern_str}'. Usando {dir_matches[0]}")
                return dir_matches[0]
        except Exception as e: logger.error(f"Erro find_property_image_dir: {e}"); continue
    logger.warning(f"Nenhum dir para '{property_id_search_term}' em '{base_dir}' padrões {patterns_to_try}.")
    return None


def process_property_with_gemini(
    property_dir_path: Path,
    current_model_name: str,
    current_temperature: float,
    # max_output_tokens: Optional[int] = 4096
) -> Dict[str, Any]:
    """
    Processa imagens de uma propriedade com um modelo Gemini e temperatura específicos.

    Args:
        property_dir_path: Path para o diretório das imagens da propriedade.
        current_model_name: Nome do modelo Gemini a ser usado.
        current_temperature: Temperatura para a geração.
        # max_output_tokens: (Opcional) Máximo de tokens de saída.

    Returns:
        Um dicionário contendo detalhes do processamento.
    """
    process_start_time = time.time()
    base_result = {
        "image_paths_processed_full": [], "image_filenames_processed": [],
        "num_images_found_in_dir": 0, "num_images_sent_to_api": 0,
        "raw_model_output": None, "parsed_json_object": None,
        "json_parsing_error": None,
        "api_response_time_s": None, # Vem direto do gemini_connector
        "api_usage_metadata": None, # Vem direto do gemini_connector
        "function_total_time_s": 0.0,
        "success_api_call": False,
        "success_json_parsing": False,
        "error_message_processing": None
    }

    try:
        # 1. Listar imagens do diretório
        image_paths_in_dir = get_images_from_directory(str(property_dir_path)) # de utils.helpers
        base_result["num_images_found_in_dir"] = len(image_paths_in_dir)
        base_result["image_paths_processed_full"] = image_paths_in_dir # Todos os caminhos encontrados
        base_result["image_filenames_processed"] = [Path(p).name for p in image_paths_in_dir]

        if not image_paths_in_dir:
            base_result["error_message_processing"] = "Nenhuma imagem encontrada no diretório para processar."
            logger.warning(f"{base_result['error_message_processing']} para {property_dir_path}")
            base_result["function_total_time_s"] = time.time() - process_start_time
            return base_result
        
        base_result["num_images_sent_to_api"] = len(image_paths_in_dir)

        # 2. Construir `prompt_parts` para gemini_generate
        prompt_parts_for_gemini: List[Union[str, Dict[str, Any]]] = []
        prompt_parts_for_gemini.append(HUMAN_PROMPT_CONTENT) # O prompt de texto principal
        for img_path in image_paths_in_dir:
            prompt_parts_for_gemini.append({"type": "image_path", "image_path": img_path})

        logger.debug(f"Enviando {len(image_paths_in_dir)} imagens de {property_dir_path} para o modelo {current_model_name} com temp: {current_temperature}")
        
        # 3. Chamar gemini_generate
        api_response_data = gemini_generate(
            prompt_parts=prompt_parts_for_gemini,
            system_prompt=SYSTEM_PROMPT_CONTENT,
            model_name=current_model_name,
            temperature=current_temperature,
            # max_output_tokens=max_output_tokens # Passar se definido
        )

        base_result["api_response_time_s"] = api_response_data.get("response_time_s")
        base_result["api_usage_metadata"] = api_response_data.get("usage_metadata")

        if api_response_data.get("error"):
            base_result["error_message_processing"] = f"API Gemini: {api_response_data['error']}"
            base_result["raw_model_output"] = api_response_data.get("response") # Pode haver resposta parcial ou erro
            logger.error(f"API Gemini retornou erro para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}): {api_response_data['error']}")
        else:
            base_result["success_api_call"] = True
            base_result["raw_model_output"] = api_response_data.get("response")

            if base_result["raw_model_output"]:
                parsed_json, parsing_error = extract_json_from_response(base_result["raw_model_output"])
                base_result["parsed_json_object"] = parsed_json
                base_result["json_parsing_error"] = parsing_error
                if parsed_json and not parsing_error:
                    base_result["success_json_parsing"] = True
                    logger.info(f"JSON parseado com sucesso para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}).")
                else:
                    logger.warning(f"Falha no parsing do JSON para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}). Detalhes do erro já logados.")
            else:
                base_result["error_message_processing"] = "API Gemini sucesso, mas 'response' está vazio."
                base_result["json_parsing_error"] = base_result["error_message_processing"]
                logger.warning(f"{base_result['error_message_processing']} (Dir: {property_dir_path}, Modelo: {current_model_name}, Temp: {current_temperature})")
    
    except ValueError as ve: # Ex: de get_images_from_directory
        base_result["error_message_processing"] = f"Erro ao preparar dados: {ve}"
        logger.error(f"{base_result['error_message_processing']} para {property_dir_path}")
    except Exception as e:
        base_result["error_message_processing"] = f"Exceção em process_property_with_gemini: {type(e).__name__} - {e}"
        logger.exception(f"Exceção em process_property_with_gemini para {property_dir_path} (modelo {current_model_name}, temp {current_temperature})")

    base_result["function_total_time_s"] = time.time() - process_start_time
    return base_result


# --- Função Principal de Geração para Múltiplas Configurações com Gemini ---
def run_multiconfig_gemini_generation() -> None:
    overall_start_time_iso = datetime.now(timezone.utc)

    logger.info(f"Iniciando execução multi-configuração GEMINI em: {overall_start_time_iso.isoformat()}")
    logger.info(f"Modelos Gemini a serem testados: {MODEL_NAMES_TO_TEST}")
    logger.info(f"Temperaturas a serem testadas: {TEMPERATURES_TO_TEST}")

    if not DATASET_PATH.is_file(): logger.critical(f"Dataset não encontrado: {DATASET_PATH}"); raise SystemExit(1)
    if not IMAGES_BASE_DIR.is_dir(): logger.critical(f"Dir imagens não encontrado: {IMAGES_BASE_DIR}"); raise SystemExit(1)

    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset carregado de '{DATASET_PATH}' com {len(df_dataset)} propriedades.")
    except Exception as e: logger.critical(f"Erro ao carregar dataset: {e}"); raise SystemExit(1)

    for model_name_current in MODEL_NAMES_TO_TEST:
        model_name_current = model_name_current.strip()
        if not model_name_current: continue

        logger.info(f"--- Iniciando testes para o MODELO GEMINI: {model_name_current} ---")

        for temp_current in TEMPERATURES_TO_TEST:
            logger.info(f"--- Iniciando geração para MODELO: {model_name_current}, TEMPERATURA: {temp_current} ---")

            generation_start_time_iso = datetime.now(timezone.utc)
            
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = model_name_current.replace(':', '_').replace('/', '_').replace('.', 'p')
            temp_safe = str(temp_current).replace('.', 'p')

            current_run_output_dir = OUTPUT_BASE_DIR / f"output_{model_name_safe}_temp_{temp_safe}_{run_timestamp}"
            current_run_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Resultados para {model_name_current} @ temp {temp_current} serão salvos em: {current_run_output_dir}")

            generation_results_path = current_run_output_dir / "generation_results.json"
            lista_de_resultados_formatados_run = []

            for idx, row_dataset in tqdm(df_dataset.iterrows(), total=len(df_dataset), desc=f"Modelo {model_name_current}, Temp {temp_current}"):
                current_property_id_output = idx + 1
                property_folder_search_term = str(row_dataset.get('property_folder_id', idx + 1))
                property_dir_path = find_property_image_dir(IMAGES_BASE_DIR, property_folder_search_term)
                
                resultado_item: Dict[str, Any] = {
                    "property_id": current_property_id_output, "dataset_idx": idx,
                    "modelo": model_name_current, "temperatura_usada": temp_current,
                    "search_term_folder": property_folder_search_term,
                    "property_directory_processed": None, # Será preenchido se encontrado
                }

                if not property_dir_path:
                    logger.warning(f"Idx {idx} (Modelo {model_name_current}, Temp {temp_current}): dir não encontrado para '{property_folder_search_term}'.")
                    resultado_item["status_geracao"] = "erro_diretorio_nao_encontrado"
                    resultado_item["error_message"] = f"Diretório para '{property_folder_search_term}' não encontrado."
                else:
                    resultado_item["property_directory_processed"] = str(property_dir_path)
                    processing_details = process_property_with_gemini(
                        property_dir_path,
                        current_model_name=model_name_current,
                        current_temperature=temp_current
                    )
                    resultado_item.update({
                        "num_imagens_encontradas": processing_details.get("num_images_found_in_dir", 0),
                        "num_imagens_enviadas_api": processing_details.get("num_images_sent_to_api", 0),
                        "tipo_prompt": TIPO_PROMPT_LABEL,
                        "api_usage_metadata": processing_details.get("api_usage_metadata"),
                        "latencia_api_s": processing_details.get("api_response_time_s"),
                        "latencia_funcao_processamento_s": processing_details.get("function_total_time_s"),
                        "resposta_modelo_raw_str": processing_details.get("raw_model_output"),
                        "resposta_modelo_parsed_dict": processing_details.get("parsed_json_object"),
                        "status_api_call": "sucesso" if processing_details.get("success_api_call") else "falha",
                        "status_json_parsing": "sucesso" if processing_details.get("success_json_parsing") else "falha",
                        "json_parsing_error_msg": processing_details.get("json_parsing_error"),
                        "processing_error_msg": processing_details.get("error_message_processing"),
                        "imagens_processadas_nomes": processing_details.get("image_filenames_processed", []),
                    })
                    if not processing_details.get("success_api_call"):
                        resultado_item["status_geracao"] = "erro_api"
                    elif not processing_details.get("success_json_parsing"):
                        resultado_item["status_geracao"] = "erro_parsing_json"
                    else:
                        resultado_item["status_geracao"] = "processado_com_sucesso"
                
                lista_de_resultados_formatados_run.append(resultado_item)

            output_final_geracao_run = {
                "metadata_geracao_run": {
                    "timestamp_geracao_inicial_run": generation_start_time_iso.isoformat(),
                    "timestamp_geracao_final_run": datetime.now(timezone.utc).isoformat(),
                    "timestamp_total_run": (datetime.now(timezone.utc) - generation_start_time_iso).total_seconds(),
                    "modelo_usado": model_name_current,
                    "temperatura_usada": temp_current,
                    "total_propriedades_dataset": len(df_dataset),
                    "total_itens_na_saida_run": len(lista_de_resultados_formatados_run),
                    "dataset_usado": str(DATASET_PATH.name),
                    "tipo_prompt_usado": TIPO_PROMPT_LABEL,
                    "system_prompt_file": str(SYSTEM_PROMPT_FILE.name),
                    "human_prompt_file": str(HUMAN_PROMPT_FILE.name),
                },
                "resultados_gerados_run": lista_de_resultados_formatados_run
            }
            try:
                with open(generation_results_path, 'w', encoding='utf-8') as f_out:
                    json.dump(output_final_geracao_run, f_out, indent=2, ensure_ascii=False,
                              default=lambda o: int(o) if isinstance(o, np.integer) else
                                                float(o) if isinstance(o, np.floating) else
                                                o.tolist() if isinstance(o, np.ndarray) else str(o))
                logger.info(f"Resultados para {model_name_current} @ temp {temp_current} salvos em: {generation_results_path}")
            except Exception as e_json_save:
                logger.error(f"Erro ao salvar JSON para {model_name_current} @ temp {temp_current}: {e_json_save}")
            logger.info(f"--- Finalizada geração para MODELO GEMINI: {model_name_current}, TEMPERATURA: {temp_current} ---")
        logger.info(f"--- Finalizados todos os testes de temperatura para o MODELO GEMINI: {model_name_current} ---")
    logger.info(f"Execução multi-configuração GEMINI finalizada em: {datetime.now(timezone.utc).isoformat()}")

# --- Bloco de Testes de Diagnóstico ---
def run_diagnostic_tests_simplified(model_to_test: str, temp_to_test: float, num_properties_to_test: int = 1):
    logger.info("="*20 + f" INICIANDO TESTES DE DIAGNÓSTICO (MODELO GEMINI: {model_to_test}, TEMP: {temp_to_test}) " + "="*20)
    
    logger.info(f"--- Teste 1: Validação (Heurística) do Modelo Gemini {model_to_test} ---")
    if not validate_gemini_model(model_to_test): # Função do gemini_connector
        logger.warning(f"Modelo de teste '{model_to_test}' não passou na validação heurística. Testes podem prosseguir com cautela.")
    else:
        logger.info(f"Modelo de teste '{model_to_test}' parece válido.")

    logger.info(f"--- Teste 2: Dataset e Pastas (Primeiras {num_properties_to_test} props) ---")
    if not DATASET_PATH.is_file(): logger.error(f"Dataset não encontrado {DATASET_PATH}"); return
    try:
        df_test = pd.read_csv(DATASET_PATH, nrows=num_properties_to_test)
        for idx, row_test in df_test.iterrows():
            prop_id_search = str(row_test.get('property_folder_id', idx + 1))
            test_dir = find_property_image_dir(IMAGES_BASE_DIR, prop_id_search)
            if test_dir:
                logger.info(f"Prop (busca {prop_id_search}, idx {idx}): Dir {test_dir}")
                # Apenas listar, não precisa codificar para Gemini aqui nos diagnósticos
                try:
                    imgs_in_dir = get_images_from_directory(str(test_dir))
                    logger.info(f"  Encontradas {len(imgs_in_dir)} imagens. Primeira: {imgs_in_dir[0] if imgs_in_dir else 'N/A'}")
                except Exception as e_get_imgs:
                    logger.error(f"  Erro ao listar imagens em {test_dir}: {e_get_imgs}")
            else:
                logger.warning(f"Prop (busca {prop_id_search}, idx {idx}): Dir NÃO encontrado.")
    except Exception as e: logger.error(f"Erro teste dataset/pastas: {e}")

    logger.info(f"--- Teste 3: Processamento de Uma Propriedade com {model_to_test} @ Temp {temp_to_test} ---")
    first_valid_dir = None
    try:
        df_s_test = pd.read_csv(DATASET_PATH, nrows=min(5, len(pd.read_csv(DATASET_PATH)) if DATASET_PATH.is_file() else 5))
        for idx_s, row_s in df_s_test.iterrows():
            prop_id_s = str(row_s.get('property_folder_id', idx_s + 1))
            dir_p = find_property_image_dir(IMAGES_BASE_DIR, prop_id_s)
            if dir_p:
                try: # Checar se há imagens
                    if get_images_from_directory(str(dir_p)): first_valid_dir = dir_p; break
                except: pass # Ignora erro de get_images_from_directory aqui
        
        if first_valid_dir:
            logger.info(f"Testando process_property_with_gemini com: {first_valid_dir}, Modelo: {model_to_test}, Temp: {temp_to_test}")
            res_s = process_property_with_gemini(first_valid_dir, model_to_test, temp_to_test)
            logger.info(f"Resultado (API: {res_s.get('success_api_call')}, JSON Parse: {res_s.get('success_json_parsing')}):")
            logger.info(f"  JSON Obj (preview): {str(res_s.get('parsed_json_object'))[:200] if res_s.get('parsed_json_object') else 'N/A'}")
            logger.info(f"  API Usage: {res_s.get('api_usage_metadata')}")
            if res_s.get('json_parsing_error_msg'): logger.warning(f"  Erro Parsing JSON: {res_s.get('json_parsing_error_msg')}")
            if res_s.get('processing_error_msg'): logger.error(f"  Erro Processamento: {res_s.get('processing_error_msg')}")
        else: logger.error("Nenhum dir válido com imgs para teste de processamento Gemini.")
    except Exception as e: logger.error(f"Erro teste processamento único Gemini: {e}")
    logger.info("="*20 + " TESTES DE DIAGNÓSTICO GEMINI CONCLUÍDOS " + "="*20)


# --- Função principal do script ---
if __name__ == "__main__":
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"Iniciando script de GERAÇÃO MULTI-CONFIGURAÇÃO para Modelos GEMINI.")
        logger.info(f"Modelos configurados: {MODEL_NAMES_TO_TEST}")
        logger.info(f"Temperaturas configuradas: {TEMPERATURES_TO_TEST}")
        
        if "--test" in sys.argv or "-t" in sys.argv:
            test_model = MODEL_NAMES_TO_TEST[0].strip() if MODEL_NAMES_TO_TEST else "gemini-1.5-flash-latest"
            test_temp = TEMPERATURES_TO_TEST[0] if TEMPERATURES_TO_TEST else 0.1
            
            logger.info(f"Executando testes de diagnóstico para: Modelo Gemini={test_model}, Temperatura={test_temp}")
            run_diagnostic_tests_simplified(model_to_test=test_model, temp_to_test=test_temp, num_properties_to_test=1)
            
            user_response = input("Testes de diagnóstico concluídos. Executar geração multi-configuração completa com Gemini? (S/n): ").strip().lower()
            if not (user_response == "" or user_response.startswith("s")):
                logger.info("Geração multi-configuração Gemini cancelada pelo usuário.")
                sys.exit(0)
        
        logger.info("Iniciando geração multi-configuração completa com Gemini...")
        run_multiconfig_gemini_generation()
        logger.info("Geração multi-configuração Gemini finalizada com sucesso!")
        
    except SystemExit: pass
    except KeyboardInterrupt: logger.warning("Interrompido (KeyboardInterrupt)."); sys.exit(1)
    except Exception as e: logger.exception(f"Erro fatal na geração multi-configuração Gemini: {e}"); sys.exit(1)