"""Script para geração de respostas JSON de modelos VLM do Ollama.

Este script itera sobre um dataset de imóveis, processa as imagens associadas
a cada imóvel utilizando um modelo VLM (Vision Language Model) configurado
no Ollama, e envia prompts específicos (sistema e humano) para o modelo.
O objetivo é que o modelo retorne uma análise estruturada em formato JSON.

Os resultados, incluindo a string JSON crua retornada pelo modelo, o objeto JSON
parseado (se o parsing for bem-sucedido), e metadados sobre o processamento
(latências, erros, etc.), são salvos em um arquivo JSON consolidado.

A avaliação das métricas de qualidade dessas respostas JSON é realizada por um
script separado (evaluate_results.py), que consome o arquivo JSON gerado por este script.
"""

import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import sys
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm # Para barra de progresso visual
from loguru import logger # Para logging flexível

# Tentativa de importar módulos utilitários e o conector Ollama
try:
    from utils.helpers import (
        get_images_from_directory,
        _encode_image_to_base64,
    )
    from ollama_connector import ollama_generate, fetch_available_models, validate_model

except ImportError as e:
    logger.critical(f"Erro de importação: {e}. Verifique ollama_connector.py e utils/helpers.py.")
    raise SystemExit(1)

load_dotenv()

# --- Configuração ---
MODEL_NAMES_TO_TEST = os.getenv("BENCHMARK_MODELS", "gemma3:4b,llava:7b").split(',') # Modelos VLMs a serem utilizados no Ollama
TEMPERATURES_TO_TEST = [float(t) for t in os.getenv("BENCHMARK_TEMPERATURES", "0.2,0.4").split(',')] # Ex: [0.1, 0.5, 0.8]
DATASET_PATH = Path(os.getenv("BENCHMARK_DATASET_PATH", "./data/benchmark_50_anotacao_ollama.csv")) # Caminho para o arquivo CSV contendo o dataset de imóveis a serem analisados.
IMAGES_BASE_DIR = Path(os.getenv("BENCHMARK_IMAGES_DIR", "./data/images")) # Diretório onde as imagens dos imóveis estão localizadas.
OUTPUT_BASE_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "./resultados_geracao")) # Diretório onde os resultados da geração e consolidação serão salvos.
PROMPTS_DIR = Path(os.getenv("BENCHMARK_PROMPTS_DIR", "./prompts")) # Diretório contendo os arquivos de texto para os prompts do sistema e humano.
TIPO_PROMPT_LABEL = "analise_imovel_v1.2" # Rótulo para o tipo/versão do conjunto de prompts utilizado.



# Nomes dos arquivos de prompt.
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
HUMAN_PROMPT_FILE = PROMPTS_DIR / "human_prompt_gemma.txt"

LOG_FILE_PATH = "logs/generation_ollama.log" # Caminho para o arquivo de log deste script.

# --- Configuração do Logging ---
os.makedirs("logs", exist_ok=True)
logger.remove() # Remove handlers padrão para evitar duplicação em re-execuções
logger.add(sys.stderr, level="INFO") # Envia logs de nível INFO e acima para o console (stderr)
logger.add(LOG_FILE_PATH, rotation="10 MB", retention="7 days", level="DEBUG", encoding="utf-8") # Salva logs em arquivo

# --- Carregar Conteúdos dos Prompts ---
SYSTEM_PROMPT_CONTENT = ""
HUMAN_PROMPT_CONTENT = ""
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f: SYSTEM_PROMPT_CONTENT = f.read()
    with open(HUMAN_PROMPT_FILE, 'r', encoding='utf-8') as f: HUMAN_PROMPT_CONTENT = f.read()
    logger.info(f"System prompt carregado de: {SYSTEM_PROMPT_FILE}")
    logger.info(f"Human prompt carregado de: {HUMAN_PROMPT_FILE}")
    # Levanta um erro se algum dos prompts estiver vazio, pois são essenciais.
    if not SYSTEM_PROMPT_CONTENT or not HUMAN_PROMPT_CONTENT: raise ValueError("Prompts vazios.")
except Exception as e: logger.critical(f"Erro ao ler prompts: {e}"); raise SystemExit(f"Erro crítico ao ler prompts: {e}")

def clean_json_string(json_str: str) -> str:
    
    """
    Remove caracteres de controle problemáticos de uma string que se espera ser JSON.

    Esta função itera sobre os caracteres da string de entrada e mantém apenas:
    - Caracteres ASCII imprimíveis (32-126).
    - Caracteres de escape JSON comuns (tab, newline, carriage return, form feed, backspace).
    - Caracteres Unicode acima de 127 (para suportar acentuação, etc.).
    - Barras invertidas (essenciais para sequências de escape JSON).

    Args:
        json_str: A string contendo caracteres de controle inválidos.

    Returns:
        Uma string limpa, com caracteres de controle inválidos removidos.
    """
    
    cleaned_chars = []
    for char_code in map(ord, json_str):
        # Permitir ASCII imprimíveis, escapes JSON comuns e Unicode > 127
        if (32 <= char_code <= 126) or char_code in [9, 10, 13, 12, 8] or char_code > 127:
            cleaned_chars.append(chr(char_code))
        elif chr(char_code) == '\\': cleaned_chars.append('\\') # Manter a barra invertida
    cleaned_str = "".join(cleaned_chars)
    return cleaned_str

def extract_json_from_response(raw_response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    
    """
    Tenta extrair um objeto JSON da string de resposta bruta de um modelo.

    Procura por dois padrões principais:
    1. Um bloco de código JSON demarcado por ```json ... ```.
    2. Um objeto JSON direto (começando com '{' e terminando com '}').

    A string JSON candidata é então limpa de caracteres de controle inválidos
    antes de tentar o parsing com `json.loads()`.

    Args:
        raw_response_text: A string de resposta completa do modelo VLM.

    Returns:
        Uma tupla contendo:
        - O objeto Python (dicionário) parseado se o JSON for válido, caso contrário None.
        - Uma string de mensagem de erro se o parsing falhar ou nenhum JSON for encontrado, caso contrário None.
    """
    
    if not raw_response_text: return None, "Resposta do modelo vazia."
    json_str_candidate = None
    
    # Tenta encontrar um bloco de código JSON (ex: ```json { ... } ```)
    # O padrão [\s\S]*? permite que o conteúdo JSON seja multilinha e não-guloso.
    match_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', raw_response_text, re.DOTALL)
    if match_block: json_str_candidate = match_block.group(1)
    else:
        # Se não encontrar bloco, procura por JSON direto (primeiro '{' ao último '}')
        first_brace = raw_response_text.find('{')
        last_brace = raw_response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str_candidate = raw_response_text[first_brace : last_brace+1]
        else:
            # Nenhum padrão JSON reconhecível encontrado
            return None, "Nenhum padrão JSON (bloco ou chaves) encontrado."
    if not json_str_candidate:
        return None, "Nenhum candidato a string JSON."
        
    # Limpa a string JSON com caracteres de controle inválidos
    cleaned_json_str = clean_json_string(json_str_candidate)

    # Tenta parsear a string JSON limpa
    try:
        return json.loads(cleaned_json_str), None
    except json.JSONDecodeError as e:
        error_msg = f"Falha parse JSON (limpa): {e}. Str: '{cleaned_json_str[:200]}...'" # Escreve o erro de parsing para depuração
        return None, error_msg # Retorna None e a mensagem de erro

def _prepare_images_for_ollama(image_dir_path: Path) -> Tuple[List[str], List[str], Optional[str]]:

    """
    Prepara as imagens de um diretório para serem enviadas à API Ollama.

    Esta função:
    1. Lista todos os arquivos de imagem no diretório fornecido usando `get_images_from_directory`.
    2. Realiza uma verificação heurística para pular caminhos que parecem ser strings base64 em vez de caminhos de arquivo.
    3. Valida se cada caminho de imagem realmente aponta para um arquivo.
    4. Codifica cada imagem válida para uma string base64 usando `_encode_image_to_base64`.
    5. Lida com erros durante a listagem ou codificação de imagens.

    Args:
        image_dir_path: Um objeto Path para o diretório contendo as imagens da propriedade.

    Returns:
        Uma tupla contendo:
        - List[str]: Uma lista de strings base64, cada uma representando uma imagem codificada.
                     Vazia se nenhuma imagem puder ser preparada.
        - List[str]: Uma lista dos caminhos de arquivo (como strings) das imagens que foram
                     efetivamente encontradas e tentadas para codificação.
        - Optional[str]: Uma mensagem de erro se ocorreu um problema durante o processo,
                         caso contrário None.
    """
    
    try:
        # Obtém a lista de caminhos completos para os arquivos de imagem no diretório
        image_paths_full = get_images_from_directory(str(image_dir_path))
    except ValueError as e: # Se get_images_from_directory levantar erro (ex: dir não existe) 
        return [], [], str(e) # Retorna listas vazias e a mensagem de erro
        
    valid_image_paths_obj = [] # Armazena objetos Path validados
    for img_path_str in image_paths_full:
        p = Path(img_path_str)
        # Heurística para detectar se o "caminho" é na verdade uma string base64 (sinal de problema anterior no pipeline)
        if isinstance(img_path_str, str) and len(img_path_str) > 200 and any(img_path_str.startswith(prefix) for prefix in ('data:image','UklGR','R0lGO','iVBOR','/9j/')):
            logger.error(f"Pulando imagem - Base64 como path: {img_path_str[:70]}...")
            continue
            
        # Verifica se o caminho aponta para um arquivo real
        if not p.is_file(): logger.warning(f"Pulando path inválido: {p}"); continue
        valid_image_paths_obj.append(p)
    
    if not valid_image_paths_obj: return [], [], "Nenhum path de imagem válido."
    
    encoded_images = []
    valid_image_paths_final_str = [] # Caminhos das imagens que foram efetivamente codificadas
    
    for p_obj in valid_image_paths_obj:
        try:
            # Codifica a imagem para base64
            encoded_img_str = _encode_image_to_base64(str(p_obj))
            if not isinstance(encoded_img_str, str):
                
                 logger.error(f"Encode não retornou str para {p_obj}")
                 # Tenta converter se for bytes, como um fallback
                 if isinstance(encoded_img_str, bytes): encoded_img_str = encoded_img_str.decode('utf-8')
                 else: # Se não for string nem bytes, é um erro inesperado
                     return [], [str(p) for p in valid_image_paths_obj], f"Encode tipo inesperado: {type(encoded_img_str)}"
                     
            encoded_images.append(encoded_img_str)
            valid_image_paths_final_str.append(str(p_obj))
            
        except Exception as e: # Outros erros durante a codificação
            logger.error(f"Erro ao codificar {p_obj}: {e}")
            return [], [str(p) for p in valid_image_paths_obj], f"Falha ao codificar {p_obj}: {e}"
            
    if not encoded_images: # Se todas as imagens válidas falhassem na codificação.
        return [], valid_image_paths_final_str, "Nenhuma imagem codificada."
    return encoded_images, valid_image_paths_final_str, None

def process_property_images(
    property_dir_path: Path,
    current_model_name: str,
    current_request_options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Processa imagens de uma propriedade com um modelo e opções específicas.
    (Docstring como antes, mas mencionando que model_name e options são parâmetros)
    """

    """
    Processa todas as imagens de um diretório de propriedade especificado, com modelos e opções específicas.

    Esta função orquestra:
    1. A preparação das imagens (listagem, validação, codificação base64) através de `_prepare_images_for_ollama`.
    2. A chamada à API Ollama (`ollama_generate`) com as imagens codificadas e os prompts globais.
    3. O tratamento da resposta da API, incluindo a extração e parsing do JSON retornado pelo modelo.
    4. Coleta de metadados sobre o processo (tempos, erros, etc.).

    Args:
        property_dir_path: Objeto Path para o diretório da propriedade contendo as imagens.
        current_model_name: String com o nome do modelo atualmente utilizado.
        current_request_options: Dicionário com as opções específicas do modelo (Ex: Temperatura)

    Returns:
        Um dicionário contendo detalhes do processamento, incluindo:
        - `image_paths_processed_full`: Lista de caminhos completos das imagens encontradas.
        - `image_filenames_processed`: Lista dos nomes dos arquivos de imagem processados.
        - `num_images_found`: Número total de imagens encontradas.
        - `num_images_sent_to_api`: Número de imagens efetivamente enviadas à API.
        - `raw_model_output`: A string de resposta crua do modelo VLM.
        - `parsed_json_object`: O dicionário Python resultante do parsing do JSON (None se falhar).
        - `json_parsing_error`: Mensagem de erro se o parsing do JSON falhar.
        - `api_total_duration_s`: Duração total da chamada à API Ollama em segundos (se disponível).
        - `function_total_time_s`: Tempo total de execução desta função.
        - `success_api_call`: Booleano indicando se a chamada à API foi bem-sucedida (HTTP 200).
        - `success_json_parsing`: Booleano indicando se o JSON da resposta foi parseado com sucesso.
        - `error_message_processing`: Mensagem de erro geral do processo, se houver.
    """

    process_start_time = time.time() # Início do processamento para este imóvel

    # Dicionário base para armazenar os resultados deste imóvel
    base_result = {
        "image_paths_processed_full": [], "image_filenames_processed": [],
        "num_images_found": 0, "num_images_sent_to_api": 0,
        "raw_model_output": None, "parsed_json_object": None,
        "json_parsing_error": None, "api_total_duration_s": None,
        "function_total_time_s": 0.0, "success_api_call": False,
        "success_json_parsing": False, "error_message_processing": None
    }
    # Prepara as imagens (lista, valida, codifica para base64)
    encoded_images, valid_image_paths_str, prep_error = _prepare_images_for_ollama(property_dir_path)

    # Atualiza o resultado com informações da preparação das imagens
    base_result["image_paths_processed_full"] = valid_image_paths_str
    base_result["image_filenames_processed"] = [Path(p).name for p in valid_image_paths_str]
    base_result["num_images_found"] = len(valid_image_paths_str)

    if prep_error: # Se houve erro na preparação das imagens
        base_result["error_message_processing"] = f"Erro preparação imagens: {prep_error}"
        base_result["function_total_time_s"] = time.time() - process_start_time
        return base_result
    
    base_result["num_images_sent_to_api"] = len(encoded_images)

    if not encoded_images: # Se nenhuma imagem foi codificada com sucesso
        base_result["error_message_processing"] = "Nenhuma imagem codificada para API."
        base_result["function_total_time_s"] = time.time() - process_start_time
        return base_result

    # Tentativa de chamar a API Ollama
    try:
        logger.debug(f"Enviando {len(encoded_images)} imagens de {property_dir_path} para o modelo {current_model_name} com opções: {current_request_options}")
        api_response_data = ollama_generate( # Função do ollama_connector.py
            prompt=HUMAN_PROMPT_CONTENT,
            system=SYSTEM_PROMPT_CONTENT,
            model=current_model_name, # Usa o modelo atual
            images=encoded_images,
            options=current_request_options, # Usa as opções atuais
            validate_model_name=False # Validação será feita antes do loop de propriedades
        )

        # Extrai a duração da API (convertida para segundos)
        if "total_duration" in api_response_data:
            base_result["api_total_duration_s"] = api_response_data["total_duration"] / 1_000_000_000.0

        # Verifica se a API Ollama retornou um erro em seu payload de resposta
        if "error" in api_response_data:
            base_result["error_message_processing"] = f"API Ollama: {api_response_data['error']}"
            # Tenta obter a resposta textual mesmo em caso de erro da API, se houver
            base_result["raw_model_output"] = api_response_data.get("response", api_response_data.get("response_text"))
        else:# A chamada à API não indicou erro no payload
            base_result["success_api_call"] = True # Assume sucesso se não houver chave "error"
            base_result["raw_model_output"] = api_response_data.get("response")

            if base_result["raw_model_output"]:
                # Tenta extrair e parsear o JSON da resposta crua do modelo
                parsed_json, parsing_error = extract_json_from_response(base_result["raw_model_output"])
                base_result["parsed_json_object"] = parsed_json
                base_result["json_parsing_error"] = parsing_error
                if parsed_json and not parsing_error: 
                    base_result["success_json_parsing"] = True
                else: 
                    logger.warning(f"Falha no parsing do JSON para {property_dir_path} (modelo {current_model_name}, temp {current_request_options.get('temperature')}).")
            else: # Resposta da API foi bem-sucedida, mas o campo "response" estava vazio
                base_result["error_message_processing"] = "API Ollama sucesso, mas 'response' vazio."
                base_result["json_parsing_error"] = base_result["error_message_processing"]

    except Exception as e: # Captura outras exceções durante a chamada à API
        base_result["error_message_processing"] = f"Exceção em process_property_images: {type(e).__name__} - {e}"
        logger.exception(f"Exceção em process_property_images para {property_dir_path} (modelo {current_model_name}, temp {current_request_options.get('temperature')})") # Loga o traceback completo

    base_result["function_total_time_s"] = time.time() - process_start_time # Calcula o tempo total da função
    return base_result


def find_property_image_dir(base_dir: Path, property_id_search_term: str) -> Optional[Path]:

    """
    Encontra o diretório de imagens de uma propriedade específica dentro de um diretório base.

    Tenta dois padrões de busca para o `property_id_search_term`:
    1. O termo exato.
    2. O termo seguido por `_*` (para capturar sufixos como `123_casa_rustica`).

    Args:
        base_dir: O objeto Path do diretório raiz onde as pastas das propriedades estão.
        property_id_search_term: O identificador da propriedade a ser procurado (ex: "1", "123_casa").

    Returns:
        Um objeto Path para o diretório da propriedade se encontrado, caso contrário None.
        Se múltiplos diretórios corresponderem ao padrão com `_*`, o primeiro encontrado é retornado.
    """

    patterns_to_try = [property_id_search_term, f"{property_id_search_term}_*"]
    for pattern_str in patterns_to_try:
        try:
            # Usa glob para encontrar correspondências
            matches = list(base_dir.glob(pattern_str))
            # Filtra para garantir que são diretórios
            dir_matches = [m for m in matches if m.is_dir()]
            
            if dir_matches:
                if len(dir_matches) > 1: logger.warning(f"Múltiplos dirs para '{property_id_search_term}' '{pattern_str}'. Usando {dir_matches[0]}")
                return dir_matches[0] # Retorna o primeiro diretório correspondente
                
        except Exception as e: # Captura erros durante o glob
            logger.error(f"Erro find_property_image_dir: {e}")
            continue # Tenta o próximo padrão
            
    logger.warning(f"Nenhum dir para '{property_id_search_term}' em '{base_dir}' padrões {patterns_to_try}.")
    return None

# --- Função Principal de Geração para Múltiplas Configurações ---
def run_multiconfig_json_generation() -> None:
     
    """
    Orquestra o processo completo de geração de respostas JSON para todas as propriedades no dataset,
    com diferentes modelos e temperaturas.

    Esta função realiza as seguintes etapas:
    1. Validações iniciais (existência do dataset, diretório de imagens, disponibilidade do modelo Ollama).
    2. Cria um diretório de saída único para esta execução, nomeado com o modelo e timestamp.
    3. Carrega o dataset CSV.
    4. Itera sobre cada propriedade (linha) no dataset:
        a. Encontra o diretório de imagens da propriedade.
        b. Chama `process_property_images` para obter a análise do VLM.
        c. Formata o resultado do processamento para o esquema de saída desejado.
    5. Agrega todos os resultados formatados.
    6. Salva os resultados agregados em um único arquivo JSON consolidado.
    """
     
    overall_start_time_iso = datetime.now(timezone.utc) # Timestamp do início da execução

    logger.info(f"Iniciando execução multi-configuração em: {overall_start_time_iso.isoformat()}")
    logger.info(f"Modelos a serem testados: {MODEL_NAMES_TO_TEST}")
    logger.info(f"Temperaturas a serem testadas para cada modelo: {TEMPERATURES_TO_TEST}")

    # Validações iniciais de paths (dataset, imagens)
    if not DATASET_PATH.is_file(): logger.critical(f"Dataset não encontrado: {DATASET_PATH}"); raise SystemExit(1) # Interrompe se o dataset não existir
    if not IMAGES_BASE_DIR.is_dir(): logger.critical(f"Dir imagens não encontrado: {IMAGES_BASE_DIR}"); raise SystemExit(1) # Interrompe se o diretório de imagens não existir

    # Carregar o dataset uma vez
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset carregado de '{DATASET_PATH}' com {len(df_dataset)} propriedades.")
    except Exception as e: logger.critical(f"Erro ao carregar dataset: {e}"); raise SystemExit(1)

    # Loop sobre cada modelo a ser testado
    for model_name_current in MODEL_NAMES_TO_TEST:
        model_name_current = model_name_current.strip() # Limpar espaços em branco
        if not model_name_current: continue # Pular nomes de modelo vazios

        logger.info(f"--- Iniciando testes para o MODELO: {model_name_current} ---")
        # Validar o modelo atual uma vez antes de iterar sobre as temperaturas e o dataset
        try:
            if not validate_model(model_name_current): # Valida se o modelo configurado está disponível no Ollama
                logger.error(f"Modelo VLM '{model_name_current}' não está disponível ou não pôde ser validado. Pulando este modelo.")
                continue # Pula para o próximo modelo
            logger.info(f"Modelo VLM '{model_name_current}' validado e pronto para uso.")
        except Exception as e_val:
            logger.error(f"Falha crítica ao tentar validar o modelo VLM '{model_name_current}': {e_val}. Pulando este modelo.")
            continue

        # Loop sobre cada temperatura a ser testada para o modelo atual
        for temp_current in TEMPERATURES_TO_TEST:
            logger.info(f"--- Iniciando geração para MODELO: {model_name_current}, TEMPERATURA: {temp_current} ---")

            generation_start_time_iso = datetime.now(timezone.utc)
            
            # Criação do diretório de saída para esta execução (com data e hora da execução)
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = model_name_current.replace(':', '_').replace('/', '_')
            temp_safe = str(temp_current).replace('.', 'p') # Ex: 0.1 -> 0p1

            # Diretório de saída específico para esta combinação de modelo e temperatura
            current_run_output_dir = OUTPUT_BASE_DIR / f"output_{model_name_safe}_temp_{temp_safe}_{run_timestamp}"
            current_run_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Resultados para {model_name_current} @ temp {temp_current} serão salvos em: {current_run_output_dir}")

            generation_results_path = current_run_output_dir / "generation_results.json" # Caminho para o arquivo JSON de saída consolidado
            lista_de_resultados_formatados_run = [] # Resultados para esta execução específica

            # Adicionar outros parâmetros fixos aqui se desejar, ex: "num_predict": 4096
            request_options_current = {"temperature": temp_current}

            # Itera sobre cada linha (imóvel) do dataset
            for idx, row_dataset in tqdm(df_dataset.iterrows(), total=len(df_dataset), desc=f"Modelo {model_name_current}, Temp {temp_current}"):
                
                current_property_id_output = idx + 1 # Cria um ID sequencial 1-based para a propriedade no output

                # A coluna no CSV que identifica a pasta de imagens
                # Se não existir, usa o índice + 1 como fallback
                property_folder_search_term = str(row_dataset.get('property_folder_id', idx + 1))
                property_dir_path = find_property_image_dir(IMAGES_BASE_DIR, property_folder_search_term)
                

                resultado_item: Dict[str, Any] = { # Estrutura do item como definida antes
                    "property_id": current_property_id_output, # ID numérico sequencial
                    "dataset_idx": idx, # Índice original do CSV (0-based)
                    "modelo": model_name_current, # Nome do Modelo atual
                    "temperatura_usada": temp_current, # Adicionar temperatura
                    "search_term_folder": property_folder_search_term, # Como a pasta foi procurada
                    "property_directory_processed": str(property_dir_path), # Caminho do diretório efetivamente processado
                }

                if not property_dir_path:
                    logger.warning(f"Idx {idx} (Modelo {model_name_current}, Temp {temp_current}): dir não encontrado para '{property_folder_search_term}'.")
                    resultado_item["status_geracao"] = "erro_diretorio_nao_encontrado"
                    resultado_item["error_message"] = f"Diretório para '{property_folder_search_term}' não encontrado."
                else:
                    resultado_item["property_directory_processed"] = str(property_dir_path)
                    processing_details = process_property_images(
                        property_dir_path,
                        current_model_name=model_name_current,
                        current_request_options=request_options_current
                    )
                    # Atualizar resultado_item com os detalhes do processamento
                    resultado_item.update({
                        "num_imagens_encontradas": processing_details.get("num_images_found", 0),
                        "num_imagens_enviadas_api": processing_details.get("num_images_sent_to_api", 0),
                        "tipo_prompt": TIPO_PROMPT_LABEL,
                        "latencia_total_api_s": processing_details.get("api_total_duration_s"),
                        "latencia_funcao_processamento_s": processing_details.get("function_total_time_s"),
                        "resposta_modelo_raw_str": processing_details.get("raw_model_output"), # A string JSON crua
                        "resposta_modelo_parsed_dict": processing_details.get("parsed_json_object"), # A resposta parseada e estruturada
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
                
                # Insere a lista com os resultados finais formatados
                lista_de_resultados_formatados_run.append(resultado_item)

            # Salvar resultados para esta combinação de modelo e temperatura
            output_final_geracao_run = {
                "metadata_geracao_run": {
                    "timestamp_geracao_inicial_run": generation_start_time_iso.isoformat(), # Pode ser um timestamp específico do início deste run
                    "timestamp_geracao_final_run": datetime.now(timezone.utc).isoformat(),
                    "timestamp_final_run": (datetime.now(timezone.utc) - generation_start_time_iso).total_seconds(),
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

            try:# Salva o JSON consolidado
                with open(generation_results_path, 'w', encoding='utf-8') as f_out:
                    # Usar a função default para lidar com tipos NumPy se algum escapar
                    json.dump(output_final_geracao_run, f_out, indent=2, ensure_ascii=False,
                              default=lambda o: int(o) if isinstance(o, np.integer) else
                                                float(o) if isinstance(o, np.floating) else
                                                o.tolist() if isinstance(o, np.ndarray) else str(o))
                logger.info(f"Resultados para {model_name_current} @ temp {temp_current} salvos em: {generation_results_path}")
            except Exception as e_json_save:
                logger.error(f"Erro ao salvar JSON para {model_name_current} @ temp {temp_current}: {e_json_save}")
            
            logger.info(f"--- Finalizada geração para MODELO: {model_name_current}, TEMPERATURA: {temp_current} ---")
        logger.info(f"--- Finalizados todos os testes de temperatura para o MODELO: {model_name_current} ---")
    logger.info(f"Execução multi-configuração finalizada em: {datetime.now(timezone.utc).isoformat()}")


# --- Bloco de Testes de Diagnóstico ---
def run_diagnostic_tests_simplified(model_to_test: str, temp_to_test: float, num_properties_to_test: int = 1):

    """
    Executa uma série de testes de diagnóstico rápidos para verificar a configuração
    básica e a capacidade de processar uma pequena amostra de dados com uma configuração
    específica de modelo e temperatura..

    Foca em:
    - Conexão com Ollama e disponibilidade do modelo.
    - Carregamento do dataset e localização de pastas de imagens.
    - Processamento completo (incluindo chamada à API e parsing JSON) de uma propriedade.

    Args:
        model_to_test: O nome do modelo a ser testado
        temp_to_test: A configuração de temperatura do modelo a ser testada
        num_properties_to_test: O número de propriedades do início do dataset a serem usadas nos testes de localização de pastas e imagens.
    """

    logger.info("="*20 + f" INICIANDO TESTES DE DIAGNÓSTICO (MODELO: {model_to_test}, TEMP: {temp_to_test}) " + "="*20)
    
    # Teste 1: Validação do modelo específico
    logger.info(f"--- Teste 1: Validação do Modelo {model_to_test} ---")
    try:
        if not validate_model(model_to_test):
            logger.error(f"Modelo de teste '{model_to_test}' não disponível ou inválido. Testes de diagnóstico podem falhar.")
            # Não retorna aqui, para permitir que os outros testes tentem rodar se o usuário quiser
        else:
            logger.info(f"Modelo de teste '{model_to_test}' validado.")
    except Exception as e_val_diag:
        logger.error(f"Erro ao validar modelo de teste '{model_to_test}': {e_val_diag}")

    # Teste 2: Carregamento do Dataset e Localização de Pastas de Imagens
    logger.info(f"--- Teste 2: Dataset e Pastas (Primeiras {num_properties_to_test} props) ---")
    if not DATASET_PATH.is_file(): logger.error(f"Dataset não encontrado {DATASET_PATH}"); return
    try:
        df_test = pd.read_csv(DATASET_PATH, nrows=num_properties_to_test)
        for idx, row_test in df_test.iterrows():
            prop_id_search = row_test.get('property_folder_id', str(idx + 1))
            
            test_dir = find_property_image_dir(IMAGES_BASE_DIR, prop_id_search)
            if test_dir:

                logger.info(f"Prop (busca {prop_id_search}, idx {idx}): Dir {test_dir}")

                # Testar preparação de imagens para este diretório
                encoded_imgs, valid_paths, prep_err = _prepare_images_for_ollama(test_dir)
                if prep_err:
                    logger.error(f"  Erro preparar imgs {test_dir}: {prep_err}")
                elif encoded_imgs:
                    logger.info(f"  {len(encoded_imgs)} imgs codificadas. Primeira: {valid_paths[0]}, Base64: {encoded_imgs[0][:30]}...")
                else:
                    logger.warning(f"  Nenhuma img encontrada/codificada para {test_dir}.")
            else:
                logger.warning(f"Prop (busca {prop_id_search}, idx {idx}): Dir NÃO encontrado.")
    
    except Exception as e: logger.error(f"Erro teste dataset/pastas: {e}")

    # Teste 3: Processamento Completo de Uma Única Propriedade
    logger.info(f"--- Teste 3: Processamento de Uma Propriedade com {model_to_test} @ Temp {temp_to_test} ---")
    first_valid_dir = None
    try:
        # Tenta encontrar um diretório válido com imagens nas primeiras N propriedades
        df_s_test = pd.read_csv(DATASET_PATH, nrows=min(5, len(pd.read_csv(DATASET_PATH)) if DATASET_PATH.is_file() else 5) )
        for idx_s, row_s in df_s_test.iterrows():
            prop_id_s = row_s.get('property_folder_id', str(idx_s + 1))
            dir_p = find_property_image_dir(IMAGES_BASE_DIR, prop_id_s)
            
            if dir_p:
                # Verifica se há imagens e se a preparação não dá erro
                _, _, prep_err_check = _prepare_images_for_ollama(dir_p)
                if not prep_err_check: # Se não houve erro na preparação, o diretório é bom para teste
                    first_valid_dir = dir_p
                    break # Encontrou um, pode parar de procurar
                else: logger.warning(f"Dir {dir_p} encontrado, mas com erro na preparação de imagens para teste: {prep_err_check}")
        
        if first_valid_dir:
            logger.info(f"Testando process_property_images com: {first_valid_dir}, Modelo: {model_to_test}, Temp: {temp_to_test}")
            test_options = {"temperature": temp_to_test}
            res_s = process_property_images(first_valid_dir, model_to_test, test_options) # Passa o modelo e opções
            logger.info(f"Resultado (API: {res_s.get('success_api_call')}, JSON Parse: {res_s.get('success_json_parsing')}):")
            logger.info(f"  JSON Obj (preview): {str(res_s.get('parsed_json_object'))[:200] if res_s.get('parsed_json_object') else 'N/A'}")
            if res_s.get('json_parsing_error'): logger.warning(f"  Erro Parsing JSON: {res_s.get('json_parsing_error')}")
            if res_s.get('error_message_processing'): logger.error(f"  Erro Processamento: {res_s.get('error_message_processing')}")
        else: logger.error("Nenhum dir válido com imgs para teste de processamento.")
        
    except Exception as e: logger.error(f"Erro teste processamento único: {e}")

    logger.info("="*20 + " TESTES DE DIAGNÓSTICO CONCLUÍDOS " + "="*20)

# --- Função principal do script ---
if __name__ == "__main__":
    
    try:
        # Cria diretórios de saída e logs se não existirem
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"Iniciando script de GERAÇÃO MULTI-CONFIGURAÇÃO para VLM.")
        logger.info(f"Modelos configurados: {MODEL_NAMES_TO_TEST}")
        logger.info(f"Temperaturas configuradas: {TEMPERATURES_TO_TEST}")
        
        # Permite rodar testes de diagnóstico com o argumento --test ou -t
        if "--test" in sys.argv or "-t" in sys.argv:
            # Para teste, usar o primeiro modelo e a primeira temperatura da lista como exemplo
            test_model = MODEL_NAMES_TO_TEST[0].strip() if MODEL_NAMES_TO_TEST else "llava:7b" # Fallback
            test_temp = TEMPERATURES_TO_TEST[0] if TEMPERATURES_TO_TEST else 0.1 # Fallback
            
            logger.info(f"Executando testes de diagnóstico para: Modelo={test_model}, Temperatura={test_temp}")
            run_diagnostic_tests_simplified(model_to_test=test_model, temp_to_test=test_temp, num_properties_to_test=1)
            
            # Pergunta ao usuário se deseja continuar para a execução completa
            user_response = input("Testes de diagnóstico concluídos. Executar geração multi-configuração completa? (S/n): ").strip().lower()
            if not (user_response == "" or user_response.startswith("s")):
                logger.info("Geração multi-configuração cancelada pelo usuário.")
                sys.exit(0) # Sai do script se o usuário não quiser continuar
        
        logger.info("Iniciando geração multi-configuração completa...")
        run_multiconfig_json_generation() # Chamando a função de geração
        logger.info("Geração multi-configuração finalizada com sucesso!")
        
    except SystemExit: pass # Captura SystemExit para saídas controladas
    except KeyboardInterrupt: logger.warning("Interrompido (KeyboardInterrupt)."); sys.exit(1) # Se o usuário interromper com Ctrl+C
    except Exception as e: logger.exception(f"Erro fatal na geração multi-configuração: {e}"); sys.exit(1) # Captura quaisquer outras exceções não tratadas