'''
Script para a construção de arquivos .csv com Métricas Médias Gerais por Modelo
e Métricas Médias 
'''

import json
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np
import sys
import os
import argparse
from datetime import datetime
import re # Importar re para extract_model_and_temp_from_filename
from typing import Optional, Tuple, List, Dict, Any # Adicionar tipos que faltavam
from tqdm import tqdm

try:
    from .metrics_eval import (
        plot_classification_metrics_bars,
        CAMPOS_CATEGORICOS_MAPEAMENTO, # Importar o mapeamento das categorias
    )
except ImportError:
    try:
        current_script_path = Path(__file__).resolve()
        project_root_for_import = current_script_path.parent.parent
        if str(project_root_for_import) not in sys.path:
            sys.path.insert(0, str(project_root_for_import))
        from utils.metrics_eval import (
            plot_classification_metrics_bars,
            CAMPOS_CATEGORICOS_MAPEAMENTO # Tentar importar novamente
        )
    except ImportError as e_abs:
        logger.critical(f"Erro ao importar de metrics_eval: {e_abs}. Certifique-se de que o arquivo e as constantes existem.")
        CAMPOS_CATEGORICOS_MAPEAMENTO = {} # Fallback
        plot_classification_metrics_bars = None
        logger.warning("Dependências de metrics_eval não puderam ser importadas. Funcionalidade limitada.")


# --- Configuração ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "resultados_avaliacao"
DEFAULT_SUMMARY_OUTPUT_DIR = PROJECT_ROOT / "resultados_sumarizados"
LOG_FILE_PATH_SUMMARY = PROJECT_ROOT / "logs" / "summary_report_creation.log"

# --- Configurar Logging ---
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH_SUMMARY, rotation="2 MB", retention="3 days", level="DEBUG", encoding="utf-8")


def extract_model_and_temp_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]: # Temp como string
    """
    Tenta extrair o nome do modelo e a string da temperatura do nome do arquivo.
    """
    # Padrão mais robusto que tenta capturar o nome do modelo e a temperatura opcional.
    # Aceita nomes de arquivo como:
    # evaluation_metrics_MODELO_temp_TEMPERATURA.json
    match = re.search(r"evaluation_metrics_(.+?)_temp_([\d_p]+)\.json", filename)
    if match:
        model_part = match.group(1)
        temp_part = match.group(2) # Pode ser None

        # Reconstruir nome do modelo (reverter _safe_ para :)
        model_name = model_part.replace('_', ':').replace('p', '.') # gemini-2.5-flash
        
        # Tentar remover sufixo de timestamp do nome do modelo se ele foi concatenado
        model_name_no_ts_match = re.match(r"(.+?)_\d{8}_\d{6}", model_name)
        if model_name_no_ts_match:
            model_name = model_name_no_ts_match.group(1)
        
        # A temperatura é mantida como string (ex: "0p1", "1p0") ou None
        temperature_str = temp_part if temp_part else None
        return model_name, temperature_str
    
    logger.warning(f"Não foi possível extrair modelo/temperatura do nome do arquivo: {filename}")
    return None, None

def create_summary_report(input_dir: Path, output_dir: Path):
    logger.info(f"Lendo arquivos de avaliação de: {input_dir}")
    logger.info(f"Salvando relatório sumarizado em: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_categories_summary_data = [] # Para o CSV de métricas por categoria
    all_overall_averages_data = []   # Para o CSV de médias gerais do modelo

    evaluation_files = list(input_dir.glob("**/evaluation_metrics_*.json"))
    if not evaluation_files:
        logger.error(f"Nenhum arquivo 'evaluation_metrics_*.json' encontrado em {input_dir} ou subdiretórios.")
        return
    logger.info(f"Encontrados {len(evaluation_files)} arquivos de avaliação para processar.")

    for eval_file_path in tqdm(evaluation_files, desc="Processando arquivos de avaliação"):
        logger.debug(f"Processando arquivo: {eval_file_path.name}")
        try:
            with open(eval_file_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar ou parsear JSON de {eval_file_path}: {e}")
            continue

        metadata = eval_data.get("metadata_avaliacao", {})
        model_name_from_meta = metadata.get("modelo_avaliado")
        # Tenta obter a temperatura dos metadados do arquivo de avaliação
        temp_from_meta = metadata.get("temperatura") 

        # Fallback para extrair do nome do arquivo
        parsed_model_fn, parsed_temp_str_fn = extract_model_and_temp_from_filename(eval_file_path.name)

        model_name = model_name_from_meta or parsed_model_fn or "unknown_model"
        model_name = model_name.replace(':', '_') 
        
        # Determinar a string de temperatura para usar como identificador
        temperature_label = "N/A"
        if temp_from_meta is not None:
            temperature_label = str(temp_from_meta)
        elif parsed_temp_str_fn is not None:
            temperature_label = parsed_temp_str_fn.replace('p', '.') # Converter "0p1" para "0.1" para o rótulo

        metrics_por_categoria = eval_data.get("Metricas_Agregadas_Por_Categoria", {}) # Nome da chave no JSON de avaliação
        
        # Coletar métricas por categoria individual
        for category_key_safe, cat_metrics in metrics_por_categoria.items():
            # Pular chaves que são agregações gerais ou de tipos diferentes
            if category_key_safe == f"Metricas_Medias_Gerais_Do_Modelo_{model_name}":
                continue # Pular métricas agregadas e de texto por enquanto para esta tabela

            if isinstance(cat_metrics, dict) and "error" not in cat_metrics:
                original_category_name = category_key_safe # Fallback
                if CAMPOS_CATEGORICOS_MAPEAMENTO: # Apenas se o mapeamento foi carregado
                    for cat_orig, config in CAMPOS_CATEGORICOS_MAPEAMENTO.items():
                        # Gerar a safe_key da mesma forma que em evaluate_results.py para encontrar a correspondência
                        expected_safe_key = cat_orig.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e')
                        if expected_safe_key == category_key_safe:
                            original_category_name = cat_orig
                            break
                
                all_categories_summary_data.append({
                    "Modelo": model_name,
                    "Temperatura": temperature_label,
                    "Categoria": original_category_name,
                    "Accuracy": cat_metrics.get("accuracy"),
                    "F1-Score": cat_metrics.get("f1_score_weighted"),
                    "Precision": cat_metrics.get("precision_weighted"),
                    "Recall": cat_metrics.get("recall_weighted")
                })
        
        # Coletar a média geral das categorias, se presente
        overall_avg = metrics_por_categoria.get(f"Metricas_Medias_Gerais_Do_Modelo_{model_name}", {})
        if "error" not in overall_avg and overall_avg:
            all_overall_averages_data.append({
                "Modelo": model_name,
                "Temperatura": temperature_label,
                "Accuracy": overall_avg.get(f"accuracy_{model_name}"),
                "F1-Score": overall_avg.get(f"f1_score_weighted_{model_name}"),
                "Precision": overall_avg.get(f"precision_weighted_{model_name}"),
                "Recall": overall_avg.get(f"recall_weighted_{model_name}"),
                "Tempo de Execucao Total (s)": metadata.get("latencia_total_resposta_modelo_s")
            })

    # --- Salvar CSV de Métricas por Categoria ---
    if all_categories_summary_data:
        df_categories_report = pd.DataFrame(all_categories_summary_data)
        column_order_cat = ["Modelo", "Temperatura", "Categoria", "Accuracy", "F1-Score", "Precision", "Recall"]
        existing_columns_cat = [col for col in column_order_cat if col in df_categories_report.columns]
        df_categories_report = df_categories_report[existing_columns_cat]

        summary_csv_cat_filename = f"report_metrics_by_category_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_csv_cat_path = output_dir / summary_csv_cat_filename
        try:
            df_categories_report.to_csv(summary_csv_cat_path, index=False, float_format='%.4f')
            logger.info(f"Relatório de métricas POR CATEGORIA salvo em CSV: {summary_csv_cat_path}")
        except Exception as e_csv_cat:
            logger.error(f"Erro ao salvar relatório por categoria em CSV: {e_csv_cat}")
    else:
        logger.warning("Nenhum dado de métrica por categoria foi extraído para o CSV.")

    # --- Salvar CSV de Médias Gerais do Modelo ---
    if all_overall_averages_data:
        df_overall_report = pd.DataFrame(all_overall_averages_data)
        column_order_overall = ["Modelo", "Temperatura", "Accuracy", "F1-Score", 
                                "Precision", "Recall", "Tempo de Execucao Total (s)"]
        existing_columns_overall = [col for col in column_order_overall if col in df_overall_report.columns]
        df_overall_report = df_overall_report[existing_columns_overall]

        summary_csv_overall_filename = f"report_overall_model_averages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_csv_overall_path = output_dir / summary_csv_overall_filename
        try:
            df_overall_report.to_csv(summary_csv_overall_path, index=False, float_format='%.4f')
            logger.info(f"Relatório de MÉDIAS GERAIS DO MODELO salvo em CSV: {summary_csv_overall_path}")
        except Exception as e_csv_overall:
            logger.error(f"Erro ao salvar relatório de médias gerais em CSV: {e_csv_overall}")
    else:
        logger.warning("Nenhum dado de média geral do modelo foi extraído para o CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria um relatório sumarizado (CSV e gráficos) a partir de múltiplos arquivos de avaliação JSON.")
    parser.add_argument(
        "input_evaluation_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        nargs='?',
        help=f"Diretório contendo os arquivos JSON de avaliação. Padrão: {DEFAULT_INPUT_DIR}"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_DIR,
        help=f"Diretório para salvar o relatório sumarizado. Padrão: {DEFAULT_SUMMARY_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    try:
        create_summary_report(args.input_evaluation_dir, args.output_dir)
        logger.info("Criação do relatório sumarizado concluída com sucesso!")
    except SystemExit:
        pass
    except Exception as e:
        logger.exception(f"Erro fatal durante a criação do relatório sumarizado: {e}")
        sys.exit(1)