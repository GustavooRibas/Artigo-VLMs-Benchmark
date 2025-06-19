'''
Script para a geração de métricas e gráficos
'''

import json
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np
import sys
import os
from tqdm import tqdm
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv


try:
    # Se metrics_eval.py está no mesmo diretório (utils)
    from .metrics_eval import (
        evaluate_categorical_field,
        # calculate_semantic_similarity,
        plot_confusion_matrix,
        # evaluate_benfeitorias,
        plot_classification_metrics_bars,
        plot_overall_average_metrics_bar,
        OPCOES_ESTRUTURA, OPCOES_ESQUADRIAS, OPCOES_PISO, OPCOES_FORRO,
        OPCOES_INSTALACAO_ELETRICA, OPCOES_INSTALACAO_SANITARIA,
        OPCOES_REVESTIMENTO_INTERNO, OPCOES_ACABAMENTO_INTERNO,
        OPCOES_REVESTIMENTO_EXTERNO, OPCOES_ACABAMENTO_EXTERNO,
        OPCOES_COBERTURA, OPCOES_TIPO_IMOVEL,
        LISTA_TODAS_BENFEITORIAS_POSSIVEIS
    )
except ImportError as e:
    try:
        from metrics_eval import ( # Tenta importar como se estivesse no mesmo nível (se utils está no PYTHONPATH)
            evaluate_categorical_field,
            # calculate_semantic_similarity,
            plot_confusion_matrix,
            # evaluate_benfeitorias,
            plot_classification_metrics_bars,
            plot_overall_average_metrics_bar,
            OPCOES_ESTRUTURA, OPCOES_ESQUADRIAS, OPCOES_PISO, OPCOES_FORRO,
            OPCOES_INSTALACAO_ELETRICA, OPCOES_INSTALACAO_SANITARIA,
            OPCOES_REVESTIMENTO_INTERNO, OPCOES_ACABAMENTO_INTERNO,
            OPCOES_REVESTIMENTO_EXTERNO, OPCOES_ACABAMENTO_EXTERNO,
            OPCOES_COBERTURA, OPCOES_TIPO_IMOVEL,
            LISTA_TODAS_BENFEITORIAS_POSSIVEIS
        )
    except ImportError:
        logger.critical(f"Erro ao importar de metrics_eval: {e}. Verifique o sys.path e a estrutura do projeto.")
        logger.critical("Certifique-se de que metrics_eval.py está no diretório 'utils' ou acessível.")
        sys.exit(1)


# --- Configuração ---
# Caminho para o arquivo JSON de resultados gerado pelo benchmark_ollama_multimodels.py ou benchmark_gemini_multimodels.py
DEFAULT_GENERATED_RESULTS_FILE = None # Será definido por argumento

# Caminho para o diretório raiz do projeto (assumindo que utils/ é um subdiretório direto)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_DATASET_CSV_PATH = PROJECT_ROOT / "data" / "benchmark_50_anotacao_gemini.csv" # Seu CSV de ground truth
DEFAULT_EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "resultados_avaliacao" # Onde salvar as métricas e plots

LOG_FILE_PATH_EVAL = "logs/evaluation_results.log"

# --- Configurar Logging ---
os.makedirs("logs", exist_ok=True) # Garante que a pasta de logs exista
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH_EVAL, rotation="5 MB", retention="3 days", level="DEBUG", encoding="utf-8")


def convert_numpy_types_for_json(obj):

    """Converte tipos NumPy para tipos Python nativos para serialização JSON."""

    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.bool_): return bool(obj)

    # Se você encontrar outros tipos, adicione aqui
    logger.warning(f"Objeto de tipo {type(obj)} encontrado durante a serialização JSON, tentando converter para string.")
    return str(obj) # Fallback para converter para string, mas o ideal é tratar explicitamente

def run_evaluation(generated_results_file: Path, dataset_csv_path: Path, eval_output_dir: Path):

    logger.info(f"Iniciando avaliação do arquivo de resultados: {generated_results_file}")
    logger.info(f"Usando dataset de ground truth: {dataset_csv_path}")
    logger.info(f"Diretório de saída da avaliação: {eval_output_dir}")

    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar resultados gerados
    try:
        with open(generated_results_file, 'r', encoding='utf-8') as f:
            generation_data = json.load(f)
        logger.info(f"Resultados da geração carregados de: {generated_results_file}")
    except Exception as e:
        logger.critical(f"Erro ao carregar arquivo de resultados gerados '{generated_results_file}': {e}")
        return

    # Carregar dataset de ground truth
    try:
        df_ground_truth = pd.read_csv(dataset_csv_path)
        logger.info(f"Dataset de ground truth carregado com {len(df_ground_truth)} registros.")
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset de ground truth '{dataset_csv_path}': {e}")
        return

    metadata_geracao = generation_data.get("metadata_geracao_run", {})
    resultados_gerados = generation_data.get("resultados_gerados_run", [])

    if not resultados_gerados:
        logger.warning("Nenhum resultado gerado encontrado no arquivo para avaliar.")
        return

    model_name_from_results = metadata_geracao.get("modelo_usado", "unknown_model")
    
    # --- Preparar listas para agregação de métricas ---
    CAMPOS_CATEGORICOS_MAPEAMENTO = {
        "Estrutura": {"gt_col": "estrutura", "opcoes": OPCOES_ESTRUTURA}, # Nome da coluna no CSV GT
        "Esquadrias": {"gt_col": "esquadrias", "opcoes": OPCOES_ESQUADRIAS},
        "Piso": {"gt_col": "piso", "opcoes": OPCOES_PISO},
        "Forro": {"gt_col": "forro", "opcoes": OPCOES_FORRO},
        "Instalação Elétrica": {"gt_col": "instalacao_eletrica", "opcoes": OPCOES_INSTALACAO_ELETRICA},
        "Instalação Sanitária": {"gt_col": "instalacao_sanitaria", "opcoes": OPCOES_INSTALACAO_SANITARIA},
        "Revestimento Interno": {"gt_col": "revestimento_interno", "opcoes": OPCOES_REVESTIMENTO_INTERNO},
        "Acabamento Interno": {"gt_col": "acabamento_interno", "opcoes": OPCOES_ACABAMENTO_INTERNO},
        "Revestimento Externo": {"gt_col": "revestimento_externo", "opcoes": OPCOES_REVESTIMENTO_EXTERNO},
        "Acabamento Externo": {"gt_col": "acabamento_externo", "opcoes": OPCOES_ACABAMENTO_EXTERNO},
        "Cobertura": {"gt_col": "cobertura", "opcoes": OPCOES_COBERTURA},
        # "Tipo de Imóvel": {"gt_col": "tipo_de_imovel", "opcoes": OPCOES_TIPO_IMOVEL}
    }
    all_gt_fields = { key.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e'): [] for key in CAMPOS_CATEGORICOS_MAPEAMENTO.keys() }
    all_pred_fields = { key.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e'): [] for key in CAMPOS_CATEGORICOS_MAPEAMENTO.keys() }

    for item_gerado in tqdm(resultados_gerados, desc="Avaliando resultados"):
        dataset_idx = item_gerado.get("dataset_idx")
        if dataset_idx is None or dataset_idx < 0 or dataset_idx >= len(df_ground_truth):
            logger.warning(f"Índice de dataset inválido ou ausente para o item: {item_gerado.get('property_id')}. Pulando.")
            # Preencher com placeholders se for pular, para manter o alinhamento das listas
            for campo_json in CAMPOS_CATEGORICOS_MAPEAMENTO.keys():
                safe_key = campo_json.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e')
                all_gt_fields[safe_key].append("NA_ITEM_PULADO")
                all_pred_fields[safe_key].append("NA_ITEM_PULADO")
            continue

        row_gt = df_ground_truth.iloc[dataset_idx]
        parsed_model_json = item_gerado.get("resposta_modelo_parsed_dict") # Usar o dict parseado

        if item_gerado.get("status_geracao") == "erro_diretorio_nao_encontrado" or \
           item_gerado.get("status_api_call") == "falha" or \
           not parsed_model_json:
            # Se houve erro na geração ou parsing, preencher predições com placeholder de falha
            for campo_json_original in CAMPOS_CATEGORICOS_MAPEAMENTO.keys():
                safe_key = campo_json_original.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e')
                gt_valor_raw = row_gt.get(CAMPOS_CATEGORICOS_MAPEAMENTO[campo_json_original]["gt_col"])
                gt_valor = "na" if pd.isna(gt_valor_raw) or str(gt_valor_raw).strip() == "" else str(gt_valor_raw).lower().strip()
                all_gt_fields[safe_key].append(gt_valor)
                all_pred_fields[safe_key].append("NA_FALHA_GERACAO_OU_PARSE")
            
            continue # Pular para o próximo item gerado

        # Se chegou aqui, temos um JSON parseado do modelo
        for campo_json_original, config_campo in CAMPOS_CATEGORICOS_MAPEAMENTO.items():
            gt_valor_raw = row_gt.get(config_campo["gt_col"])
            pred_valor_raw = parsed_model_json.get(campo_json_original)
            
            gt_valor = "na" if pd.isna(gt_valor_raw) or str(gt_valor_raw).strip() == "" else str(gt_valor_raw).lower().strip()
            pred_valor = "na" if pred_valor_raw is None or str(pred_valor_raw).strip() == "" else str(pred_valor_raw).lower().strip()
            
            safe_key = campo_json_original.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e')
            all_gt_fields[safe_key].append(gt_valor)
            all_pred_fields[safe_key].append(pred_valor)

    # --- Calcular Métricas Agregadas ---

    # Criação do diretório de saída para esta execução (com data e hora da execução)
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_metrics = model_name_from_results.replace(':', '_').replace('/', '_')
    temp_safe = str(metadata_geracao.get("temperatura_usada")).replace('.', 'p')

    aggregated_metrics_results = {} # Dicionário para guardar todas as métricas
    # Dicionário específico para as métricas que irão para o CSV e gráfico de barras
    metrics_for_csv_plot = {}

    metrics_plots_dir = eval_output_dir / f"evaluation_metrics_{model_name_metrics}_temp_{temp_safe}_{timestamp_run}" / f"metric_plots_{model_name_metrics}_temp_{temp_safe}"
    metrics_plots_dir.mkdir(parents=True, exist_ok=True)

    for campo_json_original, config_campo in CAMPOS_CATEGORICOS_MAPEAMENTO.items():
        safe_key = campo_json_original.replace(' ', '_').replace('ç', 'c').replace('ã', 'a').replace('á', 'a').replace('é', 'e')
        logger.info(f"Calculando métricas para: {campo_json_original} (key: {safe_key})")
        
        y_true_campo = all_gt_fields[safe_key]
        y_pred_campo = all_pred_fields[safe_key]

        if len(y_true_campo) != len(y_pred_campo): # Esta checagem é crucial
            logger.error(f"Desalinhamento GT/Pred para {safe_key}: GT {len(y_true_campo)}, Pred {len(y_pred_campo)}")
            aggregated_metrics_results[safe_key] = {"error": "Desalinhamento GT/Pred"}
            metrics_for_csv_plot[campo_json_original] = {"error": "Desalinhamento GT/Pred"} # Usar nome original da categoria
            continue
        if not y_true_campo: # Se a lista estiver vazia após todas as iterações
            logger.warning(f"Sem dados para avaliar {safe_key}.")
            aggregated_metrics_results[safe_key] = {"error": "Sem dados para avaliação."}
            metrics_for_csv_plot[campo_json_original] = {"error": "Sem dados para avaliação."}
            continue
            
        opcoes_normalizadas_campo = [opt.lower().strip() for opt in config_campo["opcoes"]]
        # Adicionar placeholders à lista de labels se eles foram usados nos dados
        labels_para_avaliacao_campo = sorted(list(set(opcoes_normalizadas_campo + ["na_item_invalido_ou_pulado", "na_falha_geracao_ou_parse"])))

        metrics_campo = evaluate_categorical_field(y_true_campo, y_pred_campo, labels=labels_para_avaliacao_campo)
        aggregated_metrics_results[safe_key] = metrics_campo # Contém todas as métricas, incluindo por classe

        # Coletar métricas específicas para o CSV e gráfico de barras
        if "error" not in metrics_campo:
            metrics_for_csv_plot[campo_json_original] = { # Usar nome original da categoria como chave
                "accuracy": metrics_campo.get("accuracy"),
                "f1_score_weighted": metrics_campo.get("f1_score_weighted"),
                "precision_weighted": metrics_campo.get("precision_weighted"),
                "recall_weighted": metrics_campo.get("recall_weighted")
            }
        else:
            metrics_for_csv_plot[campo_json_original] = {"error": metrics_campo["error"]}


        # Plotar a matriz de confusão
        if "confusion_matrix" in metrics_campo and metrics_campo["confusion_matrix"] and np.array(metrics_campo["confusion_matrix"]).size > 0 :
            try:
                plot_confusion_matrix(
                    np.array(metrics_campo["confusion_matrix"]), 
                    metrics_campo.get("labels_used_for_metrics", []), 
                    f"Matriz de Confusão - {campo_json_original} ({model_name_from_results})",
                    metrics_plots_dir / f"cm_{safe_key}.png"
                )
            except Exception as e_plot:
                logger.error(f"Erro ao tentar plotar matriz de confusão para {safe_key}: {e_plot}")
        
        # Remover a matriz de confusão do dicionário de métricas ANTES de salvá-lo
        if "confusion_matrix" in metrics_campo:
            del metrics_campo["confusion_matrix"]
            logger.debug(f"Campo 'confusion_matrix' removido de metrics_campo para {safe_key} antes de salvar no JSON.")
        
        aggregated_metrics_results[safe_key] = metrics_campo

    # --- Calcular Médias Gerais das Métricas de Classificação ---

    overall_average_metrics = {}
    if metrics_for_csv_plot:
        # Listas para armazenar os scores de cada categoria válida
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []

        for category_name, mets_dict in metrics_for_csv_plot.items():
            if "error" not in mets_dict and mets_dict.get("accuracy") is not None: # Considerar apenas categorias sem erro e com métricas válidas
                accuracies.append(mets_dict["accuracy"])
                f1_scores.append(mets_dict.get("f1_score_weighted", 0.0)) # Usar 0.0 se a métrica não existir
                precisions.append(mets_dict.get("precision_weighted", 0.0))
                recalls.append(mets_dict.get("recall_weighted", 0.0))
        
        if accuracies: # Se houver alguma métrica válida para calcular a média
            overall_average_metrics[f"accuracy_{model_name_metrics}"] = np.mean(accuracies)
            overall_average_metrics[f"f1_score_weighted_{model_name_metrics}"] = np.mean(f1_scores)
            overall_average_metrics[f"precision_weighted_{model_name_metrics}"] = np.mean(precisions)
            overall_average_metrics[f"recall_weighted_{model_name_metrics}"] = np.mean(recalls)
            overall_average_metrics[f"num_categories_in_{model_name_metrics}"] = len(accuracies)
            logger.info(f"Métricas médias gerais calculadas sobre {len(accuracies)} categorias.")
        else:
            overall_average_metrics["error"] = "Nenhuma categoria com métricas válidas para calcular a média geral."
            logger.warning(overall_average_metrics["error"])
        
        # Adicionar as médias gerais ao dicionário principal de métricas agregadas
        aggregated_metrics_results[f"Metricas_Medias_Gerais_Do_Modelo_{model_name_metrics}"] = overall_average_metrics

    # --- Salvar Métricas em CSV e Plotar Gráfico de Barras para Categorias ---

    if metrics_for_csv_plot:
        # Criar DataFrame a partir das métricas coletadas
        # Primeiro, preparar os dados: cada categoria será uma linha no CSV
        csv_data_list = []
        valid_categories_for_plot = [] # Categorias que não tiveram erro
        for category, mets in metrics_for_csv_plot.items():
            if "error" not in mets:
                csv_data_list.append({
                    "Categoria": category,
                    "Accuracy": mets.get("accuracy"),
                    "F1-Score (Weighted)": mets.get("f1_score_weighted"),
                    "Precision (Weighted)": mets.get("precision_weighted"),
                    "Recall (Weighted)": mets.get("recall_weighted")
                })
                valid_categories_for_plot.append(category)
        
        if csv_data_list:
            df_metrics_summary = pd.DataFrame(csv_data_list)
            csv_output_path = eval_output_dir / f"evaluation_metrics_{model_name_metrics}_temp_{temp_safe}_{timestamp_run}" / f"summary_classification_metrics_{model_name_metrics}_temp_{temp_safe}.csv"
            
             # Adicionar a linha de média geral ao DataFrame ANTES de salvar
            if "error" not in overall_average_metrics and overall_average_metrics: # Verifica se as médias foram calculadas
                average_row = pd.DataFrame([{
                    "Categoria": "MÉDIA GERAL (Categorias)", # Nome para a linha da média
                    "Accuracy": overall_average_metrics.get(f"accuracy_{model_name_metrics}"),
                    "F1-Score (Weighted)": overall_average_metrics.get(f"f1_score_weighted_{model_name_metrics}"),
                    "Precision (Weighted)": overall_average_metrics.get(f"precision_weighted_{model_name_metrics}"),
                    "Recall (Weighted)": overall_average_metrics.get(f"recall_weighted_{model_name_metrics}")
                }])
                # Usar pd.concat para adicionar a nova linha
                df_metrics_summary = pd.concat([df_metrics_summary, average_row], ignore_index=True)
            
            try:
                df_metrics_summary.to_csv(csv_output_path, index=False, float_format='%.4f')
                logger.info(f"Sumário de métricas de classificação salvo em CSV: {csv_output_path}")
            except Exception as e_csv:
                logger.error(f"Erro ao salvar sumário de métricas em CSV: {e_csv}")

            # Preparar dados para o gráfico de barras (apenas categorias válidas)
            metrics_for_plot_filtered = {
                cat: metrics_for_csv_plot[cat] for cat in valid_categories_for_plot if "error" not in metrics_for_csv_plot[cat] and metrics_for_csv_plot[cat].get("accuracy") is not None
            }
            if metrics_for_plot_filtered:
                bar_plot_path = metrics_plots_dir / f"categories_metrics_{model_name_from_results.replace(':', '_').replace('/', '_')}.png"
                
                plot_classification_metrics_bars(
                    metrics_by_category=metrics_for_plot_filtered,
                    model_name=model_name_from_results,
                    output_path=bar_plot_path
                )
        else:
            logger.warning("Nenhuma categoria com métricas válidas para gerar gráfico de barras.")

    # --- Plotar Gráfico de Barras para Médias Gerais ---
    if "error" not in overall_average_metrics and overall_average_metrics.get(f"num_categories_in_{model_name_metrics}", 0) > 0:
        overall_avg_plot_path = metrics_plots_dir / f"bar_overall_average_metrics.png" # Nome do arquivo

        plot_overall_average_metrics_bar(
            overall_avg_metrics=overall_average_metrics, # Passa o dicionário com as médias
            model_name=model_name_from_results,
            output_path=overall_avg_plot_path
        )
    else:
        logger.warning("Não foi possível plotar o gráfico de médias gerais devido à ausência de dados ou erro no cálculo.")

    # --- Salvar Resultados da Avaliação ---
    evaluation_summary = {
        "metadata_avaliacao": {
            "timestamp_avaliacao": datetime.now(timezone.utc).isoformat(),
            "arquivo_resultados_gerados_avaliado": str(generated_results_file.name),
            "dataset_ground_truth_usado": str(dataset_csv_path.name),
            "modelo_avaliado": model_name_from_results,
            "temperatura": metadata_geracao.get("temperatura_usada"),
            "latencia_total_resposta_modelo_s": metadata_geracao.get("timestamp_final_run") or metadata_geracao.get("timestamp_total_run"),
            "total_itens_avaliados": len(resultados_gerados), # Ou len dos que efetivamente foram comparados
            "total_categorias_com_media_calculada": overall_average_metrics.get("num_categories_in_average", 0) if overall_average_metrics else 0
        },
        "Metricas_Agregadas_Por_Categoria": aggregated_metrics_results
    }

    eval_output_file = eval_output_dir / f"evaluation_metrics_{model_name_metrics}_temp_{temp_safe}.json"

    try:
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False, default=convert_numpy_types_for_json)
        logger.info(f"Métricas de avaliação salvas em: {eval_output_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar JSON de métricas de avaliação: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avalia resultados JSON gerados por um VLM benchmark.")
    parser.add_argument(
        "generated_results_file",
        type=Path,
        help="Caminho para o arquivo .json consolidado contendo os resultados da geração do modelo."
    )
    parser.add_argument(
        "--dataset_csv",
        type=Path,
        default=Path(DEFAULT_DATASET_CSV_PATH),
        help=f"Caminho para o arquivo CSV de ground truth. Padrão: {DEFAULT_DATASET_CSV_PATH}"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_EVALUATION_OUTPUT_DIR),
        help=f"Diretório para salvar os resultados da avaliação. Padrão: {DEFAULT_EVALUATION_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    try:
        run_evaluation(args.generated_results_file, args.dataset_csv, args.output_dir)
        logger.info("Avaliação concluída com sucesso!")
    except SystemExit:
        pass # Permitir saídas controladas se houver erro crítico no setup
    except Exception as e:
        logger.exception(f"Erro fatal durante a avaliação: {e}")
        sys.exit(1)