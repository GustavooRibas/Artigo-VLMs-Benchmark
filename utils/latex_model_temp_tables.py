'''
Gera tabelas em Latex com cabeçalhos dinâmicos (Modelo, Temperatura,
Tempo de Execução (s) e Métricas Médias Gerais)
'''

import pandas as pd
from pathlib import Path
from loguru import logger
import argparse
import sys
from typing import List, Dict, Optional
import numpy as np

# --- Configuração de Paths e Logger ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_TEX_OUTPUT_DIR = PROJECT_ROOT / "tabelas_latex"

logger.remove()
logger.add(sys.stderr, level="INFO")

def escape_latex_special_chars(text: str, is_model_name: bool = False) -> str:

    if not isinstance(text, str): text = str(text)
    if is_model_name: text = text.replace('_', ':')
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
    }
    for old, new in replacements.items(): text = text.replace(old, new)
    return text

# --- Função de Geração da Tabela ---
def df_to_latex_table_final_style(
    df: pd.DataFrame,
    caption: str,
    label: str,
    group_by_column: str = "Modelo",
    other_header_cols: List[str] = [], # Será preenchida dinamicamente
    metric_columns: List[str] = [], # Será preenchida dinamicamente
    column_display_names: Optional[Dict[str, str]] = None,
    float_format_spec: str = "%.2f",
    remove_toprule: bool = False
) -> str:
    r"""
    Converte um DataFrame para LaTeX com cabeçalhos e agrupamento.
    Estrutura flexível para colunas intermediárias e de métricas.
    """
    if df.empty: return "% DataFrame vazio.\n"

    # Selecionar e ordenar as colunas que realmente existem no DataFrame
    cols_to_render = [group_by_column] + \
                     [col for col in other_header_cols if col in df.columns] + \
                     [mc for mc in metric_columns if mc in df.columns]
    df_subset = df[[col for col in cols_to_render if col in df.columns]].copy()

    actual_metric_columns = [mc for mc in metric_columns if mc in df_subset.columns]
    num_metric_cols = len(actual_metric_columns)
    
    actual_other_header_cols = [col for col in other_header_cols if col in df_subset.columns]
    num_other_cols = len(actual_other_header_cols)

    # --- Alinhamento Centralizado ---
    # l para Modelo (primeira coluna), c para todas as outras colunas (Temp, Tempo, Métricas)
    col_alignments = ['l'] # Modelo
    col_alignments.extend(['c'] * (num_other_cols + num_metric_cols)) # Todas as outras centralizadas
    col_format = "".join(col_alignments)

    latex_string = []
    latex_string.append(r"% Certifique-se de ter \usepackage{booktabs} e \usepackage{multirow} no seu preâmbulo LaTeX")
    latex_string.append(r"\begin{table}[htbp]")
    latex_string.append(r"  \centering")
    latex_string.append(f"  \\caption{{{caption}}}")
    latex_string.append(f"  \\label{{{label}}}")
    latex_string.append(f"  \\")
    latex_string.append(f"  \\begin{{tabular}}{{{col_format}}}")
    if not remove_toprule:
        latex_string.append(r"    \toprule")

    # Mapeamento de nomes para exibição
    effective_display_names = column_display_names if column_display_names else {}
    for col in df_subset.columns: # Garantir que todos os nomes de colunas tenham um mapeamento (fallback para si mesmo)
        if col not in effective_display_names:
            effective_display_names[col] = col

    # Linha 1 do Cabeçalho
    header_line1_parts = []
    header_line1_parts.append(f"\\textbf{{{escape_latex_special_chars(effective_display_names[group_by_column])}}}")
    for col_name in actual_other_header_cols:
        header_line1_parts.append(f"\\textbf{{{escape_latex_special_chars(effective_display_names[col_name])}}}")
    if num_metric_cols > 0:
        header_line1_parts.append(f"\\multicolumn{{{num_metric_cols}}}{{c}}{{\\textbf{{Métricas}}}}")
    latex_string.append(f"    {' & '.join(header_line1_parts)} \\\\")

    if num_metric_cols > 0:
        latex_string.append(r"    \midrule")
        header_line2_parts = [""] * (1 + num_other_cols)
        for mc in actual_metric_columns:
            display_name = effective_display_names[mc].replace(r"\\", " ")
            header_line2_parts.append(f"\\textbf{{{escape_latex_special_chars(display_name)}}}")
        latex_string.append(f"    {' & '.join(header_line2_parts)} \\\\")
        cmidrule_start_col = 1 + num_other_cols + 1
        cmidrule_end_col = cmidrule_start_col + num_metric_cols - 1
        if cmidrule_start_col <= cmidrule_end_col:
            latex_string.append(f"    \\cmidrule(lr){{{cmidrule_start_col}-{cmidrule_end_col}}}")
    else:
        latex_string.append(r"    \midrule")
    
    # Dados das Linhas
    last_group_val = None
    group_counts = df_subset[group_by_column].value_counts().sort_index().to_dict()
    is_first_row_of_any_group = True

    for index, row in df_subset.iterrows():
        row_values_latex = []
        current_group_val = row[group_by_column]
        if current_group_val != last_group_val and not is_first_row_of_any_group:
            latex_string.append(r"    \midrule")
        is_first_row_of_any_group = False
        if current_group_val != last_group_val:
            count = group_counts.get(current_group_val, 1)
            display_model_name_data = escape_latex_special_chars(str(current_group_val), is_model_name=True)
            row_values_latex.append(f"\\multirow{{{count}}}{{*}}{{\\textbf{{{display_model_name_data}}}}}")
            last_group_val = current_group_val
        else:
            row_values_latex.append("")
        
        data_cols_for_row = [col for col in cols_to_render if col != group_by_column]
        for col_name in data_cols_for_row:
            value = row[col_name]
            if isinstance(value, (float, np.floating)):
                row_values_latex.append(float_format_spec % value)
            else:
                row_values_latex.append(escape_latex_special_chars(str(value)))
        latex_string.append(f"    {' & '.join(row_values_latex)} \\\\")

    latex_string.append(r"    \bottomrule")
    latex_string.append(r"  \end{tabular}")
    latex_string.append(r"\end{table}")
    return "\n".join(latex_string)


# --- Função main ---
def main(csv_file_path: Path, output_latex_file: Path, no_top_rule: bool, no_metrics: bool, no_temperature: bool, no_exec_time: bool): # NOVOS PARÂMETROS
    if not csv_file_path.is_file(): logger.error(f"Arquivo CSV não encontrado: {csv_file_path}"); return
    try:
        df_metrics = pd.read_csv(csv_file_path)
        logger.info(f"Lido arquivo CSV com {len(df_metrics)} linhas: {csv_file_path}")
    except Exception as e: logger.error(f"Erro ao ler CSV {csv_file_path}: {e}"); return

    # Ordenar o DataFrame
    sort_columns = ["Modelo"]
    if "Temperatura" in df_metrics.columns and not no_temperature:
        df_metrics["Temperatura_num_sort"] = pd.to_numeric(df_metrics["Temperatura"], errors='coerce')
        sort_columns.append("Temperatura_num_sort")
    df_metrics.sort_values(by=sort_columns, inplace=True, kind='mergesort', na_position='last')
    if "Temperatura_num_sort" in df_metrics.columns:
        df_metrics.drop(columns=["Temperatura_num_sort"], inplace=True)

    # --- Lógica Flexível de Colunas ---

    flag = "_com_temp"
    
    # Definir os nomes exatos das colunas no DataFrame
    time_col_name_df = "Tempo de Execucao Total (s)"
    temp_col_name_df = "Temperatura"
    metric_col_names_df = ["Accuracy", "F1-Score", "Precision", "Recall"]

    # Construir a lista de colunas intermediárias (other_header_cols) com base nas flags
    other_cols_for_header = []
    if temp_col_name_df in df_metrics.columns and not no_temperature:
        other_cols_for_header.append(temp_col_name_df)
    if time_col_name_df in df_metrics.columns and not no_exec_time:
        other_cols_for_header.append(time_col_name_df)
    if no_exec_time:
        flag += "_sem_tempo_exec"

    # Decidir quais colunas de métricas usar com base na flag --no-metrics
    if no_metrics:
        actual_metric_col_names_df = []
        flag += "_sem_metricas"
        logger.info("Opção --no-metrics selecionada. A tabela não incluirá colunas de métricas.")
    else:
        actual_metric_col_names_df = [mc for mc in metric_col_names_df if mc in df_metrics.columns]
        logger.info(f"Incluindo colunas de métricas: {actual_metric_col_names_df}")
    
    # Mapeamento para nomes de exibição (mantido completo, não causa problema se colunas não forem usadas)
    column_display_names_map = {
        "Modelo": "Modelo",
        "Temperatura": "Temp.",
        time_col_name_df: "Tempo Exec. (s)", # Cabeçalho de duas linhas
        "Accuracy": "Accuracy",
        "F1-Score": "F1-Score",
        "Precision": "Precision",
        "Recall": "Recall"
    }

    # Construir legenda dinâmica
    caption_parts = ["Desempenho dos Modelos VLM"]
    if "Temperatura" in other_cols_for_header:
        caption_parts.append("por Temperatura")
    if "Tempo de Execucao Total (s)" in other_cols_for_header:
        caption_parts.append("e Tempo de Execução")
    table_caption = " ".join(caption_parts)


    # Gerar a string LaTeX da tabela
    latex_table_string = df_to_latex_table_final_style(
        df_metrics,
        caption=table_caption,
        label=f"tab:vlm_performance_report{flag}",
        group_by_column="Modelo",
        other_header_cols=other_cols_for_header,
        metric_columns=actual_metric_col_names_df,
        column_display_names=column_display_names_map,
        remove_toprule=no_top_rule
    )

    # Salvar a tabela em um arquivo .tex
    try:
        output_latex_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_latex_file, "w", encoding="utf-8") as f:
            f.write(latex_table_string)
        logger.info(f"Código LaTeX da tabela salvo em: {output_latex_file}")
        print("\nCódigo LaTeX Gerado:\n")
        print(latex_table_string)
        print(f"\nSalvo em: {output_latex_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar o arquivo LaTeX {output_latex_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera código LaTeX para tabelas de métricas/tempo a partir de um CSV.")
    parser.add_argument("csv_input_file", type=Path, help="Caminho para o CSV de entrada.")
    parser.add_argument("--output", "-o", type=Path, help="Caminho para o arquivo .tex de saída. Padrão: tabelas_latex/tabela_gerada.tex")
    parser.add_argument("--no-top-rule", action="store_true", help="Remove a linha \\toprule da tabela.")
    parser.add_argument("--no-metrics", action="store_true", help="Remove as colunas de métricas (Accuracy, F1, etc.).")
    parser.add_argument("--no-temperature", action="store_true", help="Remove a coluna de Temperatura.")
    parser.add_argument("--no-exec-time", action="store_true", help="Remove a coluna de Tempo de Execução.")

    args = parser.parse_args()

    DEFAULT_TEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    suffix = "sem_temp" if args.no_temperature else "com_temp"
    suffix += "_sem_metricas" if args.no_metrics else "_com_metricas"
    suffix += "_sem_exec_tempo" if args.no_exec_time else "_com_exec_tempo"
    output_filename = DEFAULT_TEX_OUTPUT_DIR / f"tabela_{suffix}.tex"
    logger.info(f"Nome de arquivo de saída não especificado, usando padrão: {output_filename}")
    
    main(args.csv_input_file, output_filename, args.no_top_rule, args.no_metrics, args.no_temperature, args.no_exec_time)