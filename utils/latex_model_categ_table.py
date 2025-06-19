'''
Gera tabelas em Latex com cabeçalhos (Modelo, Temperatura,
Categorias e Métricas Médias Por Categoria)
'''

import pandas as pd
from pathlib import Path
from loguru import logger
import argparse
import sys
from typing import List, Dict, Optional, Tuple # Adicionado Tuple
import numpy as np # Adicionado para np.floating
from dotenv import load_dotenv

# Caminho para o diretório raiz do projeto (assumindo que utils/ é um subdiretório direto)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_TEX_OUTPUT = PROJECT_ROOT / "tabelas_latex" # Onde salvar as métricas e plots

# Configuração básica do logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def escape_latex_special_chars(text: str, is_model_name: bool = False) -> str:
    """Escapa caracteres especiais do LaTeX em uma string."""
    if not isinstance(text, str):
        text = str(text)
    
    # Se for nome de modelo, primeiro reverter o underscore para dois pontos
    if is_model_name:
        text = text.replace('_', ':') # Ex: gemma3_27b -> gemma3:27b

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def df_to_latex_table_model_category_metrics(
    df: pd.DataFrame,
    caption: str = "Métricas de Desempenho por Modelo e Categoria",
    label: str = "tab:model_category_metrics",
    model_col: str = "Modelo",
    category_col: str = "Categoria",
    temp_col: Optional[str] = None,
    metric_cols: List[str] = ["Accuracy", "F1-Score", "Precision", "Recall"],
    float_format_spec: str = "%.2f",
    remove_toprule: bool = False
) -> str:
    r"""
    Converte um DataFrame (formato longo) para LaTeX com agrupamento por Modelo, depois Categoria.
    Cabeçalhos: Modelo, Categoria, [Temperatura opcional], Métricas (com sub-cabeçalhos).
    """
    if df.empty:
        return "% DataFrame vazio, nenhuma tabela gerada.\n"

    df_subset = df.copy()

    actual_metric_columns = [mc for mc in metric_cols if mc in df_subset.columns]
    num_metric_cols = len(actual_metric_columns)
    
    has_temp_col = temp_col and temp_col in df_subset.columns

    # Colunas base para o início da tabela
    base_header_cols = [model_col, category_col]
    if has_temp_col:
        base_header_cols.append(temp_col)
    
    num_base_cols = len(base_header_cols) # Número de colunas antes das métricas

    # Alinhamento: l para Modelo, l para Categoria, [l para Temp opcional], c para métricas
    col_alignments = ['l'] * num_base_cols + ['c'] * num_metric_cols
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

    # Cabeçalhos das Colunas - Linha 1 (Super-cabeçalhos)
    header_line1_parts = [f"\\textbf{{{escape_latex_special_chars(col_name, is_model_name=(col_name==model_col))}}}" for col_name in base_header_cols]
    if num_metric_cols > 0:
        header_line1_parts.append(f"\\multicolumn{{{num_metric_cols}}}{{c}}{{\\textbf{{Métricas}}}}")
    latex_string.append(f"    {' & '.join(header_line1_parts)} \\\\")

    # Linha \midrule abaixo dos super-cabeçalhos
    latex_string.append(r"    \midrule")

    # Cabeçalhos das Colunas - Linha 2 (Sub-cabeçalhos das métricas)
    header_line2_parts = [""] * num_base_cols # Células vazias para Modelo, Categoria, [Temp]
    for mc in actual_metric_columns:
        header_line2_parts.append(f"\\textbf{{{escape_latex_special_chars(mc)}}}")
    
    if num_metric_cols > 0:
      latex_string.append(f"    {' & '.join(header_line2_parts)} \\\\")
      cmidrule_start_col = num_base_cols + 1 
      cmidrule_end_col = cmidrule_start_col + num_metric_cols - 1
      if cmidrule_start_col <= cmidrule_end_col :
        latex_string.append(f"    \\cmidrule(lr){{{cmidrule_start_col}-{cmidrule_end_col}}}")
    
    # Dados das Linhas
    last_model_val = None
    last_category_val = None
    
    model_counts = df_subset[model_col].value_counts().sort_index().to_dict()
    # Se temp_col não existe, category_counts_per_model agrupa apenas por model e category
    grouping_for_category_counts = [model_col, category_col]
    
    category_row_counts = df_subset.groupby([model_col, category_col]).size().to_dict()


    is_first_row_of_any_group = True

    for index, row in df_subset.iterrows():
        row_values_latex = []
        current_model_val = row[model_col]
        current_category_val = row[category_col]
        
        if current_model_val != last_model_val and not is_first_row_of_any_group:
            latex_string.append(r"    \midrule")
        
        if current_model_val != last_model_val:
             is_first_row_of_any_group = False

        # Coluna Modelo (com \multirow)
        if current_model_val != last_model_val:
            count_model = model_counts.get(current_model_val, 1)
            display_model_name = escape_latex_special_chars(str(current_model_val), is_model_name=True)
            row_values_latex.append(f"\\multirow{{{count_model}}}{{*}}{{\\textbf{{{display_model_name}}}}}")
            last_model_val = current_model_val
            last_category_val = None # Resetar categoria ao mudar de modelo
        else:
            row_values_latex.append("")

        # Coluna Categoria (com \multirow dentro de cada grupo de Modelo)
        if current_category_val != last_category_val:
            count_category = category_row_counts.get((current_model_val, current_category_val), 1)
            row_values_latex.append(f"\\multirow{{{count_category}}}{{*}}{{{escape_latex_special_chars(str(current_category_val))}}}")
            last_category_val = current_category_val
        else:
            row_values_latex.append("")
            
        # Coluna Temperatura (se existir)
        if has_temp_col:
            current_temp_val = row[temp_col]
            if isinstance(current_temp_val, (float, np.floating)):
                row_values_latex.append(float_format_spec % current_temp_val)
            else:
                row_values_latex.append(escape_latex_special_chars(str(current_temp_val)))

        # Colunas de Métricas
        for mc in actual_metric_columns:
            value = row[mc]
            if isinstance(value, (float, np.floating)):
                row_values_latex.append(float_format_spec % value)
            else:
                row_values_latex.append(escape_latex_special_chars(str(value)))

        latex_string.append(f"    {' & '.join(row_values_latex)} \\\\")

    latex_string.append(r"    \bottomrule")
    latex_string.append(r"  \end{tabular}")
    latex_string.append(r"\end{table}")

    return "\n".join(latex_string)

def main(csv_file_path: Path, output_latex_file: Path, no_top_rule: bool, no_temp: bool): # Adicionado no_temp
    if not csv_file_path.is_file():
        logger.error(f"Arquivo CSV de entrada não encontrado: {csv_file_path}")
        return

    try:
        df_metrics = pd.read_csv(csv_file_path)
        logger.info(f"Lido arquivo CSV com {len(df_metrics)} linhas: {csv_file_path}")
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo CSV {csv_file_path}: {e}")
        return

    # Ordenação base
    sort_columns = ["Modelo", "Categoria"]
    if "Temperatura" in df_metrics.columns and not no_temp:
        df_metrics["Temperatura_num_sort"] = pd.to_numeric(df_metrics["Temperatura"], errors='coerce')
        sort_columns.append("Temperatura_num_sort")
        
    df_metrics.sort_values(by=sort_columns, inplace=True)
    if "Temperatura_num_sort" in df_metrics.columns:
        df_metrics.drop(columns=["Temperatura_num_sort"], inplace=True)

    # Colunas a serem incluídas na tabela e sua ordem
    columns_for_table = ["Modelo", "Categoria"]
    if "Temperatura" in df_metrics.columns and not no_temp:
        columns_for_table.append("Temperatura")
    
    # Nomes exatos das colunas de métricas no seu CSV
    metric_col_names_in_df = ["Accuracy", "F1-Score", "Precision", "Recall"]
    actual_metric_col_names_in_df = [mc for mc in metric_col_names_in_df if mc in df_metrics.columns]
    columns_for_table.extend(actual_metric_col_names_in_df)
    
    # Filtrar df_metrics para conter apenas as colunas que estarão na tabela
    # e na ordem correta
    df_metrics_for_table = df_metrics[[col for col in columns_for_table if col in df_metrics.columns]].copy()


    temp_column_name_for_func = "Temperatura" if "Temperatura" in df_metrics_for_table.columns and not no_temp else None

    latex_table_string = df_to_latex_table_model_category_metrics(
        df_metrics_for_table, # Passar o DataFrame com as colunas selecionadas e ordenadas
        caption="Comparativo de Desempenho de Modelos VLM por Categoria" + (" e Temperatura" if temp_column_name_for_func else ""),
        label="tab:vlm_model_category_metrics_final",
        model_col="Modelo",
        category_col="Categoria",
        temp_col=temp_column_name_for_func, # Passar None se não houver temperatura
        metric_cols=actual_metric_col_names_in_df,
        remove_toprule=no_top_rule
    )

    try:
        output_latex_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_latex_file, "w", encoding="utf-8") as f:
            f.write(latex_table_string)
        logger.info(f"Código LaTeX da tabela (Modelo-Categoria) salvo em: {output_latex_file}")
        print("\nCódigo LaTeX Gerado (Modelo-Categoria):\n")
        print(latex_table_string)
        print(f"\nSalvo em: {output_latex_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar o arquivo LaTeX {output_latex_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera código LaTeX para uma tabela de métricas (Modelo-Categoria) a partir de um CSV.")
    
    parser.add_argument(
        "csv_input_file",
        type=Path,
        help="Caminho para o CSV de entrada."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_TEX_OUTPUT / "tabela_metricas_categ.tex"),
        help="Caminho para o arquivo .tex de saída."
    )
    parser.add_argument(
        "--no-top-rule",
        action="store_true",
        help="Remove a linha \\toprule da tabela."
    )
    parser.add_argument(
        "--no-temp",
        action="store_true",
        help="Remove a coluna Temperatura da tabela."
    )
    args = parser.parse_args()

    main(args.csv_input_file, args.output, args.no_top_rule, args.no_temp)