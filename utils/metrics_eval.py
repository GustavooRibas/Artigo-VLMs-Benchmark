'''
Script auxiliar para a geração de métricas e gráficos
'''

import json
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    multilabel_confusion_matrix # Para Benfeitorias se tratado como multilabel
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use('Agg') # Define o backend para Agg ANTES de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import pandas as pd
from collections import Counter

# --- Carregar Modelo Semântico ---
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2' # Ou outro de sua preferência
semantic_model = None
try:
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    logger.info(f"Modelo semântico '{SEMANTIC_MODEL_NAME}' carregado para métricas.")
except Exception as e:
    logger.warning(f"Falha ao carregar modelo semântico '{SEMANTIC_MODEL_NAME}': {e}. Similaridade semântica não disponível.")

def normalize_text_for_comparison(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    return ' '.join(text.lower().strip().split())

def calculate_semantic_similarity(text1: str, text2: str) -> Optional[float]:
    if not semantic_model or not text1 or not text2:
        return None
    try:
        emb1 = semantic_model.encode(normalize_text_for_comparison(text1))
        emb2 = semantic_model.encode(normalize_text_for_comparison(text2))
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    except Exception as e:
        logger.warning(f"Erro ao calcular similaridade semântica: {e}")
        return None

def evaluate_categorical_field(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None # Todas as classes possíveis, incluindo "NA"
) -> Dict[str, Any]:
    
    """Avalia um campo categórico com métricas de classificação."""

    if not y_true or not y_pred:
        return {"error": "Listas de entrada vazias."}
    if len(y_true) != len(y_pred): # Esta checagem é importante
        # Logar mais detalhes para depuração
        logger.error(f"Erro de avaliação: Listas y_true (len {len(y_true)}) e y_pred (len {len(y_pred)}) têm tamanhos diferentes.")
        # Para facilitar o debug, você pode logar as primeiras N entradas de cada lista
        # logger.debug(f"y_true (primeiros 10): {y_true[:10]}")
        # logger.debug(f"y_pred (primeiros 10): {y_pred[:10]}")
        return {"error": f"Listas de entrada com tamanhos diferentes: GT={len(y_true)}, Pred={len(y_pred)}"}


    # Normalizar para consistência (embora as opções da tabela devam ser usadas)
    y_true_norm = [normalize_text_for_comparison(str(s)) for s in y_true]
    y_pred_norm = [normalize_text_for_comparison(str(s)) for s in y_pred]

    if not labels: # Garantir que todos os labels únicos de y_true e y_pred sejam considerados.
        unique_labels_from_data = set(y_true_norm + y_pred_norm)
        labels = sorted(list(unique_labels_from_data))
    
    if not labels: # Se ainda vazio (ex: todas as entradas eram None ou vazias e resultaram em strings vazias)
        return {"error": "Nenhum label encontrado para avaliação após normalização."}

    # Calcular métricas gerais ponderadas
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true_norm, y_pred_norm, average='weighted', labels=labels, zero_division=0
    )
    accuracy = accuracy_score(y_true_norm, y_pred_norm)
    
    # Calcular métricas por classe
    p_class, r_class, f1_class, s_class_true = precision_recall_fscore_support(
        y_true_norm, y_pred_norm, labels=labels, zero_division=0
    ) # s_class_true é o support (contagem de y_true)

    # --- Contar predições por classe ---
    pred_counts = Counter(y_pred_norm)
    # ---------------------------------------

    metrics_per_class = {}
    for i, label in enumerate(labels):
        metrics_per_class[label] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1-score": float(f1_class[i]),
            "support_ground_truth": int(s_class_true[i]), # Renomeado para clareza
            "support_predicted": int(pred_counts.get(label, 0)) # <<< ADICIONADO: Contagem de predições para este label
        }

    # Matriz de confusão
    cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_score_weighted": float(f1_w),
        "labels_used_for_metrics": labels, # Labels efetivamente usados para as métricas e CM
        "confusion_matrix": cm.tolist(),
        "metrics_per_class": metrics_per_class
    }

def plot_classification_metrics_bars(
    metrics_by_category: Dict[str, Dict[str, float]], # Ex: {"Estrutura": {"accuracy": 0.8, "f1_score_weighted": 0.75, ...}}
    model_name: str,
    output_path: Path,
    figsize: Tuple[int, int] = (15, 8),
    metric_names: List[str] = ["accuracy", "f1_score_weighted", "precision_weighted", "recall_weighted"]
):
    """
    Plota um gráfico de barras comparando métricas de classificação (accuracy, f1, precision, recall)
    para diferentes categorias.

    Args:
        metrics_by_category: Dicionário onde as chaves são nomes de categorias (campos avaliados)
                             e os valores são dicionários contendo as métricas.
        model_name: Nome do modelo para incluir no título do gráfico.
        output_path: Caminho para salvar o gráfico gerado.
        figsize: Tamanho da figura do gráfico.
        metric_names: Lista das métricas a serem plotadas (devem ser chaves nos dicts de métricas).
    """
    try:
        categories = list(metrics_by_category.keys())
        if not categories:
            logger.warning("Nenhuma categoria fornecida para plotar métricas em barras.")
            return

        # Preparar dados para o DataFrame do pandas
        plot_data = []
        for cat in categories:
            for metric_name in metric_names:
                # Usar .get() para evitar KeyError se uma métrica estiver ausente para uma categoria
                metric_value = metrics_by_category[cat].get(metric_name)
                if metric_value is not None: # Apenas adicionar se a métrica existir
                    plot_data.append({"Categoria": cat, "Métrica": metric_name, "Valor": metric_value})
        
        if not plot_data:
            logger.warning("Nenhum dado de métrica válido encontrado para plotar o gráfico de barras.")
            return

        df_plot = pd.DataFrame(plot_data)

        plt.figure(figsize=figsize)
        sns.barplot(x="Categoria", y="Valor", hue="Métrica", data=df_plot, palette="viridis")
        
        plt.title(f"Métricas de Classificação por Categoria ({model_name})", fontsize=16)
        plt.xlabel("Categoria Avaliada", fontsize=12)
        plt.ylabel("Score da Métrica", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10) # Rotaciona os labels do eixo x para melhor visualização
        plt.yticks(fontsize=10)
        plt.legend(title="Métrica", fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Ajusta o layout para evitar sobreposição

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Gráfico de barras de métricas salvo em: {output_path}")

    except Exception as e:
        logger.error(f"Erro ao plotar gráfico de barras de métricas: {e}")

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plota e salva uma matriz de confusão."""
    try:
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Matriz de confusão salva em: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de confusão para '{title}': {e}")


def evaluate_benfeitorias(
    gt_benfeitorias_list: List[List[str]], # Lista de listas de benfeitorias verdadeiras
    pred_benfeitorias_list: List[List[str]], # Lista de listas de benfeitorias preditas
    all_possible_benfeitorias: List[str] # Lista de todas as benfeitorias possíveis (para binarização)
) -> Optional[Dict[str, Any]]:
    """
    Avalia o campo 'Benfeitorias' (multilabel).
    Normaliza os itens antes da comparação.
    """
    if len(gt_benfeitorias_list) != len(pred_benfeitorias_list):
        logger.error("Benfeitorias: GT e Pred listas têm tamanhos diferentes.")
        return None

    # Normalizar todas as benfeitorias possíveis
    all_possible_benfeitorias_norm = sorted([normalize_text_for_comparison(b) for b in all_possible_benfeitorias])

    # Criar vetores binários (MultiLabelBinarizer poderia ser usado aqui também)
    y_true_bin = []
    y_pred_bin = []

    for gt_items_raw, pred_items_raw in zip(gt_benfeitorias_list, pred_benfeitorias_list):
        gt_items = set(normalize_text_for_comparison(item) for item in gt_items_raw if item)
        pred_items = set(normalize_text_for_comparison(item) for item in pred_items_raw if item)

        y_true_bin.append([1 if ben in gt_items else 0 for ben in all_possible_benfeitorias_norm])
        y_pred_bin.append([1 if ben in pred_items else 0 for ben in all_possible_benfeitorias_norm])

    if not y_true_bin or not y_pred_bin: # Se todas as listas de benfeitorias estavam vazias
        return {
            "precision_micro": 1.0, "recall_micro": 1.0, "f1_score_micro": 1.0,
            "precision_macro": 1.0, "recall_macro": 1.0, "f1_score_macro": 1.0,
            "accuracy_subset": 1.0, # Acerto exato do conjunto de labels
            "note": "Todas as listas de benfeitorias estavam vazias ou resultaram em vetores vazios."
        }

    y_true_np = np.array(y_true_bin)
    y_pred_np = np.array(y_pred_bin)

    # Métricas Micro (agregadas globalmente)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='micro', zero_division=0
    )
    # Métricas Macro (média não ponderada por classe)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='macro', zero_division=0
    )
    # Accuracy de subconjunto (exact match ratio)
    subset_accuracy = accuracy_score(y_true_np, y_pred_np)

    # Matriz de confusão multilabel (uma matriz por label/benfeitoria)
    mcm = multilabel_confusion_matrix(y_true_np, y_pred_np)

    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_score_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_score_macro": float(f1_macro),
        "accuracy_subset": float(subset_accuracy),
        "labels_benfeitorias": all_possible_benfeitorias_norm,
        "multilabel_confusion_matrices": mcm.tolist() # mcm.tolist() deve ser ok
    }

def plot_overall_average_metrics_bar(
    overall_avg_metrics: Dict[str, float], # Ex: {"mean_accuracy": 0.85, "mean_f1_score_weighted": 0.80, ...}
    model_name: str,
    output_path: Path,
    figsize: Tuple[int, int] = (8, 6),
    title_prefix: str = "Médias Gerais de Classificação"
):
    """
    Plota um gráfico de barras para as métricas médias gerais do modelo.

    Args:
        overall_avg_metrics: Dicionário contendo as métricas médias gerais.
                             As chaves devem ser nomes de métricas (ex: "mean_accuracy")
                             e os valores devem ser os scores numéricos.
        model_name: Nome do modelo para incluir no título do gráfico.
        output_path: Caminho para salvar o gráfico gerado.
        figsize: Tamanho da figura do gráfico.
        title_prefix: Prefixo para o título do gráfico.
    """
    model_name_metrics = model_name.replace(':', '_').replace('/', '_')

    try:
        # Filtrar apenas as métricas que queremos plotar e que não sejam erros ou contagens
        metrics_to_plot = {
            "Accuracy": overall_avg_metrics.get(f"accuracy_{model_name_metrics}"),
            "F1-Score (Weighted)": overall_avg_metrics.get(f"f1_score_weighted_{model_name_metrics}"),
            "Precision (Weighted)": overall_avg_metrics.get(f"precision_weighted_{model_name_metrics}"),
            "Recall (Weighted)": overall_avg_metrics.get(f"recall_weighted_{model_name_metrics}")
        }
        
        # Remover métricas que são None (se alguma não foi calculada)
        plot_data = {k: v for k, v in metrics_to_plot.items() if v is not None}

        if not plot_data:
            logger.warning("Nenhuma métrica média geral válida encontrada para plotar.")
            return

        metric_names = list(plot_data.keys())
        metric_values = list(plot_data.values())

        plt.figure(figsize=figsize)
        bars = plt.bar(metric_names, metric_values, color=sns.color_palette("YlGnBu", len(metric_names)))

        # Adicionar os valores no topo de cada barra
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

        plt.title(f"{title_prefix} ({model_name})", fontsize=15)
        plt.ylabel("Score da Métrica", fontsize=12)
        plt.xlabel("Métrica Média Geral", fontsize=12) # Ou remover xlabel se os nomes das barras forem suficientes
        plt.xticks(rotation=15, ha="right", fontsize=10) # Pequena rotação se os nomes das métricas forem longos
        plt.ylim(0, 1.05) # Definir limite do eixo Y (0 a 1 para métricas comuns, mais um pouco de espaço)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Gráfico de barras de médias gerais salvo em: {output_path}")

    except Exception as e:
        logger.error(f"Erro ao plotar gráfico de barras de médias gerais: {e}")

OPCOES_ESTRUTURA = ["Alvenaria", "Concreto", "Mista", "Madeira Tratada", "Metálica", "Adobe/Taipa/Rudimentar", "NA"]
OPCOES_ESQUADRIAS = ["Ferro", "Alumínio", "Madeira", "Rústica", "Especial", "Sem", "NA"]
OPCOES_PISO = ["Cerâmica", "Cimento", "Taco", "Tijolo", "Terra", "Especial/Porcelanato", "NA"]
OPCOES_FORRO = ["Laje", "Madeira", "Gesso Simples/Pvc", "Especial", "Sem", "NA"]
OPCOES_INSTALACAO_ELETRICA = ["Embutida", "Semi Embutida", "Externa", "Sem", "NA"]
OPCOES_INSTALACAO_SANITARIA = ["Interna", "Completa", "Mais de uma", "Externa", "Sem", "NA"]
OPCOES_REVESTIMENTO_INTERNO = ["Reboco", "Massa", "Material Cerâmico", "Especial", "Sem", "NA"]
OPCOES_ACABAMENTO_INTERNO = ["Pintura Lavável", "Pintura Simples", "Caiação", "Especial", "Sem", "NA"]
OPCOES_REVESTIMENTO_EXTERNO = ["Reboco", "Massa", "Material Cerâmico", "Especial", "Sem", "NA"]
OPCOES_ACABAMENTO_EXTERNO = ["Pintura Lavável", "Pintura Simples", "Caiação", "Especial", "Sem", "NA"]
OPCOES_COBERTURA = ["Telha de Barro", "Fibrocimento", "Alumínio", "Zinco", "Laje", "Palha", "Especial", "Sem", "NA"]
OPCOES_TIPO_IMOVEL = ["Residencial", "Comercial", "Industrial", "NA"]

LISTA_TODAS_BENFEITORIAS_POSSIVEIS = [
    "Piscina", "Sauna", "Home Cinema", "Churrasqueira coletiva", 
    "Churrasqueira privativa", "Quadra de poliesportiva", "Quadra de tênis",
    "Playground", "Brinquedoteca", "Elevador", "Energia solar", "Academia de ginástica",
    "Salão de festas", "Espaço gourmet", "Gerador", "Heliponto", "Escaninhos",
    "Mais de dois box de garagem", "Laje técnica", "Sala Reunião", "Coworking",
    "Isolamento acústico", "Rede Frigorígena", "Mais de uma suíte", "Lavabo"
]

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