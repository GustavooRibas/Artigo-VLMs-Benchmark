# Projeto de Benchmark e Avaliação de Modelos VLM para Análise de Imóveis

Este projeto fornece um pipeline completo para avaliar o desempenho de Modelos de Linguagem de Visão (VLMs), como as famílias Gemini, Qwen, Gemma e Llava, em uma tarefa de análise técnica de imagens de imóveis. O pipeline consiste em três etapas principais:
1.  **Geração:** Executa um ou mais modelos VLM com diferentes configurações para analisar conjuntos de imagens de imóveis e gerar saídas estruturadas em formato JSON.
2.  **Avaliação:** Compara as respostas JSON geradas com um dataset de ground truth para calcular métricas de desempenho detalhadas (accuracy, F1-score, etc.).
3.  **Relatório:** Agrega os resultados de múltiplas execuções de avaliação para criar relatórios sumarizados em formato CSV e tabelas LaTeX.

## Estrutura de Diretórios

O projeto está organizado da seguinte forma:

```
vlm_benchmark/
├── benchmark_gemini_multimodels.py    # Script principal para GERAÇÃO de respostas
├── benchmark_ollama_multimodels.py    # Script principal para GERAÇÃO de respostas
├── gemini_connector.py                # Conector para a API Gemini (requer langchain-google-genai)
├── ollama_connector.py                # Conector para a API Ollama
├── utils/
│   ├── __init__.py                    # Torna 'utils' um pacote Python
│   ├── helpers.py                     # Funções auxiliares (ex: get_images_from_directory)
│   ├── metrics_eval.py                # Funções para cálculo de métricas (accuracy, F1, etc.)
│   ├── evaluate_results.py            # Script para AVALIAÇÃO (lê a saída da geração)
│   └── create_summary_report.py       # Script para RELATÓRIOS (.csv) (lê a saída da avaliação)
│   └── latex_model_categ_table.py     # Script para gerar tabelas (Modelo, Categoria e Métricas) em LaTeX (lê o .csv gerado nos RELATÓRIOS)
|   └── latex_model_temp_tables.py      # Script para gerar tabelas (Modelo, Temperatura, Tempo de Execução e Métricas) em LaTeX (lê o .csv gerado nos RELATÓRIOS)
│
├── prompts/
│   ├── human_prompt_gemini.txt        # Prompt humano com as instruções da tarefa (Gemini)
│   ├── human_prompt_gemma.txt         # Prompt humano com as instruções da tarefa (Gemma3)
│   ├── human_prompt_qwen.txt          # Prompt humano com as instruções da tarefa (Qwen2.5)
│   ├── human_prompt_llava.txt         # Prompt humano com as instruções da tarefa (Llava)
│   ├── system_prompt_gemini.txt       # Prompt de sistema definindo o papel do VLM (Gemini)
│   └── system_prompt.txt              # Prompt de sistema definindo o papel do VLM (Ollama)
│
├── data/
│   ├── benchmark_50_anotacao_ollama.csv      # Dataset com ground truth das características dos imóveis
│   ├── benchmark_50_anotacao_gemini.csv      # Dataset com ground truth das características dos imóveis
│   └── images/
│       ├── 1_algum_id/                # Subpasta para cada imóvel com suas imagens
│       │   ├── img1.jpg
│       │   └── ...
│       └── ...
│
├── logs/                              # Diretório para arquivos de log (criado automaticamente)
│
├── resultados_geracao/                # Onde os resultados da GERAÇÃO são salvos
│   └── output_MODEL_temp_TEMPERATURE_TIMESTAMP/
│       └── generation_results.json
│
├── resultados_avaliacao/              # Onde os resultados da AVALIAÇÃO são salvos
|   ├── evaluation_metrics_MODEL_temp_TEMPERATURE.json
│   └── evaluation_output_MODEL_temp_TEMP_TIMESTAMP/
│       ├── summary_classification_metrics_MODEL_temp_TEMPERATURE.csv
│       └── metric_plots_MODEL_temp_TEMPERATURE/
│           ├── cm_*.png
│           └── bar_*.png
|           └── ...
│
├── resultados_sumarizados/            # Onde os RELATÓRIOS finais são salvos
|   ├── report_metrics_by_category_TIMESTAMP.csv
|   └── report_overall_model_averages_TIMESTAMP.csv
|   └── best_models_report_metrics_by_category_TIMESTAMP.csv
|   └── all_models_report_metrics_by_category_TIMESTAMP.csv
|
└── tabelas_latex/                     # Onde as TABELAS em .tex são salvas
    ├── tabela_com_temp_com_metricas_sem_exec_tempo.tex # Tabela com Modelos, Temperaturas e Métricas
    ├── tabela_com_temp_sem_metricas_com_exec_tempo.tex # Tabela com Modelos, Temperaturas e Tempo de Execução
    └── tabela_melhores_modelos_metricas_categ.tex      # Tabela com os Melhores Modelos, Categorias e Métricas
    └── tabela_todos_modelos_metricas_categ.tex         # Tabela com Todos Modelos, Categorias e Métricas
```

## Configuração do Ambiente

1.  **Pré-requisitos:**
    *   Python 3.9+
    *   Uma chave de API do Google Cloud para a API Gemini e sua chave de acesso Ollama (ou Ollama Local).

2.  **Ambiente Virtual (Recomendado):**
    Crie e ative um ambiente virtual para isolar as dependências do projeto.
    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No Linux/macOS:
    source venv/bin/activate
    ```

3.  **Instalação de Dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo e instale-o:
    ```
    # requirements.txt
    annotated-types==0.7.0
    anyio==4.9.0
    cachetools==5.5.2
    certifi==2025.4.26
    charset-normalizer==3.4.2
    google-ai-generativelanguage
    google-api-core
    google-api-python-client
    google-auth
    google-auth-httplib2
    google-generativeai
    googleapis-common-protos
    grpcio
    grpcio-status
    h11==0.16.0
        # via httpcore
    httpcore==1.0.9
        # via httpx
    httplib2==0.22.0
        # via
        #   google-api-python-client
        #   google-auth-httplib2
    httpx==0.28.1
        # via vlm-benchmark (pyproject.toml)
    idna==3.10
        # via
        #   anyio
        #   httpx
        #   requests
    loguru==0.7.3
        # via vlm-benchmark (pyproject.toml)
    proto-plus==1.26.1
        # via
        #   google-ai-generativelanguage
        #   google-api-core
    protobuf==5.29.4
        # via
        #   google-ai-generativelanguage
        #   google-api-core
        #   google-generativeai
        #   googleapis-common-protos
        #   grpcio-status
        #   proto-plus
    pyasn1==0.6.1
        # via
        #   pyasn1-modules
        #   rsa
    pyasn1-modules==0.4.2
        # via google-auth
    pydantic==2.11.4
        # via google-generativeai
    pydantic-core==2.33.2
        # via pydantic
    pyparsing==3.2.3
        # via httplib2
    requests==2.32.3
        # via google-api-core
    rsa==4.9.1
        # via google-auth
    sniffio==1.3.1
        # via anyio
    tqdm==4.67.1
        # via google-generativeai
    typing-extensions==4.13.2
        # via
        #   anyio
        #   google-generativeai
        #   pydantic
        #   pydantic-core
        #   typing-inspection
    typing-inspection==0.4.1
        # via pydantic
    uritemplate==4.1.1
        # via google-api-python-client
    urllib3==2.4.0
        # via requests
    
    scikit-learn
    matplotlib
    seaborn
    sentence-transformers
    python-dotenv
    langchain
    langchain-google-genai
    pandas
    tqdm
    loguru
    numpy
    
    ```
    Execute o comando de instalação:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração da Chave de API:**
    *   Crie um arquivo chamado `.env` na raiz do projeto.
    *   Adicione sua chave da API do Google e Ollama, além dos modelos e temperaturas a serem testadas, a este arquivo:
        ```
        OLLAMA_URL= url_do_seu_servidor_ollama_ou_local
        OLLAMA_TOKEN= seu_token_ollama
        GOOGLE_API_KEY= sua_chave_API_gemini

        BOLLAMA_BENCHMARK_MODELS= "gemma3:4b,gemma3:12b,gemma3:27b,qwen2.5:7b,qwen2.5:14b,qwen2.5:32b,llava:7b,llava:13b" 
        BENCHMARK_TEMPERATURES= "0.0,0.5,1.0"
        GEMINI_BENCHMARK_MODELS="gemini-2.5-pro,gemini-2.5-flash"
        ```
    O script carregará esta chave automaticamente.

## Fluxo de Execução

O processo é dividido em três scripts principais, que devem ser executados em sequência.

### Etapa 1: Geração das Respostas do Modelo

Este passo executa o modelo VLM contra o dataset de imagens e gera um arquivo JSON com as análises.

**Scripts:** `benchmark_gemini_multimodels.py` ou `benchmark_ollama_multimodels.py`

**Configuração:**
*   **Modelos e Temperaturas:** Abra o `benchmark_gemini_multimodels.py` e ajuste as listas `MODEL_NAMES_TO_TEST` e `TEMPERATURES_TO_TEST` no topo do arquivo, ou defina as variáveis de ambiente (arquivo `.env`) `GEMINI_BENCHMARK_MODELS`, `OLLAMA_BENCHMARK_MODELS` e `BENCHMARK_TEMPERATURES`.
    *   **Exemplo de variável de ambiente (`.env`):**
        ```bash
        OLLAMA_URL= url_do_seu_servidor_ollama_ou_local
        OLLAMA_TOKEN= seu_token_ollama
        GOOGLE_API_KEY= sua_chave_API_gemini

        OLLAMA_BENCHMARK_MODELS= "gemma3:4b,gemma3:12b,gemma3:27b,qwen2.5:7b,qwen2.5:14b,qwen2.5:32b,llava:7b,llava:13b" 
        BENCHMARK_TEMPERATURES= "0.0,0.5,1.0"
        GEMINI_BENCHMARK_MODELS="gemini-2.5-pro,gemini-2.5-flash"
        ```
*   **Dataset e Prompts:** Verifique se os caminhos em `DATASET_PATH`, `IMAGES_BASE_DIR`, e `PROMPTS_DIR` estão corretos.

**Execução:**
1.  **Testes de Diagnóstico (Recomendado):** Antes da execução completa, rode os testes para verificar a configuração.

   *    Para o benchmark de modelos Gemini:
        ```bash
        python benchmark_gemini_multimodels.py -t
        ```
   *    Para o benchmark de modelos no Ollama:
        ```bash
        python benchmark_ollama_multimodels.py -t
        ```
        Isso testará a conexão, a localização de uma propriedade e o processamento de um único item.
    
2.  **Execução Completa:**

   *    Para o benchmark de modelos Gemini:
        ```bash
        python benchmark_gemini_multimodels.py
        ```
   *    Para o benchmark de modelos no Ollama:
        ```bash
        python benchmark_ollama_multimodels.py
        ```
    
        Isso irá iterar sobre todas as combinações de modelo/temperatura (presentes no arquivo '.env') e sobre todas as propriedades no dataset. Para cada combinação, será criada uma subpasta em `resultados_geracao/` contendo um arquivo `generation_results.json`.

---

### Etapa 2: Avaliação e Cálculo de Métricas

Este passo lê os arquivos `generation_results.json` gerados na Etapa 1 e os compara com o ground truth do seu CSV para calcular métricas detalhadas.

**Script:** `utils/evaluate_results.py`

**Configuração:**
*   **Nomes das Colunas Ground Truth:** Abra `utils/evaluate_results.py` e verifique se os nomes das colunas em `CAMPOS_CATEGORICOS_MAPEAMENTO` (na chave `gt_col`) correspondem exatamente aos nomes das colunas no seu arquivo CSV de ground truth (ex: `estrutura`, `piso`, `benfeitorias`).

**Execução:**
O script é executado a partir da **raiz do projeto** e recebe como argumento principal o caminho para o diretório de resultados gerados na etapa anterior.

```bash
# Exemplo para avaliar UMA execução específica
python utils/evaluate_results.py "resultados_geracao/output_gemini-2p5-pro_temp_1p0_20231028_120000/generation_results.json"
```
Para cada execução avaliada, o script criará uma subpasta correspondente em `resultados_avaliacao/`. Esta pasta conterá:
*   `evaluation_metrics_MODEL_temp_TEMPERATURE.json`: Um JSON com todas as métricas detalhadas por categoria e por classe.
*   `summary_classification_metrics_MODEL_temp_TEMPERATURE.csv`: Um CSV com as métricas ponderadas por categoria.
*   Uma subpasta `metric_plots_MODEL_temp_TEMPERATURE/` com os gráficos das matrizes de confusão e o gráfico de barras das métricas por categoria.

---

### Etapa 3: Criação de Relatórios Sumarizados (CSV e LaTeX)

Este passo final agrega os resultados de múltiplas avaliações (de diferentes modelos/temperaturas) em relatórios consolidados, ideais para comparação.

**Scripts:** `utils/create_summary_report.py`, `utils/latex_model_categ_table.py` e `utils/latex_model_temp_tables.py`.

**Configuração:**
*   **`create_summary_report.py`:**
    *   Este script lê todos os arquivos `evaluation_metrics_*.json` de dentro de `resultados_avaliacao`.
    *   Gera dois CSVs comparativos.
*   **`latex_model_categ_table.py`, `latex_model_temp_tables.py`:**
    *   Estes scripts leem os CSVs gerados pelo `create_summary_report.py` para criar tabelas formatadas em código LaTeX.

**Execução:**
1.  **Gerar os CSVs Sumarizados:**
    Execute o `create_summary_report.py` a partir da **raiz do projeto**.
    ```bash
    python utils/create_summary_report.py
    ```
    Isso criará a pasta `resultados_sumarizados` com:
    *   `report_metrics_by_category_TIMESTAMP.csv`: Todas as métricas por categoria, de todos os modelos/temperaturas, em um único arquivo.
    *   `report_overall_model_averages_TIMESTAMP.csv`: Apenas as médias gerais de cada modelo/temperatura.

2.  **Gerar as Tabelas LaTeX:**
    Execute os `latex_model_categ_table.py`, `latex_model_temp_tables.py` a partir da **raiz do projeto**, apontando para um dos CSVs sumarizados gerados no passo anterior.
    ```bash
    # Exemplo para criar a tabela de métricas médias gerais com temperaturas (sem tempo de execução '--no-exec-time')
    python utils/latex_model_temp_tables.py "resultados_sumarizados/report_overall_model_averages_TIMESTAMP.csv" --no-top-rule --no-exec-time
    
    # Exemplo para criar a tabela de temperaturas com tempo de latência total do modelo em segundos (sem métricas médias gerais '--no-metrics')
    python utils/latex_model_temp_tables.py "resultados_sumarizados/report_overall_model_averages_TIMESTAMP.csv" --no-top-rule --no-metrics

    # Exemplo para criar a tabela de temperaturas com tempo de latência total do modelo em segundos e métricas médias gerais
    python utils/latex_model_temp_tables.py "resultados_sumarizados/report_overall_model_averages_TIMESTAMP.csv" --no-top-rule

    # Exemplo para criar a tabela detalhada por categoria (sem temperatura '--no-temp')
    python utils/latex_model_categ_table.py "resultados_sumarizados/report_metrics_by_category_TIMESTAMP.csv" --no-top-rule --no-temp
    ```
    Isso gerará os arquivos `.tex` que você pode incluir diretamente em seus documentos LaTeX.

---

Este fluxo de trabalho estruturado permite uma avaliação robusta e reproduzível do desempenho de diferentes modelos VLM em sua tarefa específica, desde a geração de dados brutos até a criação de relatórios finais prontos para publicação.
