'''
Script para a conexão na Gemini API
'''

import os
import io
import re
import json
import base64
import time # Para registrar tempo de resposta
from typing import List, Dict, Any, Optional, Union, Tuple

from dotenv import load_dotenv
from loguru import logger # Usaremos loguru para consistência

# Langchain e Google specific imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    # ChatPromptTemplate e StrOutputParser são mais para chains, podemos usar a chamada direta ao LLM
except ImportError:
    logger.critical("Langchain Google GenAI não instalado. Execute: pip install langchain-google-genai")
    raise

# --- Configuração Inicial e Chave API ---
load_dotenv() # Carrega variáveis de ambiente do arquivo .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Usar logger para erros críticos e levantar exceção
    logger.critical("A chave da API do Google (GOOGLE_API_KEY) não está definida nas variáveis de ambiente ou no arquivo .env.")
    raise ValueError("GOOGLE_API_KEY não definida.")

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro" # Modelo Gemini default

# Configuração do logger para este módulo
logger.add("logs/gemini_connector.log", rotation="1 MB", retention="7 days", level="DEBUG", encoding="utf-8")

# --- Funções Auxiliares Internas ---
def _encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """Codifica bytes de uma imagem em uma string base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def _read_image_bytes_from_path(image_path: str) -> Optional[bytes]:
    """Lê uma imagem de um caminho e retorna seus bytes."""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        logger.error(f"Arquivo de imagem não encontrado: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Erro ao ler arquivo de imagem {image_path}: {e}")
        return None

def _truncate_for_logging(data: Any, max_len: int = 100) -> str:
    """Trunca dados para logging, útil para strings longas como base64."""
    s = str(data)
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s

# --- Funções Principais do Conector ---

def list_available_gemini_models(llm_client: Optional[ChatGoogleGenerativeAI] = None) -> List[str]:
    """
    Lista modelos Gemini conhecidos que são tipicamente usados.
    A API Gemini não tem um endpoint de "listagem" como Ollama /api/tags.
    Esta função pode ser expandida se houver uma forma programática de listar via SDK.
    Por enquanto, retorna uma lista estática de modelos comuns.
    """
    # TODO: Verificar se o SDK do Google oferece uma forma de listar modelos dinamicamente.
    # Atualmente, a listagem de modelos é mais feita pela documentação ou Google Cloud Console.
    known_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite"
    ]
    logger.info(f"Retornando lista estática de modelos Gemini conhecidos: {known_models}")
    return known_models

def validate_gemini_model(model_name: str, llm_client: Optional[ChatGoogleGenerativeAI] = None) -> bool:
    """
    Valida se um nome de modelo Gemini é conhecido ou parece válido.
    Como não há endpoint de listagem fácil, esta validação é mais uma verificação heurística.
    Poderia tentar uma chamada muito pequena à API para ver se o modelo é aceito.
    """
    
    if "gemini" in model_name.lower(): # Heurística simples
        logger.debug(f"Nome do modelo '{model_name}' parece ser um modelo Gemini válido (verificação heurística).")
        return True
    logger.warning(f"Nome do modelo '{model_name}' não parece ser um modelo Gemini conhecido.")
    return False


def gemini_generate(
    prompt_parts: List[Union[str, Dict[str, Any]]], # Lista de partes do prompt (texto, imagem)
    system_prompt: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.1, # Gemini geralmente se beneficia de temperaturas baixas para tarefas factuais/JSON
    max_output_tokens: Optional[int] = None, # Ex: 2048 ou 8192
    # Outras opções do Gemini podem ser adicionadas aqui como kwargs
    **kwargs
) -> Dict[str, Any]:
    """
    Envia uma requisição de geração para a API Gemini (Vertex AI ou Google AI Studio).

    Args:
        prompt_parts: Uma lista contendo as partes do prompt.
                      Para texto: uma string.
                      Para imagens: um dicionário no formato Langchain para Gemini:
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"}}
                      Ou, alternativamente, bytes de imagem ou caminhos de arquivo que serão processados.
        system_prompt: O prompt do sistema (instruções gerais).
        model_name: Nome do modelo Gemini a ser usado (ex: "gemini-1.5-flash-latest").
        temperature: Temperatura para a geração.
        max_output_tokens: Número máximo de tokens a serem gerados.
        **kwargs: Argumentos adicionais para passar para ChatGoogleGenerativeAI.

    Returns:
        Um dicionário contendo:
        - "response": A string de resposta do modelo, se bem-sucedido.
        - "error": Uma mensagem de erro, se ocorreu um problema.
        - "model_name": O nome do modelo usado.
        - "usage_metadata": Metadados de uso (tokens) retornados pela API, se disponíveis.
        - "response_time_s": Tempo de resposta da chamada à API em segundos.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs
    )

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
        logger.debug(f"System Prompt (primeiros 100 chars): {_truncate_for_logging(system_prompt)}")


    # Construir o HumanMessage com todas as partes (texto e imagens)
    human_content = []
    for part in prompt_parts:
        if isinstance(part, str): # Parte de texto
            human_content.append({"type": "text", "text": part})
            logger.debug(f"Human Prompt Text Part (primeiros 100 chars): {_truncate_for_logging(part)}")
        elif isinstance(part, bytes): # Bytes de imagem
            base64_image = _encode_image_bytes_to_base64(part)
            # Assumir JPEG por padrão, ou detectar/passar o tipo de imagem
            human_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            logger.debug(f"Human Prompt Image Part (bytes): base64 (primeiros 30): {base64_image[:30]}...")
        elif isinstance(part, dict) and part.get("type") == "image_path": # Caminho da imagem
            image_bytes = _read_image_bytes_from_path(part["image_path"])
            if image_bytes:
                base64_image = _encode_image_bytes_to_base64(image_bytes)
                # Tentar inferir o tipo de imagem a partir da extensão do arquivo
                mime_type = "image/jpeg" # Default
                ext = os.path.splitext(part["image_path"])[1].lower()
                if ext == ".png": mime_type = "image/png"
                elif ext == ".gif": mime_type = "image/gif"
                elif ext == ".webp": mime_type = "image/webp"
                # Adicione outros tipos se necessário

                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
                logger.debug(f"Human Prompt Image Part (path: {part['image_path']}): base64 (primeiros 30): {base64_image[:30]}...")
            else:
                # Falha ao ler a imagem do caminho, registrar e pular esta imagem
                logger.error(f"Não foi possível ler a imagem do caminho: {part['image_path']}")
        elif isinstance(part, dict) and part.get("type") == "image_url": # Já formatado
            human_content.append(part)
            logger.debug(f"Human Prompt Image Part (url): {_truncate_for_logging(part['image_url']['url'])}")
        else:
            logger.warning(f"Tipo de parte de prompt desconhecido ou malformado: {part}")

    if not human_content:
        logger.error("Nenhum conteúdo válido (texto ou imagem) fornecido para o HumanMessage.")
        return {"error": "Conteúdo do prompt humano vazio.", "model_name": model_name}
        
    messages.append(HumanMessage(content=human_content))

    start_time = time.time()
    try:
        logger.info(f"Enviando requisição para Gemini model: {model_name}")
        ai_message: AIMessage = llm.invoke(messages) # Chamada direta ao LLM
        response_content = ai_message.content
        
        # A resposta do Gemini via Langchain já é uma string.
        # Metadados de uso podem estar em ai_message.response_metadata
        usage_metadata = ai_message.response_metadata.get("usage_metadata") or \
                         ai_message.response_metadata.get("token_usage") # Varia conforme a versão/API

        response_time_s = time.time() - start_time
        logger.success(f"Resposta recebida de Gemini ({model_name}) em {response_time_s:.2f}s.")
        logger.debug(f"Resposta crua do Gemini: {_truncate_for_logging(response_content, 500)}")

        return {
            "response": response_content,
            "model_name": model_name,
            "usage_metadata": usage_metadata, # Ex: {"prompt_token_count": X, "candidates_token_count": Y}
            "response_time_s": response_time_s,
            "error": None
        }
    except Exception as e:
        response_time_s = time.time() - start_time
        logger.exception(f"Erro durante a chamada à API Gemini ({model_name}) após {response_time_s:.2f}s: {e}")
        return {
            "error": str(e),
            "model_name": model_name,
            "response_time_s": response_time_s,
            "response": None,
            "usage_metadata": None
        }

# --- Bloco de Teste (if __name__ == "__main__") ---
if __name__ == "__main__":
    logger.info("Executando testes do gemini_connector...")

    # Teste 1: Listar modelos (informativo)
    print("\n--- Teste 1: Modelos Gemini Conhecidos ---")
    print(list_available_gemini_models())

    # Teste 2: Validar um modelo
    test_model = "gemini-2.5-pro" # Ou outro modelo que você tenha acesso
    print(f"\n--- Teste 2: Validar Modelo ({test_model}) ---")
    is_valid = validate_gemini_model(test_model)
    print(f"O modelo '{test_model}' passou na validação heurística? {is_valid}")

    # Teste 3: Gerar texto simples
    print("\n--- Teste 3: Geração de Texto Simples ---")
    simple_prompt_parts = ["Explique o que é um Modelo de Linguagem de Visão (VLM) em uma frase."]
    result_text = gemini_generate(prompt_parts=simple_prompt_parts, model_name=test_model, temperature=0.7)
    if result_text.get("error"):
        print(f"  Erro: {result_text['error']}")
    else:
        print(f"  Resposta: {result_text.get('response')}")
        print(f"  Metadados de Uso: {result_text.get('usage_metadata')}")
        print(f"  Tempo de Resposta: {result_text.get('response_time_s'):.2f}s")

    # Teste 4: Gerar texto com uma imagem
    print("\n--- Teste 4: Geração com Imagem ---")
    example_image_path = "data/example.jpg" # CRIE ESTA PASTA E IMAGEM, ou ajuste o caminho
    
    # Criar imagem dummy se não existir para o teste
    if not os.path.exists(example_image_path):
        try:
            from PIL import Image, ImageDraw
            os.makedirs(os.path.dirname(example_image_path), exist_ok=True)
            img = Image.new('RGB', (100, 30), color = 'red')
            d = ImageDraw.Draw(img)
            d.text((5,5), "TEST", fill=(255,255,0))
            img.save(example_image_path)
            logger.info(f"Criada imagem dummy em: {example_image_path}")
        except ImportError:
            logger.warning("Pillow não instalado. Não foi possível criar imagem dummy. Pule o teste com imagem se 'data/example.jpg' não existir.")
        except Exception as e_img:
            logger.error(f"Erro ao criar imagem dummy: {e_img}")

    if os.path.exists(example_image_path):
        # Ler os bytes da imagem para o teste
        with open(example_image_path, "rb") as f:
            image_bytes_for_test = f.read()

        multimodal_prompt_parts = [
            "Descreva esta imagem em poucas palavras.", # Parte de texto
            image_bytes_for_test # Parte de imagem (bytes)
            # Ou: {"type": "image_path", "image_path": example_image_path}
        ]
        result_multimodal = gemini_generate(
            prompt_parts=multimodal_prompt_parts,
            model_name=test_model, # Certifique-se que é um modelo multimodal
            temperature=0.5
        )
        if result_multimodal.get("error"):
            print(f"  Erro: {result_multimodal['error']}")
        else:
            print(f"  Resposta: {result_multimodal.get('response')}")
            print(f"  Metadados de Uso: {result_multimodal.get('usage_metadata')}")
            print(f"  Tempo de Resposta: {result_multimodal.get('response_time_s'):.2f}s")
    else:
        print(f"  Pulando teste com imagem: '{example_image_path}' não encontrado.")
    
    logger.info("Testes do gemini_connector concluídos.")