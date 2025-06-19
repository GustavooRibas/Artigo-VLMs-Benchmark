'''
Script para a conexão no Servidor/API Ollama
'''

import os
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Set
import json # Importado para decodificação JSON
import time # Para sleep
from pathlib import Path # Para _encode_image_if_path
from dotenv import load_dotenv

import httpx
from loguru import logger

def _truncate_payload_for_logging(payload: Dict[str, Any], max_len: int = 500) -> str:

    truncated_payload = {}
    for key, value in payload.items():
        if key == "images" and isinstance(value, list):
            truncated_payload[key] = [
                f"{img_b64[:30]}...[len:{len(img_b64)}]" if isinstance(img_b64, str) else img_b64
                for img_b64 in value
            ]
        elif isinstance(value, str) and len(value) > max_len:
            truncated_payload[key] = f"{value[:max_len]}...[len:{len(value)}]"
        else:
            truncated_payload[key] = value
    return str(truncated_payload)

load_dotenv()

# --- CONFIGURAÇÕES PARA O SERVIDOR OLLAMA EXTERNO ---
BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOKEN = os.getenv("OLLAMA_TOKEN", "TOKEN_DEFAULT")
DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "gemma3:4b")

# Endpoints específicos do servidor, caso necessário
API_TAGS_ENDPOINT = os.getenv("OLLAMA_API_TAGS_ENDPOINT", "/ollama/api/tags")
API_GENERATE_ENDPOINT = os.getenv("OLLAMA_API_GENERATE_ENDPOINT", "/ollama/api/generate")

logger.add("logs/ollama_connector.log", rotation="1 MB", retention="7 days", level="DEBUG", encoding="utf-8")


class ModelCache:

    def __init__(self, expiration_minutes: int = 60):
        self._models: Set[str] = set()
        self._last_update: Optional[datetime] = None
        self._expiration_minutes = expiration_minutes
        self._is_fetching = False
    def is_valid(self) -> bool:
        if self._is_fetching: return False
        if not self._last_update: return False
        return datetime.now() - self._last_update < timedelta(minutes=self._expiration_minutes)
    def update(self, models: Set[str]):
        self._models = models
        self._last_update = datetime.now()
    def get_models(self) -> Set[str]:
        return self._models.copy()

_model_cache = ModelCache()


def _get_full_url(endpoint: str, base_url_override: Optional[str] = None) -> str:
    """Constrói a URL completa, cuidando de barras duplas."""
    current_base = base_url_override or BASE_URL
    # Remove barra final de current_base e barra inicial de endpoint para evitar duplicação
    return current_base.rstrip('/') + '/' + endpoint.lstrip('/')


def fetch_available_models(
    base_url: Optional[str] = None, # Permite sobrescrever BASE_URL
    token: Optional[str] = None,    # Permite sobrescrever TOKEN
    force_refresh: bool = False,
    timeout: int = 30 # Timeout para buscar modelos do servidor externo
) -> Set[str]:
    if not force_refresh and _model_cache.is_valid():
        logger.debug("Usando lista de modelos do cache.")
        return _model_cache.get_models()

    _model_cache._is_fetching = True

    url = _get_full_url(API_TAGS_ENDPOINT, base_url)
    current_token = token if token is not None else TOKEN # Prioriza token passado como argumento

    headers = {"Content-Type": "application/json"}
    if current_token: # Adicionar Authorization apenas se o token existir e não for vazio
        headers["Authorization"] = f"Bearer {current_token}"
    else:
        logger.warning(f"Nenhum token fornecido para {url}. A requisição pode falhar se for necessária autenticação.")

    logger.info(f"FETCH_MODELS: Usando URL: {url}, Token Presente: {bool(current_token)}")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
            logger.debug(f"Resposta de {url}: Status {response.status_code}, Conteúdo (primeiros 200 chars): {response.text[:200]}")
            response.raise_for_status()
            data = response.json()

            models_data = data.get("models", [])
            if not isinstance(models_data, list): # Adicionar verificação de tipo
                 logger.error(f"Campo 'models' na resposta de {url} não é uma lista: {type(models_data)}. Resposta: {data}")
                 models = set()
            else:
                models = {model_info["name"] for model_info in models_data if isinstance(model_info, dict) and "name" in model_info}

            logger.debug(f"Encontrados {len(models)} modelos disponíveis em {BASE_URL}: {sorted(list(models))}")
            _model_cache.update(models)
            return models
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP {e.response.status_code} ao buscar modelos de {url}: {e.response.text}")
        # Se for 401 Unauthorized ou 403 Forbidden, o token pode estar errado ou ausente.
        if e.response.status_code in [401, 403]:
            logger.error(f"Erro de autenticação/autorização ({e.response.status_code}). Verifique o TOKEN.")
    except json.JSONDecodeError as e_json:
        logger.error(f"Erro ao decodificar JSON da resposta de {url}. Resposta não era JSON válido. Erro: {e_json}. Texto da Resposta: {response.text[:500] if 'response' in locals() else 'N/A'}")
    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar modelos disponíveis de {url}: {e}")
    finally:
        _model_cache._is_fetching = False

    if _model_cache.get_models():
        logger.warning(f"Usando cache de modelos (possivelmente expirado) devido a erro na busca de {url}.")
        return _model_cache.get_models()
    # Se falhar e não houver cache, levanta um erro para que o chamador saiba.
    raise RuntimeError(f"Não foi possível buscar modelos de {url} e o cache está vazio.")


def validate_model(
    model_name: str,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 30
) -> bool:
    try:
        available_models = fetch_available_models(base_url, token, timeout=timeout)
        is_valid = model_name in available_models
        if not is_valid:
            logger.warning(f"Modelo '{model_name}' não encontrado nos modelos de {base_url or BASE_URL}: {sorted(list(available_models))}")
        return is_valid
    except RuntimeError:
        logger.error(f"Não foi possível validar o modelo {model_name} (lista de modelos indisponível).")
        return False
    except Exception as e:
        logger.error(f"Erro inesperado ao validar o modelo {model_name}: {e}")
        return False

# _encode_image_if_path deve estar aqui ou importada
def _encode_image_if_path(image_data: Union[str, bytes]) -> str:

    if isinstance(image_data, str):
        try:
            path_obj = Path(image_data)
            if not path_obj.is_file():
                path_obj_abs = Path(os.path.abspath(image_data))
                if not path_obj_abs.is_file():
                    raise FileNotFoundError(f"Img não encontrada: '{image_data}' ou '{path_obj_abs}'")
                path_obj = path_obj_abs
            with open(path_obj, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError: logger.error(f"Img não encontrada: {image_data}"); raise
        except Exception as e: logger.error(f"Erro ao ler/codificar img {image_data}: {e}"); raise
    elif isinstance(image_data, bytes): return base64.b64encode(image_data).decode("utf-8")
    else: raise ValueError(f"Tipo de img não sup: {type(image_data)}")


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    template: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    images: Optional[List[Union[str, bytes]]] = None,
    validate_model_name: bool = True,
    timeout: int = 300 # Timeout maior para servidor externo e múltiplas imagens
) -> Dict[str, Any]:
    model_name = model or DEFAULT_MODEL
    
    if validate_model_name:
        # Passa explicitamente o token para validação
        if not validate_model(model_name, base_url, token if token is not None else TOKEN):
            error_msg = f"Modelo '{model_name}' não está disponível em {base_url or BASE_URL} ou não pôde ser validado."
            logger.error(error_msg)
            return {"error": error_msg, "model": model_name}

    url = _get_full_url(API_GENERATE_ENDPOINT, base_url)
    current_token = token if token is not None else TOKEN

    headers = {"Content-Type": "application/json"}
    if current_token:
        headers["Authorization"] = f"Bearer {current_token}"
    else:
        logger.warning(f"Nenhum token fornecido para POST {url}. Pode falhar se autenticação for necessária.")

    payload: Dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": stream}
    if system: payload["system"] = system
    if template: payload["template"] = template
    if options: payload["options"] = options
        
    if images:
        encoded_image_list = []
        for img_data in images:
            try:
                if isinstance(img_data, str) and not (img_data.startswith('data:image') or len(img_data) > 200 and any(img_data.startswith(p) for p in ('UklGR','R0lGO','iVBOR','/9j/'))):
                    encoded_image_list.append(_encode_image_if_path(img_data))
                elif isinstance(img_data, bytes):
                     encoded_image_list.append(_encode_image_if_path(img_data))
                elif isinstance(img_data, str):
                    encoded_image_list.append(img_data)
                else: raise ValueError(f"Tipo de imagem não suportado: {type(img_data)}")
            except Exception as e:
                logger.error(f"Erro ao processar imagem para payload: {e}")
                return {"error": f"Erro ao processar imagem: {e}", "model": model_name, "prompt": prompt}
        if encoded_image_list: payload["images"] = encoded_image_list

    log_payload_summary = {k: (f"<{type(v).__name__} len={len(v)}>" if isinstance(v, (str,list)) else v) for k,v in payload.items() if k != 'images'}
    if 'images' in payload: log_payload_summary['num_images'] = len(payload['images'])
    logger.info(f"OLLAMA_GENERATE: Usando URL: {url}, Token Presente: {bool(current_token)}. Payload (sumário): {log_payload_summary}")

    try:

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            logger.debug(f"Resposta de POST {url}: Status {response.status_code}, Conteúdo (primeiros 200 chars): {response.text[:200]}")
            response.raise_for_status()
            result = response.json()
            logger.success(f"Resposta recebida com sucesso de {url} para o modelo {model_name}.")
            return result
        
    except httpx.ReadTimeout: # Blocos de erro com logging da URL e token status
        error_msg = f"Timeout ({timeout}s) durante POST para {url} com modelo {model_name}."
        logger.error(error_msg); return {"error": error_msg, "model": model_name}
    
    except httpx.HTTPStatusError as e:
        error_msg = f"Erro HTTP {e.response.status_code} da API ({url}): {e.response.text[:500]}"
        logger.error(error_msg)
        if e.response.status_code in [401, 403]: logger.error(f"Erro de autenticação/autorização ({e.response.status_code}). Verifique o TOKEN.")
        # O erro 413 (Request Entity Too Large) e 405 (Method Not Allowed) retornados.
        return {"error": error_msg, "model": model_name, "response_text": e.response.text}
    
    except json.JSONDecodeError as e_json:
        error_msg = f"Erro ao decodificar JSON da resposta de POST {url}. Erro: {e_json}. Texto: {response.text[:500] if 'response' in locals() else 'N/A'}"
        logger.error(error_msg); return {"error": error_msg, "model": model_name, "response_text": response.text if 'response' in locals() else 'N/A'}
    
    except Exception as e:
        error_msg = f"Erro inesperado durante POST para {url}: {e}"
        logger.exception(error_msg); return {"error": error_msg, "model": model_name}

# Função main
if __name__ == "__main__":
    logger.info("Executando testes do ollama_connector...")

    # Teste 1: Buscar modelos disponíveis
    print("\n--- Teste 1: Buscar Modelos Disponíveis ---")
    try:
        models = fetch_available_models(force_refresh=True)
        if models:
            print(f"Modelos disponíveis: {sorted(list(models))}")
            # Selecionar um modelo para os próximos testes (idealmente um VLM)
            test_model_name = DEFAULT_MODEL
            if "llava" not in test_model_name.lower() and any("llava" in m.lower() for m in models):
                test_model_name = next(m for m in models if "llava" in m.lower())
            elif not models:
                test_model_name = DEFAULT_MODEL # Fallback se nenhum llava for encontrado
            print(f"Usando modelo para testes: {test_model_name}")
        else:
            print("Nenhum modelo encontrado ou falha ao buscar.")
            test_model_name = DEFAULT_MODEL # Fallback
    except Exception as e:
        print(f"Erro ao buscar modelos: {e}")
        test_model_name = DEFAULT_MODEL # Fallback

    # Teste 2: Validar um modelo
    print(f"\n--- Teste 2: Validar Modelo ({test_model_name}) ---")
    is_valid = validate_model(test_model_name)
    print(f"O modelo '{test_model_name}' é válido? {is_valid}")

    # Teste 3: Gerar texto sem imagem
    print("\n--- Teste 3: Gerar Texto (sem imagem) ---")
    if is_valid or not validate_model_name: # Tentar mesmo se a validação falhar, mas validate_model_name=False
        response_text_only = ollama_generate(
            prompt="Why is the sky blue?",
            model=test_model_name,
            options={"temperature": 0.7},
            validate_model_name=False # Já validamos ou queremos testar mesmo assim
        )
        print("Resposta (texto apenas):")
        if "error" in response_text_only:
            print(f"  Erro: {response_text_only['error']}")
        else:
            print(f"  Modelo: {response_text_only.get('model')}")
            print(f"  Resposta: {response_text_only.get('response', '').strip()}")
            print(f"  Duração Total (ns): {response_text_only.get('total_duration')}")
    else:
        print(f"Pulando teste de geração de texto pois o modelo '{test_model_name}' não é válido ou não foi possível validar.")

    # Teste 4: Gerar texto com uma imagem (requer uma imagem de exemplo)
    print("\n--- Teste 4: Gerar Texto com Imagem ---")
    example_image_path = "data/example.jpg" # CRIE ESTE ARQUIVO DE IMAGEM PARA O TESTE
    # Crie uma pasta 'data' e coloque uma imagem 'example.jpg' nela, ou ajuste o caminho.
    if not os.path.exists(example_image_path):
        # Tentar criar uma imagem dummy se não existir para o teste rodar
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 100), color = 'red')
            d = ImageDraw.Draw(img)
            d.text((10,10), "Test IMG", fill=(255,255,0))
            os.makedirs(os.path.dirname(example_image_path), exist_ok=True)
            img.save(example_image_path)
            print(f"Criada imagem dummy em: {example_image_path}")
        except ImportError:
            print(f"PIL/Pillow não instalado. Não foi possível criar imagem dummy.")
        except Exception as e:
            print(f"Erro ao criar imagem dummy: {e}")


    if os.path.exists(example_image_path) and (is_valid  or not validate_model_name):
        response_with_image = ollama_generate(
            prompt="What do you see in this image?",
            model=test_model_name, # Use um modelo VLM como llava
            images=[example_image_path],
            validate_model_name=False
        )
        print(f"Resposta (com imagem '{example_image_path}'):")
        if "error" in response_with_image:
            print(f"  Erro: {response_with_image['error']}")
        else:
            print(f"  Modelo: {response_with_image.get('model')}")
            print(f"  Resposta: {response_with_image.get('response', '').strip()}")
            print(f"  Duração Total (ns): {response_with_image.get('total_duration')}")
    else:
        if not os.path.exists(example_image_path):
            print(f"Pulando teste com imagem: arquivo de exemplo '{example_image_path}' não encontrado.")
        else:
            print(f"Pulando teste com imagem pois o modelo '{test_model_name}' não é válido ou não foi possível validar.")
            
    logger.info("Testes do ollama_connector concluídos.")