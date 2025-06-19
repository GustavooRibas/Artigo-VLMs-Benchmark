'''
Script com funções auxiliares
'''

import os
from pathlib import Path
from loguru import logger
from typing import Any, Dict, List # Adicionado List para get_images_from_directory
import base64
from PIL import Image
import io # Necessário para salvar a imagem convertida em memória

def _truncate_log_value(value: Any, max_length: int = 100) -> str:

    """
    Trunca valores longos para registro em log, mostrando o início e o fim de strings longas.

    Argumentos:
        value: Valor a ser truncado
        max_length: Comprimento máximo antes do truncamento

    Retorna:
        Representação em string do valor truncado
    """

    if isinstance(value, str):
        if len(value) <= max_length:
            return value
        half = (max_length - 3) // 2
        return f"{value[:half]}...{value[-half:]}"
    elif isinstance(value, (list, tuple, set)):
        # Limit a visualização de listas/tuplas/sets longos
        str_value = str(value)
        if len(str_value) <= max_length:
            return str_value
        # Tenta mostrar o início e o fim se for muito longo
        try:
            items = list(value) # Converter para lista para indexação
            if not items: return "[]" # ou "()" ou "{}" dependendo do tipo original
            if len(items) <= 2:
                 return str(value) # Se tiver 1 ou 2 itens, mostrar como está
            # Mostrar o primeiro e o último item, e a contagem dos do meio
            first_item_str = _truncate_log_value(items[0], max_length // 3)
            last_item_str = _truncate_log_value(items[-1], max_length // 3)
            return f"[{first_item_str}, ... {len(items)-2} items ..., {last_item_str}] (Total: {len(items)})"

        except TypeError: # Se não for iterável ou não suportar indexação assim
            return str_value[:max_length-3] + "..."

    elif isinstance(value, dict):
        str_value = str(value)
        if len(str_value) <= max_length:
            return str_value
        try:
            items = list(value.items())
            if not items: return "{}"
            if len(items) <= 2:
                return str(value)
            first_item_key_str = _truncate_log_value(items[0][0], max_length // 4)
            first_item_val_str = _truncate_log_value(items[0][1], max_length // 4)
            last_item_key_str = _truncate_log_value(items[-1][0], max_length // 4)
            last_item_val_str = _truncate_log_value(items[-1][1], max_length // 4)
            return f"{{{first_item_key_str}: {first_item_val_str}, ... {len(items)-2} items ..., {last_item_key_str}: {last_item_val_str}}} (Total: {len(items)})"
        except TypeError:
            return str_value[:max_length-3] + "..."
    return str(value)[:max_length] # Fallback para outros tipos

def _truncate_payload_for_logging(payload: Dict[str, Any], default_max_len: int = 100) -> Dict[str, Any]:

    """
    Cria uma cópia do payload com os valores truncados para registro em log.
    Tratamento especial para a lista 'images'.

    Argumentos:
        payload: Dicionário de payload original
        default_max_len: Comprimento máximo padrão para truncar valores de string.

    Retorna:
        Cópia do payload com valores truncados
    """

    log_payload = {}
    for key, value in payload.items():
        if key == "images" and isinstance(value, list):
            log_payload[key] = [
                _truncate_log_value(img, max_length=50) # Shorter truncation for base64 previews
                for img in value
            ]
        else:
            log_payload[key] = _truncate_log_value(value, max_length=default_max_len)
    return log_payload

# Função para converter imagens a JPEG OU PNG e, em seguida, para base64, formato utilizado em VLMs
def _encode_image_to_base64(image_path: str) -> str:

    """
    Converte uma imagem para um formato padrão (JPEG ou PNG) e, em seguida, para uma string em base64.
    """
    try:
        path = Path(image_path)
        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            logger.error(f"Arquivo de imagem nao encontrado no caminho original: {path}")
            raise FileNotFoundError(f"Imagem nao encontrada em {path}")

        logger.debug(f"Processando imagens para codificacao: {resolved_path}")
        
        # Abrir imagem com Pillow
        img = Image.open(resolved_path)

        # Converter para RGB para remover canal alfa e padronizar (a menos que queira manter PNG com alfa)
        # Se o modelo tiver problemas com transparência, converter para RGB é mais seguro.
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            logger.debug(f"Convertendo imagem {resolved_path} de {img.mode} para RGB.")
            img = img.convert('RGB')
        elif img.mode != 'RGB' and img.mode != 'L': # L é grayscale, geralmente ok
            logger.warning(f"Imagem {resolved_path} está no modo {img.mode}. Considere converter para RGB se os problemas persistirem.")


        img_byte_arr = io.BytesIO()
        target_format = 'JPEG' # Ou 'PNG'
        
        if target_format == 'JPEG':
            # Para JPEG, precisamos garantir que o modo seja compatível (ex: RGB, L)
            if img.mode not in ['RGB', 'L']:
                logger.warning(f"Imagem {resolved_path} modo {img.mode} nao pode ser diretamente salva em JPEG, convertendo para RGB.")
                img = img.convert('RGB')
            img.save(img_byte_arr, format=target_format, quality=85) # Ajuste a qualidade conforme necessário
            logger.debug(f"Imagem convertida e salva {resolved_path} como JPEG na memoria.")
        elif target_format == 'PNG':
            img.save(img_byte_arr, format=target_format, optimize=True)
            logger.debug(f"Imagem convertida e salva {resolved_path} como PNG na memoria.")
        else: # Caso queira suportar outros formatos de destino
            img.save(img_byte_arr, format=img.format or 'PNG') # Tenta salvar no formato original ou PNG como fallback
            logger.debug(f"Imagem salva {resolved_path} (formato: {img.format or 'PNG'}) na memoria.")


        img_bytes = img_byte_arr.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        logger.debug(f"Imagem codificada com sucesso: {resolved_path} (formato de destino: {target_format}, base64: {_truncate_log_value(img_b64, 50)})")
        return img_b64
    except FileNotFoundError:
        raise
    except Image.UnidentifiedImageError: # Pillow não conseguiu identificar/abrir a imagem
        logger.error(f"Pillow UnidentifiedImageError: Nao foi possivel abrir ou ler o arquivo de imagem {image_path}. Ele pode estar corrompido ou nao ser uma imagem valida.")
        raise # Re-levanta para que o benchmark possa registrar o erro para esta imagem/propriedade
    except Exception as e:
        logger.error(f"Erro processando/convertendo imagem {image_path}: {type(e).__name__} - {e}")
        raise

# Função para coletar as imagens do diretório dos imóveis
def get_images_from_directory(dir_path: str) -> List[str]:

    """
    Varre um diretório em busca de arquivos de imagem e retorna uma lista com seus caminhos absolutos.

    Argumentos:
        dir_path: O caminho para o diretório a ser analisado.

    Retorna:
        Uma lista de strings com os caminhos absolutos dos arquivos de imagem encontrados no diretório.
        Retorna uma lista vazia se o diretório não existir ou se nenhuma imagem for encontrada.

    Exceções:
        ValueError: Se dir_path não for um diretório válido.
    """

    path_obj = Path(dir_path)
    if not path_obj.is_dir():
        logger.error(f"O caminho fornecido nao e um diretorio ou nao existe: {dir_path}")
        raise ValueError(f"Pasta nao esta no diretorio: {dir_path}")

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []

    try:
        for item in path_obj.iterdir(): # itera sobre os itens diretos do diretório
            if item.is_file() and item.suffix.lower() in image_extensions:
                image_files.append(str(item.resolve())) # Adiciona o caminho absoluto resolvido
        
        if not image_files:
            logger.warning(f"Nenhum arquivo de imagem encontrado no diretorio: {dir_path}")
        else:
            logger.debug(f"Foram encontradas {len(image_files)} imagens em {dir_path}. As primeiras sao: {image_files[:3]}")
        
        return image_files
    except Exception as e:
        logger.error(f"Erro ao escanear o diretorio {dir_path} em busca de imagens: {e}")
        return [] # Retorna lista vazia em caso de erro de permissão, etc.