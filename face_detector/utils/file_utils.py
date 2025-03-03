"""
Utilitários para manipulação de arquivos e diretórios.
"""
import os
import pickle
import numpy as np
from face_detector.utils.logger import log_info

def criar_estrutura_pastas():
    """Cria a estrutura de pastas para organizar as imagens"""
    # Pasta principal para capturas temporárias
    os.makedirs("capturas", exist_ok=True)
    log_info("Estrutura de pastas criada com sucesso")

def salvar_encoding(encoding, nome_arquivo, diretorio="encodings"):
    """Salva um encoding facial em um arquivo pickle"""
    os.makedirs(diretorio, exist_ok=True)
    
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    with open(caminho_completo, "wb") as f:
        pickle.dump(encoding, f)
    
    log_info(f"Encoding salvo em '{caminho_completo}'")
    return caminho_completo

def carregar_encoding(nome_arquivo, diretorio="encodings"):
    """Carrega um encoding facial de um arquivo pickle"""
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    try:
        with open(caminho_completo, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log_info(f"Arquivo de encoding '{caminho_completo}' não encontrado.")
        return None

def salvar_encoding_teste():
    """Salva o encoding real para comparação"""
    os.makedirs("encodings", exist_ok=True)
    
    # Salvar o encoding real
    with open("encodings/eduardo_nascimento.pickle", "wb") as f:
        pickle.dump(PESSOA_CONHECIDA_ENCODING, f)
    
    log_info(f"Encoding de {PESSOA_INFO['nome']} salvo em 'encodings/eduardo_nascimento.pickle'")

def carregar_encoding_teste():
    """Carrega o encoding real"""
    try:
        with open("encodings/eduardo_nascimento.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log_info(f"Arquivo de encoding 'encodings/eduardo_nascimento.pickle' não encontrado.")
        log_info(f"Usando encoding padrão para {PESSOA_INFO['nome']}...")
        salvar_encoding_teste()
        return PESSOA_CONHECIDA_ENCODING 