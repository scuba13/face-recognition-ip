"""
Utilitários para manipulação de arquivos e diretórios.
"""
import os
import pickle
import numpy as np
from face_detector.utils.logger import log_info
from face_detector.config.settings import PESSOA_CONHECIDA_ENCODING, PESSOA_INFO

def criar_estrutura_pastas():
    """Cria a estrutura de pastas para organizar as imagens"""
    # Pasta principal para capturas
    os.makedirs("capturas", exist_ok=True)
    
    # Subpastas para organizar por tipo
    os.makedirs("capturas/movimento", exist_ok=True)  # Frames com movimento
    os.makedirs("capturas/faces", exist_ok=True)      # Faces recortadas
    os.makedirs("capturas/faces/match", exist_ok=True)      # Faces reconhecidas
    os.makedirs("capturas/faces/desconhecido", exist_ok=True)  # Faces desconhecidas
    os.makedirs("capturas/frames", exist_ok=True)     # Frames completos com anotações
    os.makedirs("capturas/manual", exist_ok=True)     # Capturas manuais
    
    # Pasta para encodings
    os.makedirs("encodings", exist_ok=True)
    
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