"""
Módulo de logging para o sistema de detecção facial.
Fornece funções para registrar diferentes tipos de eventos.
"""
from datetime import datetime

def log_info(mensagem):
    """Exibe log de informação com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO] [{timestamp}] {mensagem}")

def log_movimento(mensagem):
    """Exibe log de detecção de movimento com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[MOVIMENTO] [{timestamp}] {mensagem}")

def log_face(mensagem):
    """Exibe log de detecção de face com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[FACE] [{timestamp}] {mensagem}")

def log_captura(mensagem):
    """Exibe log específico para capturas com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[CAPTURA] [{timestamp}] {mensagem}")

def log_debug(mensagem, modo_debug=True):
    """Exibe log de depuração apenas se o modo debug estiver ativado"""
    if modo_debug:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[DEBUG] [{timestamp}] {mensagem}")

def log_error(mensagem):
    """Exibe log de erro com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[ERRO] [{timestamp}] {mensagem}")

def log_processamento(mensagem):
    """Exibe log de processamento com timestamp"""
    # Função modificada para não exibir nada
    pass 