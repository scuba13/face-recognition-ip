"""
Configurações do sistema de detecção de faces.
Contém todas as constantes e parâmetros utilizados pelo sistema.
"""
import numpy as np

# URL RTSP fixa que sabemos que funciona
RTSP_URL = "rtsp://192.168.0.133:554/0/av0"

# Configurações do MongoDB
MONGODB_HOST = "localhost"  # Alterado de "mongodb" para "localhost"
MONGODB_PORT = 27017
MONGODB_USERNAME = "admin"
MONGODB_PASSWORD = "admin123"  # Senha definida no docker-compose
MONGODB_DATABASE = "face_recognition"
MONGODB_AUTH_SOURCE = "admin"

# Configurações de detecção de movimento
MOVIMENTO_THRESHOLD = 15000  # Limiar de detecção de movimento (quanto menor, mais sensível)
AREA_MINIMA_CONTORNO = 5000  # Área mínima de contorno para considerar como movimento real
FRAMES_APOS_MOVIMENTO = 5    # Número de frames para processar após detectar movimento
INTERVALO_ENTRE_FRAMES = 0.1  # Intervalo entre cada frame após movimento (segundos)
INTERVALO_MINIMO_MOVIMENTO = 0.5  # Intervalo mínimo entre detecções de movimento (segundos)

# Configurações de reconhecimento facial
FACE_SIMILARITY_THRESHOLD = 0.6  # Reduzido para ser mais permissivo
FACE_DETECTION_INTERVAL = 30     # Intervalo para detecção de faces (a cada quantos frames)
INTERVALO_MINIMO_FACE = 0.3      # Intervalo mínimo entre processamentos de face (segundos)
TEMPO_EXPIRACAO_FACE = 3.0       # Tempo para considerar uma face como "nova" novamente (segundos)
MODELO_FACE = "hog"  # Modelo mais rápido e suficiente para a maioria dos casos
NUM_JITTERS = 1  # Reduzido para melhorar performance

# Configurações de captura e processamento
BUFFER_SIZE_CAPTURA = 10         # Tamanho do buffer de frames para captura
TAXA_FPS_CAPTURA = 30            # Taxa de FPS alvo para captura
TAXA_FPS_UI = 30                 # Taxa de FPS alvo para interface gráfica

# Configurações de qualidade de imagem
RESOLUCAO_CAPTURA = (1920, 1080)  # HD para melhor desempenho
QUALIDADE_JPEG = 95               # Qualidade de salvamento (0-100)
APLICAR_MELHORIA_IMAGEM = True
USAR_TONS_CINZA = False  # Usar imagem colorida para detecção

# Configurações de debug
MODO_DEBUG = True
MAX_FRAMES_SEM_DETECCAO = 100

# Cores para visualização (BGR)
COR_VERDE = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AZUL = (255, 0, 0)
COR_AMARELO = (0, 255, 255)

