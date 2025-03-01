"""
Utilitários para processamento e manipulação de imagens.
"""
import cv2
import numpy as np
from datetime import datetime
from face_detector.config.settings import (
    COR_VERDE, COR_VERMELHO, COR_AZUL, COR_AMARELO,
    QUALIDADE_JPEG, APLICAR_MELHORIA_IMAGEM, USAR_TONS_CINZA,
    PESSOA_INFO
)

def melhorar_imagem(imagem):
    """Aplica técnicas de processamento para melhorar a qualidade da imagem"""
    if not APLICAR_MELHORIA_IMAGEM:
        return imagem
    
    # Converter para escala de cinza para processamento
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar equalização de histograma adaptativa (CLAHE) para melhorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Se estiver usando tons de cinza para reconhecimento, retornar a imagem em tons de cinza
    if USAR_TONS_CINZA:
        # Converter de volta para BGR (3 canais) para compatibilidade com o resto do código
        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Converter de volta para BGR
    imagem_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Misturar com a imagem original para manter cores (70% original, 30% equalizada)
    imagem_melhorada = cv2.addWeighted(imagem, 0.7, imagem_eq, 0.3, 0)
    
    # Aplicar sharpening para melhorar detalhes
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    imagem_melhorada = cv2.filter2D(imagem_melhorada, -1, kernel_sharpen)
    
    # Reduzir ruído mantendo bordas
    imagem_melhorada = cv2.bilateralFilter(imagem_melhorada, 9, 75, 75)
    
    return imagem_melhorada

def adicionar_info_tela(frame):
    """Adiciona informações na tela"""
    altura, largura = frame.shape[:2]
    
    # Adicionar timestamp
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, timestamp, (10, altura - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 1)
    
    # Adicionar informações da pessoa de referência
    cv2.putText(frame, f"Ref: {PESSOA_INFO['nome']}", (largura - 300, altura - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AZUL, 1)
    
    # Adicionar instruções
    cv2.putText(frame, "ESC: Sair", (largura - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 1)

def salvar_imagem(imagem, caminho, qualidade=None):
    """Salva uma imagem com a qualidade especificada"""
    if qualidade is None:
        qualidade = QUALIDADE_JPEG
    
    cv2.imwrite(caminho, imagem, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    return caminho 