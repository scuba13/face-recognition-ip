"""
Utilitários para processamento e manipulação de imagens.
"""
import cv2
import numpy as np
from datetime import datetime
from face_detector.config.settings import (
    COR_VERDE, COR_VERMELHO, COR_AMARELO,
    QUALIDADE_JPEG, APLICAR_MELHORIA_IMAGEM, USAR_TONS_CINZA
)
import os
from face_detector.utils.logger import log_error, log_captura

def melhorar_imagem(imagem):
    """Aplica técnicas de processamento para melhorar a qualidade da imagem"""
    if not APLICAR_MELHORIA_IMAGEM:
        return imagem
    
    # Converter para escala de cinza para processamento
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar equalização de histograma adaptativa (CLAHE) com parâmetros ajustados
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Se estiver usando tons de cinza para reconhecimento, retornar a imagem em tons de cinza
    if USAR_TONS_CINZA:
        # Converter de volta para BGR (3 canais) para compatibilidade com o resto do código
        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Converter de volta para BGR
    imagem_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Misturar com a imagem original para manter cores (60% original, 40% equalizada)
    imagem_melhorada = cv2.addWeighted(imagem, 0.6, imagem_eq, 0.4, 0)
    
    # Aplicar sharpening para melhorar detalhes
    kernel_sharpen = np.array([[-1, -1, -1],
                              [-1, 9.5, -1],
                              [-1, -1, -1]])
    imagem_melhorada = cv2.filter2D(imagem_melhorada, -1, kernel_sharpen)
    
    # Reduzir ruído mantendo bordas com parâmetros ajustados
    imagem_melhorada = cv2.bilateralFilter(imagem_melhorada, 9, 100, 100)
    
    # Normalizar o contraste global
    lab = cv2.cvtColor(imagem_melhorada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    imagem_melhorada = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return imagem_melhorada

def adicionar_info_tela(frame, fps=None, movimento_area=None):
    """Adiciona informações na tela"""
    altura, largura = frame.shape[:2]
    
    # Adicionar timestamp
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, timestamp, (10, altura - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 1)
    
    # Adicionar instruções
    cv2.putText(frame, "ESC: Sair", (largura - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 1)
    
    # Adicionar FPS se disponível
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
    
    # Adicionar área de movimento se disponível
    if movimento_area is not None:
        cv2.putText(frame, f"Movimento: {movimento_area:.0f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 2)
    
    return frame

def salvar_imagem(imagem, caminho, qualidade=None):
    """
    Salva uma imagem no caminho especificado com a qualidade definida
    
    Args:
        imagem: Array numpy com a imagem
        caminho: Caminho onde salvar a imagem
        qualidade: Qualidade JPEG (0-100)
    """
    try:
        # Criar diretório se não existir
        diretorio = os.path.dirname(caminho)
        os.makedirs(diretorio, exist_ok=True)
        
        # Definir qualidade padrão se não especificada
        if qualidade is None:
            qualidade = QUALIDADE_JPEG
        
        # Verificar se a imagem é válida
        if imagem is None or imagem.size == 0:
            log_error(f"Imagem inválida ao tentar salvar em {caminho}")
            return False
            
        # Tentar salvar a imagem
        log_captura(f"Salvando imagem em: {caminho}")
        cv2.imwrite(caminho, imagem, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
        
        # Verificar se o arquivo foi criado
        if os.path.exists(caminho):
            log_captura(f"✅ Imagem salva com sucesso: {caminho}")
            return True
        else:
            log_error(f"❌ Falha ao salvar imagem: arquivo não foi criado em {caminho}")
            return False
            
    except Exception as e:
        log_error(f"❌ Erro ao salvar imagem em {caminho}: {str(e)}")
        return False 