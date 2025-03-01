"""
Serviço para detecção de movimento em frames de vídeo.
"""
import cv2
from datetime import datetime
from face_detector.config.settings import (
    MOVIMENTO_THRESHOLD, AREA_MINIMA_CONTORNO, COR_VERDE, COR_VERMELHO
)
from face_detector.utils.logger import log_movimento, log_captura
from face_detector.utils.image_utils import salvar_imagem

class MotionDetector:
    """Classe para detecção de movimento em frames de vídeo"""
    
    def __init__(self, threshold=None, area_minima=None):
        """Inicializa o detector de movimento com os parâmetros especificados"""
        self.threshold = threshold if threshold is not None else MOVIMENTO_THRESHOLD
        self.area_minima = area_minima if area_minima is not None else AREA_MINIMA_CONTORNO
    
    def detectar(self, frame1, frame2):
        """Detecta movimento entre dois frames consecutivos"""
        # Converter para escala de cinza
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reduzir ruído
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        # Calcular diferença absoluta entre os frames
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # Aplicar threshold para destacar áreas com movimento
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilatar o threshold para preencher buracos
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        movimento_detectado = False
        movimento_area = 0
        
        # Verificar se há contornos significativos
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_minima:  # Filtrar contornos pequenos (ruído)
                movimento_area += area
                # Desenhar retângulo ao redor do movimento
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), COR_VERDE, 2)
        
        # Verificar se a área total de movimento é significativa
        if movimento_area > self.threshold:
            movimento_detectado = True
            # Adicionar texto indicando movimento
            cv2.putText(frame1, f"Movimento: {movimento_area}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COR_VERMELHO, 2)
        
        return movimento_detectado, movimento_area, frame1
    
    def salvar_frame_movimento(self, frame, movimento_area):
        """Salva o frame com movimento detectado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        movimento_filename = f"capturas/movimento/movimento_{movimento_area:.0f}_{timestamp}.jpg"
        
        salvar_imagem(frame, movimento_filename)
        
        log_movimento(f"Movimento detectado (área: {movimento_area:.0f}) - Limiar: {self.threshold}")
        log_captura(f"Frame de movimento salvo: {movimento_filename}")
        
        return movimento_filename 