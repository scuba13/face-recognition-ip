"""
Serviço para captura de vídeo de câmeras IP ou webcams.
Implementa captura assíncrona com buffer de frames otimizado para baixa latência.
"""
import cv2
import os
import time
import threading
from queue import Queue
import numpy as np
from face_detector.config.settings import (
    RESOLUCAO_CAPTURA, BUFFER_SIZE_CAPTURA, TAXA_FPS_CAPTURA
)
from face_detector.utils.logger import log_info, log_error

class VideoCapture:
    """Classe para captura de vídeo assíncrona otimizada para baixa latência"""
    
    def __init__(self, source, buffer_size=None, resize_width=None):
        """
        Inicializa o capturador de vídeo
        
        Args:
            source: URL RTSP ou índice da câmera
            buffer_size: Tamanho máximo do buffer de frames (menor = menor latência)
            resize_width: Largura para redimensionar frames (None = sem redimensionamento)
        """
        self.source = source
        self.buffer_size = buffer_size if buffer_size is not None else BUFFER_SIZE_CAPTURA
        self.resize_width = resize_width
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.cap = None
        self.thread = None
        self.last_frame = None
        self.frame_count = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2  # segundos
        self.drop_count = 0  # Contador de frames descartados
    
    def start(self):
        """Inicia a captura de vídeo em uma thread separada"""
        log_info(f"Iniciando captura de vídeo assíncrona: {self.source}")
        
        # Configurar captura
        self._setup_capture()
        
        if not self.cap.isOpened():
            log_info("Erro ao abrir o stream de vídeo. Tentando novamente...")
            time.sleep(2)
            self._setup_capture()
            if not self.cap.isOpened():
                log_info("Falha ao iniciar captura de vídeo após segunda tentativa.")
                return False
        
        # Iniciar thread de captura
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        log_info("Thread de captura iniciada com sucesso")
        return True
    
    def _setup_capture(self):
        """Configura a captura de vídeo com otimizações para baixa latência"""
        log_info(f"Configurando captura de vídeo: {self.source}")
        
        # Liberar recursos anteriores se existirem
        if self.cap is not None:
            self.cap.release()
        
        # Configurações avançadas para o OpenCV - otimizadas para RTSP
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            # Configurações mais robustas para RTSP com foco em baixa latência
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|fflags;discardcorrupt|stimeout;20000000|max_delay;500000|reorder_queue_size;0|buffer_size;1024000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            
            # Configurações específicas para RTSP para reduzir latência
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer para menor latência
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Buffer pequeno para câmeras locais
        
        # Tentar configurar resolução mais alta
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCAO_CAPTURA[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCAO_CAPTURA[1])
        
        # Tentar configurar para usar hardware acceleration se disponível
        try:
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except:
            pass  # Ignorar se não suportado
        
        if self.cap.isOpened():
            # Verificar resolução real obtida
            largura_real = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura_real = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log_info(f"Stream de vídeo aberto com resolução: {largura_real}x{altura_real}")
            # Resetar contador de tentativas de reconexão
            self.reconnect_attempts = 0
    
    def _update(self):
        """Atualiza continuamente o buffer de frames com otimizações para desempenho"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_drop_log = 0
        frame_interval = 1.0 / TAXA_FPS_CAPTURA  # Limitar a taxa de FPS configurada
        last_frame_time = time.time()
        
        while not self.stopped:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Limitar taxa de captura para não sobrecarregar o sistema
                if elapsed < frame_interval:
                    time.sleep(0.001)  # Pequena pausa
                    continue
                
                # Atualizar timestamp
                last_frame_time = current_time
                
                if not self.cap.isOpened():
                    log_info("Conexão perdida. Tentando reconectar...")
                    self._setup_capture()
                    if not self.cap.isOpened():
                        self.reconnect_attempts += 1
                        if self.reconnect_attempts > self.max_reconnect_attempts:
                            log_info(f"Falha após {self.max_reconnect_attempts} tentativas. Aguardando mais tempo...")
                            time.sleep(5)
                            self.reconnect_attempts = 0
                        else:
                            time.sleep(self.reconnect_delay)
                        continue
            
                # Ler o próximo frame
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        log_info(f"Múltiplos erros consecutivos ({consecutive_errors}). Reiniciando conexão...")
                        self.cap.release()
                        time.sleep(2)
                        self._setup_capture()
                        consecutive_errors = 0
                    else:
                        log_info(f"Erro ao ler frame ({consecutive_errors}/{max_consecutive_errors}). Aguardando...")
                        time.sleep(0.5)
                    continue
                else:
                    consecutive_errors = 0  # Resetar contador de erros consecutivos
                
                # Redimensionar frame se necessário para melhorar desempenho
                if self.resize_width is not None and frame.shape[1] > self.resize_width:
                    # Calcular nova altura mantendo a proporção
                    aspect_ratio = frame.shape[0] / frame.shape[1]
                    new_height = int(self.resize_width * aspect_ratio)
                    frame = cv2.resize(frame, (self.resize_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Calcular FPS
                self.fps_counter += 1
                if (time.time() - self.fps_start_time) > 1:
                    self.fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Incrementar contador de frames
                self.frame_count += 1
                
                # Se o buffer estiver cheio, remover o frame mais antigo e contar como descartado
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.drop_count += 1
                        
                        # Logar a cada 100 frames descartados
                        if self.drop_count - last_drop_log >= 100:
                            log_info(f"Buffer cheio: {self.drop_count} frames descartados até agora")
                            last_drop_log = self.drop_count
                    except:
                        pass
                
                # Adicionar o novo frame ao buffer
                try:
                    self.frame_queue.put_nowait(frame)
                    self.last_frame = frame
                except:
                    pass
            except Exception as e:
                log_error(f"Erro ao atualizar buffer de frames: {str(e)}")
                time.sleep(2)
    
    def read(self):
        """Lê o próximo frame do buffer com verificações de validade"""
        if self.stopped:
            return False, None
            
        if self.frame_queue.empty():
            if self.last_frame is not None:
                return True, self.last_frame.copy()  # Retornar o último frame válido se o buffer estiver vazio
            return False, None
        
        # Obter o próximo frame do buffer
        try:
            frame = self.frame_queue.get()
            if frame is None or frame.size == 0:
                # Frame inválido, retornar o último frame válido
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
                return False, None
            return True, frame
        except:
            # Em caso de erro, retornar o último frame válido
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    def get_fps(self):
        """Retorna o FPS atual"""
        return self.fps
    
    def get_frame_count(self):
        """Retorna o número total de frames capturados"""
        return self.frame_count
    
    def get_drop_count(self):
        """Retorna o número de frames descartados devido ao buffer cheio"""
        return self.drop_count
    
    def get_queue_size(self):
        """Retorna o tamanho atual da fila de frames"""
        return self.frame_queue.qsize()
    
    def stop(self):
        """Para a captura de vídeo com timeout reduzido para evitar bloqueio"""
        self.stopped = True
        try:
            if self.thread is not None:
                self.thread.join(timeout=0.5)  # Timeout reduzido para evitar bloqueio
            if self.cap is not None:
                self.cap.release()
            log_info("Captura de vídeo encerrada")
        except Exception as e:
            log_error(f"Erro ao encerrar captura de vídeo: {str(e)}") 