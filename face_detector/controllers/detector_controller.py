"""
Controlador principal para detecção de faces e movimento.
Implementa processamento assíncrono separando captura e processamento.
"""
import cv2
import time
import threading
import queue
from queue import Queue
from datetime import datetime
import numpy as np
import concurrent.futures
import signal
import os

from face_detector.config.settings import (
    RTSP_URL, PESSOA_CONHECIDA_ENCODING, PESSOA_INFO,
    MOVIMENTO_THRESHOLD, AREA_MINIMA_CONTORNO, FRAMES_APOS_MOVIMENTO,
    MAX_FRAMES_SEM_DETECCAO, MODO_DEBUG, COR_VERDE, COR_AMARELO,
    INTERVALO_MINIMO_MOVIMENTO, INTERVALO_MINIMO_FACE, TEMPO_EXPIRACAO_FACE,
    BUFFER_SIZE_CAPTURA, TAXA_FPS_CAPTURA, TAXA_FPS_UI
)
from face_detector.services.face_detector import FaceDetector
from face_detector.services.motion_detector import MotionDetector
from face_detector.services.video_capture import VideoCapture
from face_detector.utils.logger import log_info, log_debug, log_movimento, log_face, log_captura, log_error
from face_detector.utils.file_utils import criar_estrutura_pastas, carregar_encoding_teste
from face_detector.utils.image_utils import adicionar_info_tela, salvar_imagem

class DetectorController:
    """Controlador principal para detecção de faces e movimento com processamento paralelo"""
    
    def __init__(self, rtsp_url=None, camera_id=0, num_workers=4):
        """
        Inicializa o controlador com a fonte de vídeo especificada
        
        Args:
            rtsp_url: URL RTSP para conexão com câmera IP
            camera_id: ID da câmera local (0 para webcam padrão)
            num_workers: Número de workers para processamento paralelo
        """
        log_info("Inicializando sistema de detecção facial com processamento paralelo...")
        
        # Criar estrutura de pastas
        criar_estrutura_pastas()
        
        # Carregar encoding da pessoa conhecida
        self.pessoa_conhecida_encoding = carregar_encoding_teste()
        
        # Fonte de vídeo (RTSP ou câmera)
        self.rtsp_url = rtsp_url if rtsp_url else RTSP_URL
        self.camera_id = camera_id
        self.source = self.rtsp_url if camera_id is None else camera_id
        
        # Número de workers para processamento paralelo
        self.num_workers = num_workers
        
        # Inicializar serviços
        self.face_detector = FaceDetector(max_workers=num_workers)
        self.motion_detector = MotionDetector(
            threshold=MOVIMENTO_THRESHOLD,
            area_minima=AREA_MINIMA_CONTORNO
        )
        
        # Variáveis para controle de processamento
        self.frames_restantes_apos_movimento = 0
        self.frames_sem_deteccao = 0
        self.ultimo_frame = None
        self.running = False
        self.connection_errors = 0
        self.max_connection_errors = 20
        
        # Filas para comunicação entre threads
        self.capture_queue = Queue(maxsize=10)  # Frames capturados
        self.motion_queue = Queue(maxsize=10)   # Frames com movimento detectado
        self.face_queue = Queue(maxsize=10)     # Frames para processamento facial
        self.result_queue = Queue(maxsize=10)   # Frames processados para exibição
        
        # Threads
        self.capture_thread = None
        self.motion_thread = None
        self.face_thread = None
        self.worker_threads = []
        
        # Pool de threads para processamento paralelo
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        
        # Controle de estatísticas
        self.stats = {
            'frames_capturados': 0,
            'frames_processados': 0,
            'movimento_detectado': 0,
            'faces_detectadas': 0,
            'faces_reconhecidas': 0,
            'tempo_inicio': time.time()
        }
        
        # Flag para controle de finalização
        self.shutdown_requested = False
    
    def iniciar(self):
        """Inicia o processamento do stream de vídeo com threads separadas"""
        # Inicializar captura de vídeo assíncrona com buffer menor para menor latência
        self.video_capture = VideoCapture(self.source, buffer_size=BUFFER_SIZE_CAPTURA)
        if not self.video_capture.start():
            log_error("Falha ao iniciar captura de vídeo. Verifique a conexão com a câmera.")
            return False
        
        # Informações iniciais
        log_info(f"Pessoa de referência: {PESSOA_INFO['nome']} (ID: {PESSOA_INFO['id']})")
        log_info("Controles: ESC = Sair")
        log_info(f"Detecção baseada em movimento: {FRAMES_APOS_MOVIMENTO} frames após movimento")
        log_info(f"Processando e salvando faces APENAS após detecção de movimento")
        log_info(f"Modo de depuração: {MODO_DEBUG}")
        log_info(f"Processamento paralelo com {self.num_workers} workers")
        
        # Iniciar threads de processamento
        self.running = True
        
        # Thread de captura
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Thread de detecção de movimento
        self.motion_thread = threading.Thread(target=self._motion_detection_loop, daemon=True)
        self.motion_thread.start()
        
        # Thread de processamento facial
        self.face_thread = threading.Thread(target=self._face_processing_loop, daemon=True)
        self.face_thread.start()
        
        # Thread de monitoramento de estatísticas
        self.stats_thread = threading.Thread(target=self._monitor_stats, daemon=True)
        self.stats_thread.start()
        
        # Criar janela com tamanho ajustável
        cv2.namedWindow("Detector de Faces por Movimento", cv2.WINDOW_NORMAL)
        
        # Configurar handler para SIGINT (Ctrl+C)
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        
        # Loop principal (thread principal - UI)
        self._main_loop()
        
        return True
    
    def _handle_sigint(self, sig, frame):
        """Handler para SIGINT (Ctrl+C)"""
        log_info("Sinal de interrupção recebido (Ctrl+C). Finalizando...")
        self.shutdown_requested = True
        self.running = False
    
    def _capture_loop(self):
        """Thread dedicada para captura de frames"""
        log_info("Thread de captura iniciada")
        
        last_frame_time = time.time()
        frame_interval = 1.0 / TAXA_FPS_CAPTURA  # Limitar a taxa de FPS configurada
        
        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Limitar taxa de captura para não sobrecarregar o sistema
                if elapsed < frame_interval:
                    time.sleep(0.001)  # Pequena pausa
                    continue
                
                # Ler o próximo frame
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Atualizar timestamp
                last_frame_time = current_time
                
                # Guardar uma cópia do frame para referência
                self.ultimo_frame = frame.copy()
                
                # Incrementar contador de estatísticas
                self.stats['frames_capturados'] += 1
                
                # Enviar para processamento se a fila não estiver cheia
                if not self.capture_queue.full():
                    self.capture_queue.put((frame.copy(), current_time))
                
            except Exception as e:
                log_error(f"Erro na thread de captura: {str(e)}")
                time.sleep(0.1)
    
    def _motion_detection_loop(self):
        """Thread dedicada para detecção de movimento"""
        log_info("Thread de detecção de movimento iniciada")
        
        frame_anterior = None
        movimento_count = 0
        ultimo_movimento = 0  # Timestamp do último movimento detectado
        
        while self.running:
            try:
                # Obter próximo frame para processamento de movimento
                if self.capture_queue.empty():
                    time.sleep(0.01)  # Pequena pausa para não consumir CPU
                    continue
                
                frame, timestamp = self.capture_queue.get()
                
                # Se for o primeiro frame, inicializar frame_anterior
                if frame_anterior is None:
                    frame_anterior = frame.copy()
                    continue
                
                # Verificar se já passou tempo suficiente desde a última detecção de movimento
                tempo_desde_ultimo_movimento = timestamp - ultimo_movimento
                
                # Detectar movimento
                movimento_detectado, movimento_area, frame_com_movimento = self.motion_detector.detectar(
                    frame.copy(), frame_anterior.copy())
                
                # Atualizar frame anterior para próxima detecção de movimento
                frame_anterior = frame.copy()
                
                # Se detectou movimento e passou tempo suficiente desde a última detecção
                if movimento_detectado and (tempo_desde_ultimo_movimento >= INTERVALO_MINIMO_MOVIMENTO or self.frames_restantes_apos_movimento == 0):
                    self.frames_sem_deteccao = 0  # Resetar contador de frames sem detecção
                    self.stats['movimento_detectado'] += 1
                    ultimo_movimento = timestamp  # Atualizar timestamp do último movimento
                    
                    # Limitar logs de movimento para reduzir poluição no terminal
                    movimento_count += 1
                    if movimento_count % 5 == 0:  # Logar apenas a cada 5 detecções
                        log_movimento(f"Movimento detectado (área: {movimento_area:.0f}) - Limiar: {MOVIMENTO_THRESHOLD}")
                    
                    # Salvar frame com movimento
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    movimento_filename = f"capturas/movimento/movimento_{movimento_area:.0f}_{timestamp_str}.jpg"
                    salvar_imagem(frame_com_movimento, movimento_filename)
                    
                    # Configurar para processar 5 frames após movimento (para ambiente de linha de produção)
                    self.frames_restantes_apos_movimento = FRAMES_APOS_MOVIMENTO  # Voltando para 5 frames
                    
                    # Enviar para processamento facial
                    if not self.face_queue.full():
                        self.face_queue.put((frame.copy(), timestamp, movimento_area))
                elif self.frames_restantes_apos_movimento > 0:
                    # Processar frames restantes após movimento
                    self.frames_restantes_apos_movimento -= 1
                    
                    # Enviar para processamento facial
                    if not self.face_queue.full():
                        self.face_queue.put((frame.copy(), timestamp, movimento_area if movimento_detectado else 0))
                else:
                    self.frames_sem_deteccao += 1
                    
                    # Enviar para exibição direta (sem processamento facial)
                    frame_processado = frame.copy()
                    adicionar_info_tela(frame_processado)
                    
                    # Adicionar FPS
                    fps = self.video_capture.get_fps()
                    cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
                    
                    # Enviar para exibição
                    if not self.result_queue.full():
                        self.result_queue.put((frame_processado, timestamp))
            
            except Exception as e:
                log_error(f"Erro na thread de detecção de movimento: {str(e)}")
                time.sleep(0.1)
    
    def _face_processing_loop(self):
        """Thread dedicada para processamento facial"""
        log_info("Thread de processamento facial iniciada")
        
        ultima_face_timestamp = 0
        faces_processadas = {}  # Dicionário para rastrear faces já processadas recentemente
        
        while self.running:
            try:
                # Obter próximo frame para processamento facial
                if self.face_queue.empty():
                    time.sleep(0.01)  # Pequena pausa para não consumir CPU
                    continue
                
                frame, timestamp, movimento_area = self.face_queue.get()
                
                # Verificar se já passou tempo suficiente desde o último processamento facial
                tempo_desde_ultima_face = timestamp - ultima_face_timestamp
                
                # Processar faces no frame
                frame_processado, face_encontrada = self.face_detector.processar_faces_no_frame(
                    frame, self.pessoa_conhecida_encoding, PESSOA_INFO)
                
                # Se encontrou face, atualizar timestamp
                if face_encontrada:
                    # Só atualizar o timestamp se passou tempo suficiente ou se é uma nova detecção
                    if tempo_desde_ultima_face >= INTERVALO_MINIMO_FACE:
                        ultima_face_timestamp = timestamp
                    
                    self.stats['faces_detectadas'] += 1
                
                # Adicionar informações na tela
                adicionar_info_tela(frame_processado)
                
                # Adicionar FPS e informações de movimento
                fps = self.video_capture.get_fps()
                cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
                
                cv2.putText(frame_processado, f"Movimento: {movimento_area}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 2)
                
                # Enviar frame processado para exibição
                if not self.result_queue.full():
                    self.result_queue.put((frame_processado, timestamp))
                
                # Decrementar contador de frames após movimento
                if self.frames_restantes_apos_movimento > 0:
                    self.frames_restantes_apos_movimento -= 1
                
                # Incrementar contador de frames processados
                self.stats['frames_processados'] += 1
                
            except Exception as e:
                log_error(f"Erro na thread de processamento facial: {str(e)}")
                time.sleep(0.1)
    
    def _main_loop(self):
        """Loop principal para exibição de frames processados"""
        try:
            ui_frame_interval = 1.0 / TAXA_FPS_UI
            last_ui_update = time.time()
            
            while self.running and not self.shutdown_requested:
                current_time = time.time()
                
                # Verificar se é hora de atualizar a UI
                if current_time - last_ui_update < ui_frame_interval:
                    # Verificar teclas mesmo sem atualizar a UI
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        log_info("Tecla ESC pressionada. Encerrando...")
                        self.running = False
                        break
                    
                    # Pequena pausa para não sobrecarregar a CPU
                    time.sleep(0.001)
                    continue
                
                # Atualizar timestamp da última atualização da UI
                last_ui_update = current_time
                
                # Verificar se há resultados processados para exibir
                frame_processado = None
                while not self.result_queue.empty():
                    frame_processado, _ = self.result_queue.get()
                
                # Se não houver frame processado, usar o último frame com informações básicas
                if frame_processado is None and self.ultimo_frame is not None:
                    frame_processado = self.ultimo_frame.copy()
                    # Adicionar informações básicas
                    adicionar_info_tela(frame_processado)
                    # Adicionar FPS
                    fps = self.video_capture.get_fps()
                    cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
                
                # Mostrar frame processado se disponível
                if frame_processado is not None:
                    cv2.imshow("Detector de Faces por Movimento", frame_processado)
                
                # Capturar tecla
                key = cv2.waitKey(1) & 0xFF
                
                # ESC para sair
                if key == 27:
                    log_info("Tecla ESC pressionada. Encerrando...")
                    break
                
        except KeyboardInterrupt:
            log_info("Interrupção de teclado detectada. Encerrando...")
        except Exception as e:
            log_error(f"Erro no loop principal: {str(e)}")
        finally:
            # Restaurar o handler original para SIGINT
            signal.signal(signal.SIGINT, self.original_sigint_handler)
            self.finalizar()
    
    def _monitor_stats(self):
        """Thread para monitorar estatísticas de desempenho"""
        while self.running:
            try:
                # Calcular estatísticas
                tempo_total = time.time() - self.stats['tempo_inicio']
                fps_medio = self.stats['frames_capturados'] / tempo_total if tempo_total > 0 else 0
                
                # Tamanhos das filas
                capture_size = self.capture_queue.qsize()
                face_size = self.face_queue.qsize()
                result_size = self.result_queue.qsize()
                
                # Logar estatísticas
                log_info(f"Estatísticas: {self.stats['frames_capturados']} frames capturados, "
                         f"{self.stats['frames_processados']} processados, "
                         f"{self.stats['movimento_detectado']} movimentos, "
                         f"{self.stats['faces_detectadas']} faces. "
                         f"FPS médio: {fps_medio:.1f}, "
                         f"Filas: Captura={capture_size}, Face={face_size}, Resultado={result_size}")
                
                # Aguardar antes da próxima atualização
                time.sleep(15.0)
                
            except Exception as e:
                log_error(f"Erro ao monitorar estatísticas: {str(e)}")
                time.sleep(5.0)
    
    def finalizar(self):
        """Finaliza o controlador e libera recursos"""
        log_info("Finalizando sistema...")
        self.running = False
        self.shutdown_requested = True
        
        # Aguardar threads
        threads = [
            self.capture_thread,
            self.motion_thread,
            self.face_thread,
            self.stats_thread
        ]
        
        for thread in threads:
            if thread is not None:
                thread.join(timeout=1.0)
        
        # Encerrar pool de threads
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
        # Parar captura de vídeo
        if hasattr(self, 'video_capture'):
            self.video_capture.stop()
        
        # Fechar janelas
        cv2.destroyAllWindows()
        
        log_info("Sistema finalizado.") 