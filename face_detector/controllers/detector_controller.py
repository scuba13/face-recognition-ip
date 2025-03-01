"""
Controlador principal para detecção de faces e movimento.
Implementa processamento assíncrono separando captura e processamento.
"""
import cv2
import time
import threading
from queue import Queue
from datetime import datetime
import numpy as np

from face_detector.config.settings import (
    RTSP_URL, PESSOA_CONHECIDA_ENCODING, PESSOA_INFO,
    MOVIMENTO_THRESHOLD, AREA_MINIMA_CONTORNO, FRAMES_APOS_MOVIMENTO,
    MAX_FRAMES_SEM_DETECCAO, MODO_DEBUG, COR_VERDE, COR_AMARELO
)
from face_detector.services.face_detector import FaceDetector
from face_detector.services.motion_detector import MotionDetector
from face_detector.services.video_capture import VideoCapture
from face_detector.utils.logger import log_info, log_debug, log_movimento, log_face, log_captura, log_error
from face_detector.utils.file_utils import criar_estrutura_pastas, carregar_encoding_teste
from face_detector.utils.image_utils import adicionar_info_tela, salvar_imagem

class DetectorController:
    """Controlador principal para detecção de faces e movimento"""
    
    def __init__(self, rtsp_url=None, camera_id=0):
        """Inicializa o controlador com a fonte de vídeo especificada"""
        log_info("Inicializando sistema de detecção facial...")
        
        # Criar estrutura de pastas
        criar_estrutura_pastas()
        
        # Carregar encoding da pessoa conhecida
        self.pessoa_conhecida_encoding = carregar_encoding_teste()
        
        # Fonte de vídeo (RTSP ou câmera)
        self.rtsp_url = rtsp_url if rtsp_url else RTSP_URL
        self.camera_id = camera_id
        self.source = self.rtsp_url if camera_id is None else camera_id
        
        # Inicializar serviços
        self.face_detector = FaceDetector()
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
        self.process_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # Threads
        self.process_thread = None
    
    def iniciar(self):
        """Inicia o processamento do stream de vídeo"""
        # Inicializar captura de vídeo assíncrona
        self.video_capture = VideoCapture(self.source)
        if not self.video_capture.start():
            log_error("Falha ao iniciar captura de vídeo. Verifique a conexão com a câmera.")
            return False
        
        # Informações iniciais
        log_info(f"Pessoa de referência: {PESSOA_INFO['nome']} (ID: {PESSOA_INFO['id']})")
        log_info("Controles: ESC = Sair")
        log_info(f"Detecção baseada em movimento: {FRAMES_APOS_MOVIMENTO} frames após movimento")
        log_info(f"Processando e salvando faces APENAS após detecção de movimento")
        log_info(f"Modo de depuração: {MODO_DEBUG}")
        
        # Iniciar thread de processamento
        self.running = True
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()
        
        # Criar janela com tamanho ajustável
        cv2.namedWindow("Detector de Faces por Movimento", cv2.WINDOW_NORMAL)
        
        # Loop principal (thread principal - UI)
        self._main_loop()
        
        return True
    
    def _main_loop(self):
        """Loop principal para captura e exibição"""
        try:
            consecutive_errors = 0
            last_successful_read_time = time.time()
            
            while self.running:
                # Ler o próximo frame
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    
                    # Se tivermos muitos erros consecutivos, exibir uma mensagem de status
                    if consecutive_errors % 5 == 0:
                        log_info(f"Erro ao ler frame. Aguardando... ({consecutive_errors} erros consecutivos)")
                    
                    # Se não recebemos frames por muito tempo, exibir uma tela de status
                    current_time = time.time()
                    if current_time - last_successful_read_time > 5.0:
                        # Criar uma imagem preta com mensagem de status
                        status_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(status_frame, "Aguardando conexão...", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(status_frame, f"Última conexão: {int(current_time - last_successful_read_time)}s atrás", 
                                    (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Detector de Faces por Movimento", status_frame)
                    
                    # Pequena pausa para não sobrecarregar a CPU
                    time.sleep(0.1)
                    
                    # Verificar teclas mesmo sem frame
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        log_info("Tecla ESC pressionada. Encerrando...")
                        break
                        
                    continue
                
                # Resetar contadores quando recebemos um frame com sucesso
                consecutive_errors = 0
                last_successful_read_time = time.time()
                
                # Guardar uma cópia do frame para processamento
                self.ultimo_frame = frame.copy()
                
                # Enviar para processamento se a fila não estiver cheia
                if not self.process_queue.full():
                    self.process_queue.put((frame.copy(), time.time()))
                
                # Verificar se há resultados processados para exibir
                frame_processado = None
                while not self.result_queue.empty():
                    frame_processado, _ = self.result_queue.get()
                
                # Se não houver frame processado, usar o original com informações básicas
                if frame_processado is None:
                    frame_processado = frame.copy()
                    # Adicionar informações básicas
                    adicionar_info_tela(frame_processado)
                    # Adicionar FPS
                    fps = self.video_capture.get_fps()
                    cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
                
                # Mostrar frame processado
                cv2.imshow("Detector de Faces por Movimento", frame_processado)
                
                # Capturar tecla
                key = cv2.waitKey(1) & 0xFF
                
                # ESC para sair
                if key == 27:
                    log_info("Tecla ESC pressionada. Encerrando...")
                    break
        except Exception as e:
            log_error(f"Erro no loop principal: {str(e)}")
        finally:
            self.finalizar()
    
    def _process_frames(self):
        """Thread de processamento de frames"""
        frame_anterior = None
        
        while self.running:
            try:
                # Obter próximo frame para processamento
                if self.process_queue.empty():
                    time.sleep(0.01)  # Pequena pausa para não consumir CPU
                    continue
                
                frame, timestamp = self.process_queue.get()
                
                # Se for o primeiro frame, inicializar frame_anterior
                if frame_anterior is None:
                    frame_anterior = frame.copy()
                    continue
                
                # Criar cópia do frame para processamento
                frame_processado = frame.copy()
                
                # Detectar movimento
                movimento_detectado, movimento_area, frame_com_movimento = self.motion_detector.detectar(
                    frame.copy(), frame_anterior.copy())
                
                # Atualizar frame anterior para próxima detecção de movimento
                frame_anterior = frame.copy()
                
                # Se detectou movimento, salvar frame e configurar para processar os próximos frames
                if movimento_detectado:
                    self.frames_sem_deteccao = 0  # Resetar contador de frames sem detecção
                    
                    # Salvar frame com movimento
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    movimento_filename = f"capturas/movimento/movimento_{movimento_area:.0f}_{timestamp_str}.jpg"
                    salvar_imagem(frame_com_movimento, movimento_filename)
                    
                    log_movimento(f"Movimento detectado (área: {movimento_area:.0f}) - Limiar: {MOVIMENTO_THRESHOLD}")
                    log_captura(f"Frame de movimento salvo: {movimento_filename}")
                    
                    # Configurar para processar os próximos frames
                    self.frames_restantes_apos_movimento = FRAMES_APOS_MOVIMENTO
                    log_debug(f"Configurado para processar os próximos {FRAMES_APOS_MOVIMENTO} frames")
                else:
                    self.frames_sem_deteccao += 1
                    if self.frames_sem_deteccao >= MAX_FRAMES_SEM_DETECCAO:
                        self.frames_sem_deteccao = 0
                        log_info(f"Nenhum movimento detectado nos últimos {MAX_FRAMES_SEM_DETECCAO} frames")
                
                # Determinar se deve processar faces neste frame
                deve_processar_faces = False
                motivo_processamento = ""
                
                # Apenas processar faces após detecção de movimento
                if self.frames_restantes_apos_movimento > 0:
                    deve_processar_faces = True
                    motivo_processamento = f"Frame após movimento ({self.frames_restantes_apos_movimento} restantes)"
                
                # Processar faces se necessário
                if deve_processar_faces:
                    log_debug(f"Processando faces - Motivo: {motivo_processamento}")
                    
                    # Processar faces no frame atual
                    frame_processado, face_encontrada = self.face_detector.processar_faces_no_frame(
                        frame_processado, self.pessoa_conhecida_encoding, PESSOA_INFO)
                    
                    # Se encontrou face durante o processamento após movimento, registrar
                    if face_encontrada and self.frames_restantes_apos_movimento > 0:
                        log_face(f"Face detectada após movimento! Frames restantes: {self.frames_restantes_apos_movimento}")
                    
                    # Decrementar contador de frames após movimento
                    if self.frames_restantes_apos_movimento > 0:
                        self.frames_restantes_apos_movimento -= 1
                
                # Adicionar informações na tela
                adicionar_info_tela(frame_processado)
                
                # Adicionar FPS
                fps = self.video_capture.get_fps()
                cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
                
                if movimento_detectado:
                    cv2.putText(frame_processado, f"Movimento: {movimento_area}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 2)
                
                # Enviar frame processado para exibição
                if not self.result_queue.full():
                    self.result_queue.put((frame_processado, timestamp))
            
            except Exception as e:
                log_error(f"Erro no processamento: {str(e)}")
                time.sleep(0.1)
    
    def finalizar(self):
        """Finaliza o controlador e libera recursos"""
        self.running = False
        
        # Aguardar thread de processamento
        if self.process_thread is not None:
            self.process_thread.join(timeout=1.0)
        
        # Parar captura de vídeo
        if hasattr(self, 'video_capture'):
            self.video_capture.stop()
        
        # Fechar janelas
        cv2.destroyAllWindows()
        
        log_info("Sistema finalizado.") 