"""
Módulo principal do sistema de detecção facial.
Implementa processamento assíncrono para melhor desempenho.
"""
import argparse
from face_detector.controllers.detector_controller import DetectorController
from face_detector.utils.logger import log_info

def main():
    """Função principal do sistema"""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Sistema de Detecção Facial com Processamento Assíncrono')
    parser.add_argument('--rtsp', type=str, help='URL RTSP para conexão com câmera IP')
    parser.add_argument('--camera', type=int, default=None, help='ID da câmera local (0 para webcam padrão)')
    args = parser.parse_args()
    
    # Determinar a fonte de vídeo
    rtsp_url = args.rtsp
    camera_id = args.camera
    
    if rtsp_url:
        log_info(f"Usando stream RTSP: {rtsp_url}")
    elif camera_id is not None:
        log_info(f"Usando câmera local ID: {camera_id}")
    else:
        # Se nenhum argumento for fornecido, usar a URL RTSP padrão das configurações
        rtsp_url = None
        camera_id = None
        log_info("Usando configuração padrão de vídeo")
    
    # Inicializar e executar o controlador
    detector = DetectorController(rtsp_url=rtsp_url, camera_id=camera_id)
    detector.iniciar()

if __name__ == "__main__":
    main() 