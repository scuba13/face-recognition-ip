"""
Serviço para detecção e reconhecimento facial.
Implementa processamento paralelo para melhor desempenho.
"""
import cv2
import face_recognition
from datetime import datetime
import concurrent.futures
import numpy as np
from face_detector.config.settings import (
    FACE_SIMILARITY_THRESHOLD, MODELO_FACE, NUM_JITTERS,
    COR_VERDE, COR_VERMELHO, QUALIDADE_JPEG
)
from face_detector.utils.logger import log_face, log_captura
from face_detector.utils.image_utils import melhorar_imagem, salvar_imagem

class FaceDetector:
    """Classe para detecção e reconhecimento facial com processamento paralelo"""
    
    def __init__(self, similarity_threshold=None, modelo=None, num_jitters=None, max_workers=4):
        """
        Inicializa o detector facial com os parâmetros especificados
        
        Args:
            similarity_threshold: Limiar de similaridade para reconhecimento
            modelo: Modelo de detecção (hog ou cnn)
            num_jitters: Número de vezes para amostrar a face durante o encoding
            max_workers: Número máximo de threads para processamento paralelo
        """
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else FACE_SIMILARITY_THRESHOLD
        self.modelo = modelo if modelo is not None else MODELO_FACE
        self.num_jitters = num_jitters if num_jitters is not None else NUM_JITTERS
        self.max_workers = max_workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def detectar_faces(self, frame):
        """Detecta faces em um frame e retorna as localizações e encodings"""
        # Aplicar melhorias na imagem antes da detecção
        frame_melhorado = melhorar_imagem(frame)
        
        # Reduzir o tamanho do frame para processamento mais rápido
        # Usando 0.5 em vez de 0.25 para melhor qualidade
        small_frame = cv2.resize(frame_melhorado, (0, 0), fx=0.5, fy=0.5)
        
        # Converter de BGR (OpenCV) para RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Encontrar todas as faces no frame com mais precisão
        face_locations = face_recognition.face_locations(rgb_small_frame, 
                                                        model=self.modelo, 
                                                        number_of_times_to_upsample=1)
        
        # Logar quando encontrar faces (importante para ambiente de linha de produção)
        if face_locations:
            log_face(f"✅ ENCONTRADAS {len(face_locations)} FACES")
        
        # Calcular os encodings das faces com mais precisão
        face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                        face_locations, 
                                                        num_jitters=self.num_jitters)
        
        # Ajustar as localizações das faces para o tamanho original do frame
        original_face_locations = []
        for (top, right, bottom, left) in face_locations:
            # Multiplicar por 2 porque usamos fx=0.5
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Expandir um pouco a área da face para capturar melhor
            height = bottom - top
            width = right - left
            
            # Expandir em 20% para cada lado
            top = max(0, int(top - height * 0.2))
            bottom = int(bottom + height * 0.2)
            left = max(0, int(left - width * 0.2))
            right = int(right + width * 0.2)
            
            original_face_locations.append((top, right, bottom, left))
        
        return original_face_locations, face_encodings
    
    def _processar_face_individual(self, args):
        """
        Processa uma face individual (para execução paralela)
        
        Args:
            args: Tupla contendo (frame, face_location, face_encoding, pessoa_conhecida_encoding, pessoa_info, index)
        
        Returns:
            Tupla com (face_location, match, similarity, face_filename, index)
        """
        frame, face_location, face_encoding, pessoa_conhecida_encoding, pessoa_info, index = args
        
        # Calcular a distância entre os encodings (menor = mais similar)
        face_distances = face_recognition.face_distance([pessoa_conhecida_encoding], face_encoding)
        
        # Verificar se a face é similar o suficiente
        match = face_distances[0] <= self.similarity_threshold
        similarity = 1 - face_distances[0]  # Converter distância para similaridade (0-1)
        
        # Salvar a face
        face_filename = self.salvar_face(
            frame, face_location, match, similarity, pessoa_info if match else None)
        
        # Logar resultado para todas as faces (importante em ambiente de linha de produção)
        if match:
            log_face(f"👤 Face {index+1}: {pessoa_info['nome']} RECONHECIDO "
                  f"(Similaridade: {similarity:.2f})")
        else:
            log_face(f"👤 Face {index+1}: PESSOA DESCONHECIDA "
                  f"(Similaridade: {similarity:.2f})")
        
        return (face_location, match, similarity, face_filename, index)
    
    def comparar_faces(self, face_encodings, pessoa_conhecida_encoding):
        """Compara os encodings das faces detectadas com o encoding conhecido"""
        resultados = []
        
        for face_encoding in face_encodings:
            # Calcular a distância entre os encodings (menor = mais similar)
            face_distances = face_recognition.face_distance([pessoa_conhecida_encoding], face_encoding)
            
            # Verificar se a face é similar o suficiente
            match = face_distances[0] <= self.similarity_threshold
            similarity = 1 - face_distances[0]  # Converter distância para similaridade (0-1)
            
            resultados.append((match, similarity))
        
        return resultados
    
    def salvar_face(self, frame, face_location, match, similarity, pessoa_info=None):
        """Salva uma face detectada"""
        top, right, bottom, left = face_location
        
        # Recortar a face
        face_img = frame[top:bottom, left:right]
        
        # Melhorar a qualidade da imagem recortada
        face_img = melhorar_imagem(face_img)
        
        # Redimensionar para um tamanho padrão para melhor comparação
        face_img = cv2.resize(face_img, (300, 300), interpolation=cv2.INTER_LANCZOS4)
        
        # Criar nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Adicionar informação de match ao nome do arquivo
        if match:
            status = "match"
            nome_pessoa = pessoa_info['nome'].replace(" ", "_").lower()
            filename = f"capturas/faces/match/{nome_pessoa}_{similarity:.2f}_{timestamp}.jpg"
        else:
            status = "desconhecido"
            filename = f"capturas/faces/desconhecido/desconhecido_{similarity:.2f}_{timestamp}.jpg"
        
        # Salvar imagem com alta qualidade
        salvar_imagem(face_img, filename, QUALIDADE_JPEG)
        
        # Salvar também o frame completo com a anotação
        frame_filename = f"capturas/frames/frame_{status}_{timestamp}.jpg"
        salvar_imagem(frame, frame_filename, QUALIDADE_JPEG)
        
        return filename
    
    def processar_faces_no_frame(self, frame, pessoa_conhecida_encoding, pessoa_info):
        """Processa faces em um único frame usando processamento paralelo"""
        # Detectar faces
        face_locations, face_encodings = self.detectar_faces(frame)
        
        # Se não encontrou faces, retornar o frame original
        if not face_locations:
            return frame, False
        
        # Logar quantidade de faces detectadas (importante para ambiente de linha de produção)
        log_face(f"Detectadas {len(face_locations)} faces na imagem")
        
        # Preparar argumentos para processamento paralelo
        args_list = [
            (frame.copy(), face_location, face_encoding, pessoa_conhecida_encoding, pessoa_info, i)
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings))
        ]
        
        # Processar faces em paralelo
        resultados = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._processar_face_individual, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    resultados.append(future.result())
                except Exception as e:
                    log_face(f"Erro ao processar face: {str(e)}")
        
        # Ordenar resultados pelo índice original
        resultados.sort(key=lambda x: x[4])
        
        # Desenhar resultados no frame
        for face_location, match, similarity, _, _ in resultados:
            top, right, bottom, left = face_location
            
            # Desenhar retângulo na face
            color = COR_VERDE if match else COR_VERMELHO
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Adicionar texto com similaridade e nome se reconhecido
            if match:
                texto = f"{pessoa_info['nome']}: {similarity:.2f}"
            else:
                texto = f"Desconhecido: {similarity:.2f}"
            
            cv2.putText(frame, texto, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame, len(resultados) > 0
    
    def __del__(self):
        """Destrutor para garantir que o pool de threads seja encerrado corretamente"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False) 