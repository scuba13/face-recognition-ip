"""
Serviço para detecção e reconhecimento facial.
"""
import cv2
import face_recognition
from datetime import datetime
from face_detector.config.settings import (
    FACE_SIMILARITY_THRESHOLD, MODELO_FACE, NUM_JITTERS,
    COR_VERDE, COR_VERMELHO, QUALIDADE_JPEG
)
from face_detector.utils.logger import log_face, log_captura
from face_detector.utils.image_utils import melhorar_imagem, salvar_imagem

class FaceDetector:
    """Classe para detecção e reconhecimento facial"""
    
    def __init__(self, similarity_threshold=None, modelo=None, num_jitters=None):
        """Inicializa o detector facial com os parâmetros especificados"""
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else FACE_SIMILARITY_THRESHOLD
        self.modelo = modelo if modelo is not None else MODELO_FACE
        self.num_jitters = num_jitters if num_jitters is not None else NUM_JITTERS
    
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
        """Processa faces em um único frame"""
        # Detectar faces
        face_locations, face_encodings = self.detectar_faces(frame)
        
        # Se não encontrou faces, retornar o frame original
        if not face_locations:
            return frame, False
        
        # Logar quantidade de faces detectadas
        log_face(f"Detectadas {len(face_locations)} faces na imagem")
        
        # Comparar com encoding conhecido
        resultados = self.comparar_faces(face_encodings, pessoa_conhecida_encoding)
        
        # Flag para indicar se alguma face foi encontrada
        face_encontrada = False
        
        # Processar resultados
        for j, ((top, right, bottom, left), (match, similarity)) in enumerate(
                zip(face_locations, resultados)):
            
            face_encontrada = True
            
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
            
            # Sempre salvar a face quando processada (apenas ocorre após movimento)
            face_filename = self.salvar_face(
                frame, (top, right, bottom, left), match, similarity, pessoa_info if match else None)
            
            log_captura(f"Face salva: {face_filename} - Motivo: Movimento detectado")
            
            if match:
                log_face(f"👤 Face {j+1}: {pessoa_info['nome']} RECONHECIDO "
                      f"(Similaridade: {similarity:.2f})")
            else:
                log_face(f"👤 Face {j+1}: PESSOA DESCONHECIDA "
                      f"(Similaridade: {similarity:.2f})")
        
        return frame, face_encontrada 