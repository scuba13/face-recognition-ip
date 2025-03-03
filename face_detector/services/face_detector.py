"""
Serviço para detecção e reconhecimento facial.
Implementa processamento paralelo para melhor desempenho.
"""
import cv2
import face_recognition
from datetime import datetime
import concurrent.futures
import numpy as np
import os
from face_detector.config.settings import (
    FACE_SIMILARITY_THRESHOLD, MODELO_FACE, NUM_JITTERS,
    COR_VERDE, COR_VERMELHO, QUALIDADE_JPEG
)
from face_detector.utils.logger import log_face, log_captura
from face_detector.utils.image_utils import melhorar_imagem, salvar_imagem
from face_detector.models.employee import Employee

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
        self.employee_model = Employee()
    
    def detectar_faces(self, frame):
        """Detecta faces em um frame e retorna as localizações e encodings"""
        try:
            # Aplicar melhorias na imagem antes da detecção
            frame_melhorado = melhorar_imagem(frame)
            
            # Converter de BGR (OpenCV) para RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame_melhorado, cv2.COLOR_BGR2RGB)
            
            # Encontrar todas as faces no frame
            face_locations = face_recognition.face_locations(rgb_frame, 
                                                           model=self.modelo, 
                                                           number_of_times_to_upsample=2)
            
            # Logar quando encontrar faces
            if face_locations:
                log_face(f"✅ ENCONTRADAS {len(face_locations)} FACES")
                
                # Calcular os encodings das faces
                face_encodings = face_recognition.face_encodings(rgb_frame, 
                                                               face_locations, 
                                                               num_jitters=self.num_jitters)
                
                # Expandir um pouco a área da face para capturar melhor
                original_face_locations = []
                for (top, right, bottom, left) in face_locations:
                    height = bottom - top
                    width = right - left
                    
                    # Expandir em 20% para cada lado
                    top = max(0, int(top - height * 0.2))
                    bottom = int(bottom + height * 0.2)
                    left = max(0, int(left - width * 0.2))
                    right = int(right + width * 0.2)
                    
                    original_face_locations.append((top, right, bottom, left))
                
                return original_face_locations, face_encodings
            else:
                log_face("😕 Nenhuma face detectada no frame")
                return [], []
                
        except Exception as e:
            log_face(f"❌ ERRO ao detectar faces: {str(e)}")
            return [], []
    
    def _processar_face_individual(self, args):
        """
        Processa uma face individual (para execução paralela)
        
        Args:
            args: Tupla contendo (frame, face_location, face_encoding, index)
        
        Returns:
            Tupla com (face_location, match, similarity, index, pessoa_info)
        """
        frame, face_location, face_encoding, index = args
        
        try:
            # Buscar todos os funcionários do MongoDB
            funcionarios = self.employee_model.get_all_employees()
            
            if not funcionarios:
                log_face("⚠️ Nenhum funcionário cadastrado no banco de dados")
                return (face_location, False, 0, index, None)
            
            log_face(f"👥 Face {index+1}: Comparando com {len(funcionarios)} funcionários...")
            
            # Inicializar variáveis para o melhor match
            best_match = False
            best_similarity = 0
            best_funcionario = None
            
            # Comparar com todos os funcionários
            for funcionario in funcionarios:
                try:
                    # Verificar se o funcionário tem todos os campos necessários
                    if not all(key in funcionario for key in ['_id', 'employee_id', 'name', 'face_encoding']):
                        log_face(f"⚠️ Funcionário com dados incompletos: {funcionario}")
                        continue
                    
                    # O encoding já está como array no MongoDB, só precisamos converter para numpy array
                    known_encoding = np.array(funcionario['face_encoding'])
                    
                    # Calcular a distância entre os encodings (menor = mais similar)
                    face_distances = face_recognition.face_distance([known_encoding], face_encoding)
                    similarity = float(1 - face_distances[0])  # Converter distância para similaridade (0-1)
                    
                    log_face(f"🔄 Face {index+1} vs {funcionario['name']}: {similarity:.2f}")
                    
                    # Atualizar melhor match se encontrar um mais similar
                    if similarity > self.similarity_threshold:
                        best_match = True
                        best_similarity = similarity
                        best_funcionario = funcionario
                        break  # Encontrou um match bom o suficiente, pode parar
                    elif similarity > best_similarity:
                        best_similarity = similarity  # Atualizar melhor similaridade mesmo se abaixo do threshold
                except Exception as e:
                    log_face(f"❌ Erro ao comparar face {index+1} com funcionário {funcionario.get('name', 'desconhecido')}: {str(e)}")
                    continue
            
            # Logar resultado
            if best_match:
                log_face(f"✅ Face {index+1}: {best_funcionario['name']} RECONHECIDO! "
                      f"(Similaridade: {best_similarity:.2f})")
                
                # Registrar reconhecimento no MongoDB
                self.employee_model.register_recognition(best_funcionario['employee_id'], best_similarity)
            else:
                log_face(f"❓ Face {index+1}: DESCONHECIDO "
                      f"(Melhor similaridade: {best_similarity:.2f})")
            
            return (face_location, best_match, best_similarity, index, best_funcionario)
            
        except Exception as e:
            log_face(f"❌ ERRO ao processar face {index+1}: {str(e)}")
            return (face_location, False, 0, index, None)
    
    def processar_faces_no_frame(self, frame):
        """Processa faces em um único frame usando processamento paralelo"""
        # Detectar faces
        face_locations, face_encodings = self.detectar_faces(frame)
        
        # Se não encontrou faces, retornar o frame original
        if not face_locations:
            return frame, False
        
        # Logar quantidade de faces detectadas
        log_face(f"Detectadas {len(face_locations)} faces na imagem")
        
        # Preparar argumentos para processamento paralelo
        args_list = [
            (frame.copy(), face_location, face_encoding, i)
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
        resultados.sort(key=lambda x: x[3])
        
        # Desenhar resultados no frame
        for face_location, match, similarity, _, funcionario in resultados:
            top, right, bottom, left = face_location
            
            # Desenhar retângulo na face
            color = COR_VERDE if match else COR_VERMELHO
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Adicionar texto com similaridade e nome se reconhecido
            if match and funcionario:
                texto = f"{funcionario['name']}: {similarity:.2f}"
            else:
                texto = f"Desconhecido: {similarity:.2f}"
            
            cv2.putText(frame, texto, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame, len(resultados) > 0
    
    def __del__(self):
        """Destrutor para garantir que o pool de threads seja encerrado corretamente"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False) 