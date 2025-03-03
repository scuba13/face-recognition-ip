#!/usr/bin/env python3
"""
Script para processar imagens de funcionários e cadastrar no MongoDB.
As imagens devem estar na pasta 'cadastro_funcionarios' com o formato: nome|id.jpg
Exemplo: joao_silva|12345.jpg
"""
import os
import face_recognition
import cv2
from face_detector.models.employee import Employee
from face_detector.utils.logger import log_info
from face_detector.utils.image_utils import melhorar_imagem

# Pasta fixa para as imagens
PASTA_CADASTRO = "cadastro_funcionarios"
MONGODB_URL = "mongodb://admin:admin123@localhost:27017/"

def processar_imagem(caminho_imagem):
    """Processa uma imagem e retorna o encoding facial"""
    # Carregar imagem
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    
    # Melhorar qualidade
    imagem = melhorar_imagem(imagem)
    
    # Converter BGR para RGB (face_recognition usa RGB)
    rgb_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    # Detectar faces
    face_locations = face_recognition.face_locations(rgb_imagem, model="hog")
    
    if not face_locations:
        raise ValueError(f"Nenhuma face detectada na imagem: {caminho_imagem}")
    
    if len(face_locations) > 1:
        raise ValueError(f"Múltiplas faces detectadas na imagem: {caminho_imagem}")
    
    # Gerar encoding
    face_encodings = face_recognition.face_encodings(rgb_imagem, face_locations, num_jitters=2)
    
    return face_encodings[0]

def processar_pasta_funcionarios():
    """Processa todas as imagens na pasta e cadastra no MongoDB"""
    # Criar pasta se não existir
    os.makedirs(PASTA_CADASTRO, exist_ok=True)
    
    # Verificar se existem imagens na pasta
    if not os.listdir(PASTA_CADASTRO):
        log_info(f"❌ Nenhuma imagem encontrada na pasta '{PASTA_CADASTRO}'")
        log_info(f"Adicione imagens no formato 'nome|id.jpg' (exemplo: joao_silva|12345.jpg)")
        return
    
    # Inicializar conexão com MongoDB
    employee_model = Employee(db_url=MONGODB_URL)
    
    # Criar índices necessários
    employee_model.create_indexes()
    
    # Processar cada imagem na pasta
    for arquivo in os.listdir(PASTA_CADASTRO):
        if not arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        try:
            # Extrair nome e ID do nome do arquivo
            nome_base = os.path.splitext(arquivo)[0]
            if '|' not in nome_base:
                log_info(f"Arquivo {arquivo} não segue o padrão nome|id. Ignorando...")
                continue
            
            nome, employee_id = nome_base.split('|')
            nome = nome.replace('_', ' ').title()
            
            # Caminho completo da imagem
            caminho_imagem = os.path.join(PASTA_CADASTRO, arquivo)
            
            # Processar imagem
            log_info(f"Processando imagem de {nome} (ID: {employee_id})...")
            face_encoding = processar_imagem(caminho_imagem)
            
            # Cadastrar no MongoDB
            employee_model.add_employee(
                employee_id=employee_id,
                name=nome,
                face_encoding=face_encoding.tolist()
            )
            
            log_info(f"✅ Funcionário {nome} cadastrado com sucesso!")
            
        except Exception as e:
            log_info(f"❌ Erro ao processar {arquivo}: {str(e)}")

if __name__ == "__main__":
    log_info("Iniciando cadastro de funcionários...")
    log_info(f"Usando pasta: {PASTA_CADASTRO}")
    log_info("Formato esperado das imagens: nome|id.jpg (exemplo: joao_silva|12345.jpg)")
    processar_pasta_funcionarios() 