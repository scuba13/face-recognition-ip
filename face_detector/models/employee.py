"""
Modelo para gerenciamento de funcionários no MongoDB.
"""
from datetime import datetime
import numpy as np
from pymongo import MongoClient
from face_detector.config.settings import (
    MONGODB_HOST, MONGODB_PORT, MONGODB_DATABASE,
    MONGODB_USERNAME, MONGODB_PASSWORD, MONGODB_AUTH_SOURCE
)
from face_detector.utils.logger import log_info, log_error

class Employee:
    """Classe para gerenciar funcionários no MongoDB"""
    
    def __init__(self):
        """Inicializa a conexão com o MongoDB"""
        try:
            # Conectar ao MongoDB
            self.client = MongoClient(
                host=MONGODB_HOST,
                port=MONGODB_PORT,
                username=MONGODB_USERNAME,
                password=MONGODB_PASSWORD,
                authSource=MONGODB_AUTH_SOURCE
            )
            
            # Selecionar banco e coleções
            self.db = self.client[MONGODB_DATABASE]
            self.employees = self.db.employees
            self.recognitions = self.db.recognitions
            
            log_info("✅ Conexão com MongoDB estabelecida com sucesso")
            
        except Exception as e:
            log_error(f"❌ Erro ao conectar ao MongoDB: {str(e)}")
            raise
    
    def get_all_employees(self):
        """Retorna todos os funcionários cadastrados"""
        try:
            # Buscar funcionários que têm encoding facial cadastrado
            funcionarios = list(self.employees.find({
                "face_encoding": {"$exists": True},
                "employee_id": {"$exists": True},
                "name": {"$exists": True}
            }))
            log_info(f"📋 {len(funcionarios)} funcionários encontrados no banco")
            return funcionarios
            
        except Exception as e:
            log_error(f"❌ Erro ao buscar funcionários: {str(e)}")
            return []
    
    def register_recognition(self, employee_id, similarity):
        """
        Registra um reconhecimento facial
        
        Args:
            employee_id: ID do funcionário reconhecido
            similarity: Valor de similaridade do reconhecimento (0-1)
        """
        try:
            # Criar documento de reconhecimento
            recognition = {
                "employee_id": employee_id,
                "similarity": float(similarity),  # Garantir que é float
                "timestamp": datetime.now(),
                "status": "success"
            }
            
            # Inserir no MongoDB
            result = self.recognitions.insert_one(recognition)
            
            if result.inserted_id:
                log_info(f"✅ Reconhecimento registrado: {employee_id} ({similarity:.2f})")
            else:
                log_error("❌ Falha ao registrar reconhecimento")
                
        except Exception as e:
            log_error(f"❌ Erro ao registrar reconhecimento: {str(e)}")
    
    def __del__(self):
        """Fecha a conexão com o MongoDB ao destruir o objeto"""
        if hasattr(self, 'client'):
            self.client.close() 