"""
Configurações do sistema de detecção de faces.
Contém todas as constantes e parâmetros utilizados pelo sistema.
"""
import numpy as np

# URL RTSP fixa que sabemos que funciona
RTSP_URL = "rtsp://192.168.0.136:554/0/av0"

# Configurações de detecção de movimento
MOVIMENTO_THRESHOLD = 15000  # Limiar de detecção de movimento (quanto menor, mais sensível)
AREA_MINIMA_CONTORNO = 5000  # Área mínima de contorno para considerar como movimento real
FRAMES_APOS_MOVIMENTO = 5    # Número de frames para processar após detectar movimento
INTERVALO_MINIMO_MOVIMENTO = 0.5  # Intervalo mínimo entre detecções de movimento (segundos)

# Configurações de reconhecimento facial
FACE_SIMILARITY_THRESHOLD = 0.6  # Limiar de similaridade (quanto menor, mais restritivo)
FACE_DETECTION_INTERVAL = 30     # Intervalo para detecção de faces (a cada quantos frames)
INTERVALO_MINIMO_FACE = 0.3      # Intervalo mínimo entre processamentos de face (segundos)
TEMPO_EXPIRACAO_FACE = 3.0       # Tempo para considerar uma face como "nova" novamente (segundos)
MODELO_FACE = "hog"              # Modelo para detecção facial (hog ou cnn)
NUM_JITTERS = 3                  # Número de vezes para amostrar a face durante o encoding

# Configurações de captura e processamento
BUFFER_SIZE_CAPTURA = 10         # Tamanho do buffer de frames para captura
TAXA_FPS_CAPTURA = 30            # Taxa de FPS alvo para captura
TAXA_FPS_UI = 30                 # Taxa de FPS alvo para interface gráfica

# Configurações de qualidade de imagem
RESOLUCAO_CAPTURA = (1920, 1080)  # HD para melhor desempenho
QUALIDADE_JPEG = 95               # Qualidade de salvamento (0-100)
APLICAR_MELHORIA_IMAGEM = True    # Aplicar melhorias de imagem
USAR_TONS_CINZA = True            # Usar tons de cinza para comparação facial

# Configurações de debug
MODO_DEBUG = True
MAX_FRAMES_SEM_DETECCAO = 100

# Encoding real da pessoa para comparação
PESSOA_CONHECIDA_ENCODING = np.array([
    -0.05037004500627518, 0.08081860095262527, 0.013230549171566963, -0.058972179889678955,
    0.031775400042533875, -0.04935765266418457, 0.045750122517347336, -0.10653585940599442,
    0.1998727023601532, -0.04209068417549133, 0.11739099025726318, 0.05334269255399704,
    -0.23903197050094604, -0.014060728251934052, 0.03702852874994278, 0.017634190618991852,
    -0.05829431861639023, -0.10795681923627853, -0.17121778428554535, -0.07984332740306854,
    -0.019725970923900604, 0.0061602843925356865, 0.011421307921409607, 0.10115454345941544,
    -0.14343732595443726, -0.22639824450016022, -0.06681777536869049, -0.1025376170873642,
    0.008905860595405102, -0.13407540321350098, 0.06989164650440216, 0.002723999321460724,
    -0.1294497847557068, 0.010295376181602478, -0.06715257465839386, 0.03900441154837608,
    -0.11792229115962982, -0.08725777268409729, 0.259025514125824, 0.07957687228918076,
    -0.17149513959884644, 0.0065041184425354, -0.04273233190178871, 0.33151888847351074,
    0.17277418076992035, -0.08130183070898056, 0.04827491194009781, -0.07910497486591339,
    0.15448616445064545, -0.2834199070930481, 0.041074998676776886, 0.15529049932956696,
    0.1031646654009819, 0.04282759130001068, 0.14144113659858704, -0.13453440368175507,
    0.0271356999874115, 0.09792690724134445, -0.27022039890289307, 0.1485123336315155,
    0.04481695592403412, -0.04401165246963501, -0.08844450861215591, -0.0836171805858612,
    0.10269097983837128, 0.14938758313655853, -0.11636527627706528, -0.08060094714164734,
    0.10706610977649689, -0.22144050896167755, 0.0020930320024490356, 0.1379607766866684,
    -0.05111892521381378, -0.27859172224998474, -0.18193170428276062, 0.048309117555618286,
    0.37621551752090454, 0.23665854334831238, -0.1806277632713318, 0.01589256525039673,
    -0.13478127121925354, -0.04998968169093132, 0.1227211207151413, 0.005320116877555847,
    -0.08633564412593842, 0.0067610591650009155, -0.07683606445789337, 0.06881460547447205,
    0.1791447401046753, 0.015951231122016907, -0.04933106154203415, 0.2073071300983429,
    0.04152250289916992, -0.03778393566608429, -0.00926598347723484, -0.0075337886810302734,
    -0.12055684626102448, 0.028685832396149635, -0.12823981046676636, 0.008179515600204468,
    0.03266230225563049, -0.11627024412155151, 0.0633450448513031, -0.02620960772037506,
    -0.16261953115463257, 0.2309507131576538, 0.11812739074230194, -0.04646923020482063,
    0.0345793291926384, 0.047732606530189514, -0.18671737611293793, -0.02541157603263855,
    0.20789499580860138, -0.2948143482208252, 0.18943622708320618, 0.13059723377227783,
    0.005113288294523954, 0.11006122827529907, 0.023252852261066437, 0.08387812972068787,
    -0.02853265404701233, -0.01336362212896347, -0.08350375294685364, -0.023510165512561798,
    0.01866171881556511, 0.03550122678279877, 0.06424954533576965, -0.006630636751651764
], dtype=np.float64)

# Informações da pessoa para exibição
PESSOA_INFO = {
    "id": "1233445",
    "nome": "Eduardo Nascimento"
}

# Cores para visualização (BGR)
COR_VERDE = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AZUL = (255, 0, 0)
COR_AMARELO = (0, 255, 255) 