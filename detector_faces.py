import cv2
import os
import time
import numpy as np
from datetime import datetime
import face_recognition
import pickle

# URL RTSP fixa que sabemos que funciona
RTSP_URL = "rtsp://192.168.0.136:554/0/av0"

# Limiar de detecção de movimento (quanto menor, mais sensível)
MOVIMENTO_THRESHOLD = 20000  # Aumentado de 10000 para 20000 para reduzir sensibilidade

# Área mínima de contorno para considerar como movimento real
AREA_MINIMA_CONTORNO = 5000  # Aumentado de 2000 para 5000

# Limiar de similaridade para reconhecimento facial (quanto menor, mais restritivo)
FACE_SIMILARITY_THRESHOLD = 0.6  # Aumentado de 0.6 para 0.55 para ser mais restritivo

# Intervalo para detecção de faces (a cada quantos frames)
FACE_DETECTION_INTERVAL = 30  # Aumentado de 10 para 30 para reduzir frequência

# Intervalo mínimo entre salvamentos de faces (em segundos)
# Para evitar salvar muitas imagens da mesma face em sequência
INTERVALO_SALVAMENTO_FACES = 1.0

# Número de frames para processar após detectar movimento
FRAMES_APOS_MOVIMENTO = 5

# Contador de frames sem detecção para debug
MAX_FRAMES_SEM_DETECCAO = 100

# Configurações de qualidade de imagem
RESOLUCAO_CAPTURA = (1920, 1080)  # HD para melhor desempenho (original: 2304x1296)
QUALIDADE_JPEG = 95              # Qualidade de salvamento (0-100)
APLICAR_MELHORIA_IMAGEM = True   # Aplicar melhorias de imagem
USAR_TONS_CINZA = True           # Usar tons de cinza para comparação facial
MODELO_FACE = "hog"              # Modelo para detecção facial (hog ou cnn)
NUM_JITTERS = 3                  # Número de vezes para amostrar a face durante o encoding

# Modo de depuração para mostrar mais informações
MODO_DEBUG = True

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

# Cores para visualização
COR_VERDE = (0, 255, 0)  # BGR
COR_VERMELHO = (0, 0, 255)
COR_AZUL = (255, 0, 0)
COR_AMARELO = (0, 255, 255)

def log_info(mensagem):
    """Exibe log de informação com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO] [{timestamp}] {mensagem}")

def log_movimento(mensagem):
    """Exibe log de detecção de movimento com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[MOVIMENTO] [{timestamp}] {mensagem}")

def log_face(mensagem):
    """Exibe log de detecção de face com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[FACE] [{timestamp}] {mensagem}")

def log_captura(mensagem):
    """Exibe log específico para capturas com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[CAPTURA] [{timestamp}] {mensagem}")

def log_debug(mensagem):
    """Exibe log de depuração apenas se o modo debug estiver ativado"""
    if MODO_DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[DEBUG] [{timestamp}] {mensagem}")

def log_processamento(mensagem):
    """Exibe log de processamento com timestamp"""
    # Função modificada para não exibir nada
    pass

def criar_estrutura_pastas():
    """Cria a estrutura de pastas para organizar as imagens"""
    # Pasta principal para capturas
    os.makedirs("capturas", exist_ok=True)
    
    # Subpastas para organizar por tipo
    os.makedirs("capturas/movimento", exist_ok=True)  # Frames com movimento
    os.makedirs("capturas/faces", exist_ok=True)      # Faces recortadas
    os.makedirs("capturas/faces/match", exist_ok=True)      # Faces reconhecidas
    os.makedirs("capturas/faces/desconhecido", exist_ok=True)  # Faces desconhecidas
    os.makedirs("capturas/frames", exist_ok=True)     # Frames completos com anotações
    os.makedirs("capturas/manual", exist_ok=True)     # Capturas manuais
    
    # Pasta para encodings
    os.makedirs("encodings", exist_ok=True)
    
    log_info("Estrutura de pastas criada com sucesso")

def salvar_encoding_teste():
    """Salva o encoding real para comparação"""
    os.makedirs("encodings", exist_ok=True)
    
    # Salvar o encoding real
    with open("encodings/eduardo_nascimento.pickle", "wb") as f:
        pickle.dump(PESSOA_CONHECIDA_ENCODING, f)
    
    log_info(f"Encoding de {PESSOA_INFO['nome']} salvo em 'encodings/eduardo_nascimento.pickle'")

def carregar_encoding_teste():
    """Carrega o encoding real"""
    try:
        with open("encodings/eduardo_nascimento.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log_info(f"Arquivo de encoding não encontrado. Criando um novo para {PESSOA_INFO['nome']}...")
        salvar_encoding_teste()
        return PESSOA_CONHECIDA_ENCODING

def configurar_stream(rtsp_url):
    """Configura e abre o stream RTSP"""
    log_info(f"Abrindo stream RTSP: {rtsp_url}")
    
    # Configurações avançadas para o OpenCV - otimizadas para HEVC
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|fflags;discardcorrupt|stimeout;10000000|max_delay;500000|reorder_queue_size;0|strict;experimental"
    
    # Abrir stream com configurações otimizadas
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Configurações adicionais para melhorar a estabilidade
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer para menor latência
    
    # Tentar configurar resolução mais alta
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCAO_CAPTURA[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCAO_CAPTURA[1])
    
    # Tentar configurar para usar hardware acceleration se disponível
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    except:
        pass  # Ignorar se não suportado
    
    if not cap.isOpened():
        log_info("Erro ao abrir o stream RTSP.")
        return None
    
    # Verificar resolução real obtida
    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_info(f"Stream RTSP aberto com resolução: {largura_real}x{altura_real}")
    
    return cap

def melhorar_imagem(imagem):
    """Aplica técnicas de processamento para melhorar a qualidade da imagem"""
    if not APLICAR_MELHORIA_IMAGEM:
        return imagem
    
    # Converter para escala de cinza para processamento
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar equalização de histograma adaptativa (CLAHE) para melhorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Se estiver usando tons de cinza para reconhecimento, retornar a imagem em tons de cinza
    if USAR_TONS_CINZA:
        # Converter de volta para BGR (3 canais) para compatibilidade com o resto do código
        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Converter de volta para BGR
    imagem_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    
    # Misturar com a imagem original para manter cores (70% original, 30% equalizada)
    imagem_melhorada = cv2.addWeighted(imagem, 0.7, imagem_eq, 0.3, 0)
    
    # Aplicar sharpening para melhorar detalhes
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    imagem_melhorada = cv2.filter2D(imagem_melhorada, -1, kernel_sharpen)
    
    # Reduzir ruído mantendo bordas
    imagem_melhorada = cv2.bilateralFilter(imagem_melhorada, 9, 75, 75)
    
    return imagem_melhorada

def detectar_movimento(frame1, frame2):
    """Detecta movimento entre dois frames consecutivos"""
    # Converter para escala de cinza
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reduzir ruído
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Calcular diferença absoluta entre os frames
    frame_diff = cv2.absdiff(gray1, gray2)
    
    # Aplicar threshold para destacar áreas com movimento
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Dilatar o threshold para preencher buracos
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    movimento_detectado = False
    movimento_area = 0
    
    # Verificar se há contornos significativos
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > AREA_MINIMA_CONTORNO:  # Filtrar contornos pequenos (ruído)
            movimento_area += area
            # Desenhar retângulo ao redor do movimento
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), COR_VERDE, 2)
    
    # Verificar se a área total de movimento é significativa
    if movimento_area > MOVIMENTO_THRESHOLD:
        movimento_detectado = True
        # Adicionar texto indicando movimento
        cv2.putText(frame1, f"Movimento: {movimento_area}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COR_VERMELHO, 2)
    
    return movimento_detectado, movimento_area, frame1

def adicionar_info_tela(frame):
    """Adiciona informações na tela"""
    altura, largura = frame.shape[:2]
    
    # Adicionar timestamp
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, timestamp, (10, altura - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 1)
    
    # Adicionar informações da pessoa de referência
    cv2.putText(frame, f"Ref: {PESSOA_INFO['nome']}", (largura - 300, altura - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AZUL, 1)
    
    # Adicionar instruções
    cv2.putText(frame, "ESC: Sair", (largura - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_AMARELO, 1)

def detectar_faces(frame):
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
                                                    model=MODELO_FACE, 
                                                    number_of_times_to_upsample=1)
    
    if face_locations:
        log_face(f"✅ ENCONTRADAS {len(face_locations)} FACES")
    
    # Calcular os encodings das faces com mais precisão
    face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                    face_locations, 
                                                    num_jitters=NUM_JITTERS)
    
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

def comparar_faces(face_encodings, pessoa_conhecida_encoding):
    """Compara os encodings das faces detectadas com o encoding conhecido"""
    resultados = []
    
    for face_encoding in face_encodings:
        # Calcular a distância entre os encodings (menor = mais similar)
        face_distances = face_recognition.face_distance([pessoa_conhecida_encoding], face_encoding)
        
        # Verificar se a face é similar o suficiente
        match = face_distances[0] <= FACE_SIMILARITY_THRESHOLD
        similarity = 1 - face_distances[0]  # Converter distância para similaridade (0-1)
        
        resultados.append((match, similarity))
    
    return resultados

def salvar_face(frame, face_location, match, similarity, pessoa_info=None):
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
    cv2.imwrite(filename, face_img, [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
    
    # Salvar também o frame completo com a anotação
    frame_filename = f"capturas/frames/frame_{status}_{timestamp}.jpg"
    cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
    
    return filename

def processar_faces_no_frame(frame, pessoa_conhecida_encoding):
    """Processa faces em um único frame"""
    # Detectar faces
    face_locations, face_encodings = detectar_faces(frame)
    
    # Se não encontrou faces, retornar o frame original
    if not face_locations:
        return frame, False
    
    # Logar quantidade de faces detectadas
    log_face(f"Detectadas {len(face_locations)} faces na imagem")
    
    # Comparar com encoding conhecido
    resultados = comparar_faces(face_encodings, pessoa_conhecida_encoding)
    
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
            texto = f"{PESSOA_INFO['nome']}: {similarity:.2f}"
        else:
            texto = f"Desconhecido: {similarity:.2f}"
        
        cv2.putText(frame, texto, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Sempre salvar a face quando processada (apenas ocorre após movimento)
        face_filename = salvar_face(
            frame, (top, right, bottom, left), match, similarity, PESSOA_INFO if match else None)
        
        log_captura(f"Face salva: {face_filename} - Motivo: Movimento detectado")
        
        if match:
            log_face(f"👤 Face {j+1}: {PESSOA_INFO['nome']} RECONHECIDO "
                  f"(Similaridade: {similarity:.2f})")
        else:
            log_face(f"👤 Face {j+1}: PESSOA DESCONHECIDA "
                  f"(Similaridade: {similarity:.2f})")
    
    return frame, face_encontrada

def processar_stream():
    """Processa o stream RTSP, detecta movimento e faces continuamente"""
    # Configurar stream
    cap = configurar_stream(RTSP_URL)
    if cap is None:
        return
    
    # Criar estrutura de pastas
    criar_estrutura_pastas()
    
    # Carregar encoding real
    pessoa_conhecida_encoding = carregar_encoding_teste()
    
    log_info(f"Pessoa de referência: {PESSOA_INFO['nome']} (ID: {PESSOA_INFO['id']})")
    log_info("Controles: ESC = Sair")
    log_info(f"Detecção baseada em movimento: {FRAMES_APOS_MOVIMENTO} frames após movimento")
    log_info(f"Limiar de movimento: {MOVIMENTO_THRESHOLD} (área mínima: {AREA_MINIMA_CONTORNO})")
    log_info(f"Limiar de similaridade: {FACE_SIMILARITY_THRESHOLD} (menor = mais restritivo)")
    log_info(f"Processando e salvando faces APENAS após detecção de movimento")
    log_info(f"Qualidade de imagem: Resolução {RESOLUCAO_CAPTURA}, JPEG {QUALIDADE_JPEG}%, Melhorias: {APLICAR_MELHORIA_IMAGEM}")
    log_info(f"Usando tons de cinza para comparação: {USAR_TONS_CINZA}, Modelo: {MODELO_FACE}, Jitters: {NUM_JITTERS}")
    log_info(f"Modo de depuração: {MODO_DEBUG}")
    
    # Criar janela com tamanho ajustável
    cv2.namedWindow("Detector de Faces por Movimento", cv2.WINDOW_NORMAL)
    
    # Ler o primeiro frame
    ret, frame_anterior = cap.read()
    if not ret:
        log_info("Erro ao ler o primeiro frame.")
        cap.release()
        return
    
    # Redimensionar janela para um tamanho razoável
    altura, largura = frame_anterior.shape[:2]
    cv2.resizeWindow("Detector de Faces por Movimento", largura, altura)
    
    # Variáveis para controle
    contador_frames = 0
    
    # Variáveis para FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Variáveis para controle de movimento
    frames_restantes_apos_movimento = 0
    
    # Contador para debug de detecção
    frames_sem_deteccao = 0
    
    log_info("Iniciando processamento do stream...")
    
    while True:
        # Ler o próximo frame
        ret, frame_atual = cap.read()
        if not ret:
            log_info("Erro ao ler frame. Tentando reconectar...")
            cap.release()
            time.sleep(1)
            cap = configurar_stream(RTSP_URL)
            if cap is None:
                break
            ret, frame_anterior = cap.read()
            if not ret:
                continue
            continue
        
        # Calcular FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Incrementar contador de frames
        contador_frames += 1
        
        # Criar cópia do frame para processamento
        frame_processado = frame_atual.copy()
        
        # Detectar movimento para visualização
        movimento_detectado, movimento_area, frame_com_movimento = detectar_movimento(
            frame_atual.copy(), frame_anterior.copy())
        
        # Atualizar frame anterior para próxima detecção de movimento
        frame_anterior = frame_atual.copy()
        
        # Se detectou movimento, salvar frame e configurar para processar os próximos frames
        if movimento_detectado:
            frames_sem_deteccao = 0  # Resetar contador de frames sem detecção
            
            # Salvar frame com movimento
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            movimento_filename = f"capturas/movimento/movimento_{movimento_area:.0f}_{timestamp}.jpg"
            cv2.imwrite(movimento_filename, frame_com_movimento)
            
            log_movimento(f"Movimento detectado (área: {movimento_area:.0f}) - Limiar: {MOVIMENTO_THRESHOLD}")
            log_captura(f"Frame de movimento salvo: {movimento_filename}")
            
            # Configurar para processar os próximos frames
            frames_restantes_apos_movimento = FRAMES_APOS_MOVIMENTO
            log_debug(f"Configurado para processar os próximos {FRAMES_APOS_MOVIMENTO} frames")
        else:
            frames_sem_deteccao += 1
            if frames_sem_deteccao >= MAX_FRAMES_SEM_DETECCAO:
                frames_sem_deteccao = 0
                log_info(f"Nenhum movimento detectado nos últimos {MAX_FRAMES_SEM_DETECCAO} frames")
        
        # Determinar se deve processar faces neste frame
        deve_processar_faces = False
        motivo_processamento = ""
        
        # Apenas processar faces após detecção de movimento
        if frames_restantes_apos_movimento > 0:
            deve_processar_faces = True
            motivo_processamento = f"Frame após movimento ({frames_restantes_apos_movimento} restantes)"
        
        # Processar faces se necessário
        if deve_processar_faces:
            log_debug(f"Processando faces - Motivo: {motivo_processamento}")
            
            # Processar faces no frame atual
            frame_processado, face_encontrada = processar_faces_no_frame(
                frame_processado, pessoa_conhecida_encoding)
            
            # Se encontrou face durante o processamento após movimento, registrar
            if face_encontrada and frames_restantes_apos_movimento > 0:
                log_face(f"Face detectada após movimento! Frames restantes: {frames_restantes_apos_movimento}")
            
            # Decrementar contador de frames após movimento
            if frames_restantes_apos_movimento > 0:
                frames_restantes_apos_movimento -= 1
        
        # Adicionar informações na tela
        adicionar_info_tela(frame_processado)
        
        # Adicionar FPS e informações de movimento
        cv2.putText(frame_processado, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
        
        if movimento_detectado:
            cv2.putText(frame_processado, f"Movimento: {movimento_area}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERMELHO, 2)
        
        # Mostrar frame processado
        cv2.imshow("Detector de Faces por Movimento", frame_processado)
        
        # Capturar tecla
        key = cv2.waitKey(1) & 0xFF
        
        # ESC para sair
        if key == 27:
            log_info("Tecla ESC pressionada. Encerrando...")
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    log_info("Stream fechado.")

if __name__ == "__main__":
    print("=" * 50)
    print("=== Detector de Faces por Movimento ===")
    print("=" * 50)
    processar_stream() 