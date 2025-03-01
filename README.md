# Sistema de Detecção Facial com Detecção de Movimento

Sistema de vigilância inteligente que combina detecção de movimento e reconhecimento facial para monitoramento de segurança.

## Características

- **Detecção de Movimento**: Identifica movimentos em tempo real usando algoritmos de processamento de imagem
- **Reconhecimento Facial**: Detecta e reconhece faces após a detecção de movimento
- **Captura Assíncrona**: Implementação assíncrona para melhor desempenho com threads separadas para captura e processamento
- **Suporte a RTSP**: Compatível com câmeras IP via protocolo RTSP
- **Armazenamento de Imagens**: Salva automaticamente frames com movimento detectado e faces identificadas
- **Reconexão Automática**: Sistema robusto de reconexão para streams RTSP instáveis
- **Interface Visual**: Exibe informações em tempo real sobre detecções e status do sistema

## Estrutura do Projeto

```
face_detector/
├── config/             # Configurações do sistema
├── controllers/        # Controladores de alto nível
├── core/               # Funcionalidades principais
├── models/             # Modelos de dados
├── services/           # Serviços específicos
└── utils/              # Utilitários e funções auxiliares
```

## Requisitos

- Python 3.8+
- OpenCV
- NumPy
- face_recognition

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/sistema-deteccao-facial.git
cd sistema-deteccao-facial
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Configuração

Edite o arquivo `face_detector/config/settings.py` para configurar:

- URL RTSP da câmera
- Parâmetros de detecção de movimento (threshold, área mínima)
- Configurações de reconhecimento facial
- Resolução de captura

## Uso

Execute o sistema com:

```bash
python run.py
```

Para usar uma câmera local em vez de RTSP:

```bash
python run.py --camera 0
```

## Estrutura de Pastas Criada

O sistema cria automaticamente a seguinte estrutura de pastas para organizar as capturas:

```
capturas/
├── movimento/          # Frames com movimento detectado
├── faces/              # Faces recortadas
│   ├── match/          # Faces reconhecidas
│   └── desconhecido/   # Faces não reconhecidas
├── frames/             # Frames completos com anotações
└── manual/             # Capturas manuais
```

## Controles

- **ESC**: Sair do programa

## Licença

MIT 