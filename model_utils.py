import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
import tempfile
import os
import time
import collections

# --------------------------------------------------------------------------
# 1. ARQUITETURA DO MODELO (Deve ser idêntica à do treino)
# --------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        x = self.final_conv(x)
        return torch.sigmoid(x)

# --------------------------------------------------------------------------
# 2. FUNÇÃO DE CONFIGURAÇÃO (Carrega modelo e thresholds)
# --------------------------------------------------------------------------
@st.cache_resource
def setup_model_and_config():
    """
    Carrega o modelo U-Net e os thresholds a partir dos arquivos locais.
    Usa o cache do Streamlit para executar apenas uma vez.
    """
    DEVICE = torch.device('cpu')
    
    # Caminhos locais dos arquivos (Git LFS baixa automaticamente no Streamlit Cloud)
    MODEL_PATH = "models/bottle_unet_best.pth"
    CONFIG_PATH = "models/bottle_unet_config.json"
    
    # Carrega a configuração
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar configuração: {e}")
        # Configuração padrão como fallback
        config = {
            "classification_threshold": 0.000205,
            "pixel_visualization_threshold": 20
        }
    
    # Carrega o modelo
    try:
        # Verifica se o arquivo existe
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"Arquivo do modelo não encontrado: {MODEL_PATH}")
        
        # Verifica o tamanho do arquivo
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 1000:
            raise Exception(f"Arquivo do modelo muito pequeno ({file_size} bytes), possível problema com Git LFS")
        
        # Carrega o modelo
        model = UNet().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, config
    
    return model, config

# --------------------------------------------------------------------------
# 3. FUNÇÃO DE PREDIÇÃO
# --------------------------------------------------------------------------
def predict(model, config, image: Image.Image):
    """
    Executa a predição em uma única imagem e retorna os resultados.
    """
    DEVICE = torch.device('cpu')
    
    # Pega os thresholds do arquivo de configuração
    CLASS_THRESHOLD = config['classification_threshold']
    PIXEL_THRESHOLD = config['pixel_visualization_threshold']

    # Prepara a imagem
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Executa a predição
    with torch.no_grad():
        reconstruction = model(image_tensor)
        error = torch.nn.functional.mse_loss(reconstruction, image_tensor)

    # Classifica a imagem inteira
    prediction_text = "Anomalia Detectada" if error.item() > CLASS_THRESHOLD else "Normal"

    # Gera o mapa de anomalia e a máscara
    anomaly_map = torch.mean(torch.abs(image_tensor - reconstruction), dim=1, keepdim=True)
    anomaly_map_scaled = (anomaly_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(anomaly_map_scaled, PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Retorna um dicionário com todos os resultados
    results = {
        "prediction": prediction_text,
        "error": error.item(),
        "original_image": image,
        "reconstructed_image": reconstruction,
        "anomaly_map_scaled": anomaly_map_scaled,
        "binary_mask": binary_mask,
        "pixel_threshold": PIXEL_THRESHOLD
    }
    return results

# --------------------------------------------------------------------------
# 4. FUNÇÕES DE VISUALIZAÇÃO
# --------------------------------------------------------------------------
def get_anomaly_map_image(results: dict):
    """
    Retorna a imagem do mapa de anomalia.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(results['anomaly_map_scaled'], cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_binary_mask_image(results: dict):
    """
    Retorna a imagem da máscara binária.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(results['binary_mask'], cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_heatmap_image(results: dict):
    """
    Retorna a imagem do mapa de calor.
    """
    original_np = np.array(results['original_image'].resize((256, 256)))
    heatmap = cv2.applyColorMap(results['anomaly_map_scaled'], cv2.COLORMAP_JET)
    
    # Sobrepõe o heatmap na imagem original
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout()
    return fig

def display_bounding_box(results: dict, config: dict):
    """
    Exibe a imagem original com bounding box minimalista destacando a área da anomalia.
    Gera uma máscara interna com threshold mais sensível para um bounding box mais abrangente.
    """
    # Redimensiona a imagem original para 256x256
    original_resized = results['original_image'].resize((256, 256))
    original_np = np.array(original_resized)
    
    # --- Lógica específica para o Bounding Box ---
    # Usa um threshold do arquivo de config para gerar uma máscara que captura a anomalia inteira.
    bounding_box_threshold = config.get("bounding_box_threshold", 1.5) # Fallback para 1.5
    _, sensitive_mask = cv2.threshold(
        results['anomaly_map_scaled'], 
        bounding_box_threshold, 
        255, 
        cv2.THRESH_BINARY
    )

    # Aplica dilatação para conectar regiões da anomalia, tornando o bounding box mais abrangente
    dilation_iterations = config.get("dilation_iterations", 2) # Fallback para 2
    kernel = np.ones((3,3), np.uint8)
    sensitive_mask = cv2.dilate(sensitive_mask, kernel, iterations = dilation_iterations)
    
    # Encontra contornos na máscara binária SENSÍVEL
    contours, _ = cv2.findContours(sensitive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenha bounding boxes nos contornos encontrados
    result_image = original_np.copy()
    
    if len(contours) > 0:
        # Encontra o maior contorno (área da anomalia)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcula o bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Adiciona margem ao redor da anomalia (15% do tamanho em cada direção)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        
        # Ajusta as coordenadas com margem
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(256, x + w + margin_x)
        y_end = min(256, y + h + margin_y)
        
        # Desenha o bounding box minimalista em ciano (cor do tema)
        cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 194, 255), 2)
        
        # Adiciona texto pequeno e discreto
        text = "Anomalia"
        font_scale = 0.4
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Texto simples sem fundo
        cv2.putText(result_image, text, (x_start, y_start - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 194, 255), font_thickness)
    
    return result_image

