import os
import io
import glob
import json
import time
import platform
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import cv2
from streamlit_image_select import image_select
import torch
import tempfile
import requests
from pathlib import Path
import matplotlib.pyplot as plt

# Importar funções do model_utils
from model_utils import setup_model_and_config, predict, get_anomaly_map_image, get_binary_mask_image, get_heatmap_image, display_bounding_box

# Base do app para construir caminhos robustos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "imagem")
TRAINING_HISTORY_DIR = os.path.join(BASE_DIR, "training_history")

st.set_page_config(
    page_title="Anomaly Detection • U-Net",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo premium ultra-profissional
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  /* Cores principais - Paleta quente */
  --primary: #FF6B35;
  --primary-dark: #E55A2B;
  --accent: #F7931E;
  --accent-dark: #E8851A;
  --success: #06D6A0;
  --success-dark: #05B888;
  --danger: #FF3366;
  --danger-dark: #E52D5A;
  --warning: #FFB627;
  --warning-dark: #E6A323;
  
  /* Backgrounds */
  --bg-primary: #0A0B0D;
  --bg-secondary: #111318;
  --bg-tertiary: #1A1D23;
  --bg-card: #1E2127;
  --bg-elevated: #252831;
  
  /* Textos */
  --text-primary: #F8FAFC;
  --text-secondary: #CBD5E1;
  --text-muted: #64748B;
  --text-accent: #94A3B8;
  
  /* Bordas e sombras */
  --border-primary: rgba(255,107,53,0.1);
  --border-secondary: rgba(255,107,53,0.15);
  --border-accent: rgba(255,107,53,0.25);
  --shadow-sm: 0 1px 2px 0 rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
  --shadow-glow: 0 0 20px rgba(255,107,53,0.2);
  --shadow-glow-success: 0 0 20px rgba(6,214,160,0.2);
  --shadow-glow-danger: 0 0 20px rgba(255,51,102,0.2);
  --shadow-glow-warning: 0 0 20px rgba(255,182,39,0.2);
  
  /* Gradientes */
  --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  --gradient-success: linear-gradient(135deg, var(--success) 0%, #34D399 100%);
  --gradient-danger: linear-gradient(135deg, var(--danger) 0%, #FF6B9D 100%);
  --gradient-warning: linear-gradient(135deg, var(--warning) 0%, #FFD23F 100%);
  --gradient-card: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
  --gradient-glass: linear-gradient(145deg, rgba(255,107,53,0.1) 0%, rgba(255,107,53,0.05) 100%);
  --gradient-fire: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
}

/* Reset e base */
* { box-sizing: border-box; }
.stApp { 
  background: var(--bg-primary); 
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Container principal */
.main .block-container { 
  max-width: 1400px !important; 
  padding: 2rem 1.5rem !important;
  margin: 0 auto !important;
}

/* Tipografia premium */
h1, h2, h3, h4, h5, h6 { 
  font-family: 'Inter', sans-serif; 
  color: var(--text-primary);
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.2;
}

h1 { font-size: 2.5rem; font-weight: 700; }
h2 { font-size: 2rem; font-weight: 600; }
h3 { font-size: 1.5rem; font-weight: 600; }
h4 { font-size: 1.25rem; font-weight: 500; }

/* Hero section */
.main-hero { 
  font-size: 2.5rem; 
  font-weight: 800; 
  text-align: left; 
  margin: 1rem 0 1.5rem;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(0,212,255,0.3);
}

.subtitle { 
  text-align: left; 
  color: var(--text-secondary); 
  margin-bottom: 2rem;
  font-size: 1.1rem;
  font-weight: 400;
  line-height: 1.6;
}

/* Cards premium */
.card { 
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: var(--shadow-lg);
  backdrop-filter: blur(10px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: var(--gradient-primary);
  opacity: 0.6;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-xl);
  border-color: var(--border-accent);
}

/* Métricas */
.metric { 
  background: var(--gradient-glass);
  border: 1px solid var(--border-secondary);
  border-radius: 12px;
  padding: 1.25rem;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
  position: relative;
}

.metric:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-glow);
}

/* Badges */
.badge { 
  display: inline-flex;
  align-items: center;
  padding: 0.375rem 0.75rem;
  border-radius: 50px;
  background: rgba(255,107,53,0.1);
  border: 1px solid rgba(255,107,53,0.2);
  color: var(--primary);
  font-size: 0.75rem;
  font-weight: 500;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.025em;
  backdrop-filter: blur(10px);
}

.badge-success { 
  background: rgba(6,214,160,0.1);
  border-color: rgba(6,214,160,0.2);
  color: var(--success);
}

.badge-danger { 
  background: rgba(255,51,102,0.1);
  border-color: rgba(255,51,102,0.2);
  color: var(--danger);
}

.badge-warning { 
  background: rgba(255,182,39,0.1);
  border-color: rgba(255,182,39,0.2);
  color: var(--warning);
}

/* Botões premium */
.stButton > button {
  background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.5rem !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 15px rgba(255,107,53,0.3) !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  position: relative !important;
  overflow: hidden !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(255,107,53,0.4) !important;
  background: linear-gradient(135deg, #E55A2B 0%, #E8851A 100%) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

/* Botões secundários */
.stButton > button[kind="secondary"] {
  background: linear-gradient(135deg, #06D6A0 0%, #34D399 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.5rem !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 15px rgba(6,214,160,0.3) !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button[kind="secondary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(6,214,160,0.4) !important;
  background: linear-gradient(135deg, #05B888 0%, #10B981 100%) !important;
}

/* Alertas estilizados */
.stAlert {
  border-radius: 12px !important;
  border: none !important;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
  backdrop-filter: blur(10px) !important;
}

.stAlert[data-testid="alert"] {
  background: linear-gradient(135deg, rgba(255,51,102,0.1) 0%, rgba(255,51,102,0.05) 100%) !important;
  border-left: 4px solid #FF3366 !important;
  color: #FF3366 !important;
}

.stAlert[data-testid="alert"] .stMarkdown {
  color: #FF3366 !important;
}

/* Info boxes */
.stInfo {
  background: linear-gradient(135deg, rgba(6,214,160,0.1) 0%, rgba(6,214,160,0.05) 100%) !important;
  border-left: 4px solid #06D6A0 !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 15px rgba(6,214,160,0.1) !important;
}

.stInfo .stMarkdown {
  color: #06D6A0 !important;
}

/* Warning boxes */
.stWarning {
  background: linear-gradient(135deg, rgba(255,182,39,0.1) 0%, rgba(255,182,39,0.05) 100%) !important;
  border-left: 4px solid #FFB627 !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 15px rgba(255,182,39,0.1) !important;
}

.stWarning .stMarkdown {
  color: #FFB627 !important;
}

/* Métricas estilizadas */
.stMetric {
  background: linear-gradient(145deg, rgba(255,107,53,0.05) 0%, rgba(247,147,30,0.02) 100%) !important;
  border: 1px solid rgba(255,107,53,0.1) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
  box-shadow: 0 4px 15px rgba(255,107,53,0.1) !important;
  backdrop-filter: blur(10px) !important;
  transition: all 0.3s ease !important;
}

.stMetric:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(255,107,53,0.2) !important;
  border-color: rgba(255,107,53,0.2) !important;
}

.stMetric > div {
  color: var(--text-primary) !important;
}

.stMetric > div > div:first-child {
  color: #FF6B35 !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}

.stMetric > div > div:last-child {
  color: #F7931E !important;
  font-weight: 700 !important;
  font-size: 1.2rem !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* Dataframes estilizados */
.stDataFrame {
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: 0 4px 15px rgba(255,107,53,0.1) !important;
  border: 1px solid rgba(255,107,53,0.1) !important;
}

.stDataFrame table {
  background: linear-gradient(145deg, rgba(255,107,53,0.02) 0%, rgba(247,147,30,0.01) 100%) !important;
}

.stDataFrame thead th {
  background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
  color: white !important;
  font-weight: 600 !important;
  border: none !important;
}

.stDataFrame tbody tr:nth-child(even) {
  background: rgba(255,107,53,0.02) !important;
}

.stDataFrame tbody tr:hover {
  background: rgba(255,107,53,0.05) !important;
}

/* Emojis estilizados */
.emoji {
  display: inline-block;
  font-size: 1.2em;
  filter: drop-shadow(0 0 8px rgba(255,107,53,0.3));
  transition: all 0.3s ease;
}

.emoji:hover {
  transform: scale(1.1) rotate(5deg);
  filter: drop-shadow(0 0 12px rgba(255,107,53,0.5));
}

.emoji-success {
  filter: drop-shadow(0 0 8px rgba(6,214,160,0.3));
}

.emoji-success:hover {
  filter: drop-shadow(0 0 12px rgba(6,214,160,0.5));
}

.emoji-danger {
  filter: drop-shadow(0 0 8px rgba(255,51,102,0.3));
}

.emoji-danger:hover {
  filter: drop-shadow(0 0 12px rgba(255,51,102,0.5));
}

.emoji-warning {
  filter: drop-shadow(0 0 8px rgba(255,182,39,0.3));
}

.emoji-warning:hover {
  filter: drop-shadow(0 0 12px rgba(255,182,39,0.5));
}

/* Efeitos de partículas */
.particles {
  position: relative;
  overflow: hidden;
}

.particles::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,107,53,0.1) 1px, transparent 1px);
  background-size: 20px 20px;
  animation: float 20s linear infinite;
  pointer-events: none;
  z-index: -1;
}

/* Efeitos de brilho */
.glow-effect {
  position: relative;
}

.glow-effect::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, transparent 30%, rgba(255,107,53,0.1) 50%, transparent 70%);
  animation: shimmer 3s ease-in-out infinite;
  pointer-events: none;
  z-index: 1;
}

/* Efeitos de neblina */
.fog-effect {
  position: relative;
  overflow: hidden;
}

.fog-effect::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(255,107,53,0.05) 0%, transparent 50%, rgba(247,147,30,0.05) 100%);
  animation: fog 8s ease-in-out infinite;
  pointer-events: none;
  z-index: -1;
}

/* Animações premium */
@keyframes float {
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(180deg); }
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes fog {
  0%, 100% { opacity: 0.3; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(1.1); }
}

/* Efeitos de vidro premium */
.glass-premium {
  background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.2);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.2);
  position: relative;
}

.glass-premium::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
}

/* Efeitos de gradiente animado */
.gradient-animated {
  background: linear-gradient(-45deg, #FF6B35, #F7931E, #FFD23F, #FF6B35);
  background-size: 400% 400%;
  animation: gradientShift 8s ease infinite;
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Separadores */
hr { 
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-primary), transparent);
  margin: 2rem 0;
}

/* Anomaly detection icons */
.anomaly-wrap { 
  display: inline-flex; 
  align-items: center; 
  gap: 1rem; 
}

.anomaly-icon { 
  width: 32px; 
  height: 32px; 
  border-radius: 50%; 
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(10px);
}

.anomaly-dot { 
  width: 16px; 
  height: 16px; 
  border-radius: 50%;
  position: relative;
}

.anomaly-normal { 
  background: var(--gradient-success);
  box-shadow: var(--shadow-glow-success);
}

.anomaly-detected { 
  background: var(--gradient-danger);
  box-shadow: var(--shadow-glow-danger);
}

/* Status cards */
.status-normal { 
  background: linear-gradient(135deg, rgba(6,214,160,0.08), rgba(6,214,160,0.03));
  border: 1px solid rgba(6,214,160,0.2);
  box-shadow: var(--shadow-glow-success);
}

.status-anomaly { 
  background: linear-gradient(135deg, rgba(255,51,102,0.08), rgba(255,51,102,0.03));
  border: 1px solid rgba(255,51,102,0.2);
  box-shadow: var(--shadow-glow-danger);
}

.status-warning { 
  background: linear-gradient(135deg, rgba(255,182,39,0.08), rgba(255,182,39,0.03));
  border: 1px solid rgba(255,182,39,0.2);
  box-shadow: var(--shadow-glow-warning);
}

/* Sidebar premium */
.css-1d391kg {
  background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
  border-right: 1px solid var(--border-accent) !important;
  box-shadow: 4px 0 20px rgba(0,0,0,0.3) !important;
  backdrop-filter: blur(20px) !important;
}

/* Sidebar navigation items */
.css-1d391kg .stSelectbox > div > div {
  background: rgba(255,107,53,0.05) !important;
  border: 1px solid rgba(255,107,53,0.1) !important;
  border-radius: 12px !important;
  transition: all 0.3s ease !important;
}

.css-1d391kg .stSelectbox > div > div:hover {
  background: rgba(255,107,53,0.1) !important;
  border-color: rgba(255,107,53,0.2) !important;
  box-shadow: 0 0 15px rgba(255,107,53,0.1) !important;
}

/* Sidebar text */
.css-1d391kg .stMarkdown {
  color: var(--text-primary) !important;
}

.css-1d391kg .stMarkdown h1,
.css-1d391kg .stMarkdown h2,
.css-1d391kg .stMarkdown h3 {
  color: var(--primary) !important;
  text-shadow: 0 0 10px rgba(255,107,53,0.3) !important;
}

/* Streamlit components */
.stSelectbox > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-primary) !important;
  border-radius: 8px !important;
}

.stTextInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-primary) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}

/* Scrollbar personalizada */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
  background: var(--border-secondary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--primary);
}

/* Animações avançadas */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes glow {
  0%, 100% {
    box-shadow: 0 0 20px rgba(255,107,53,0.2);
  }
  50% {
    box-shadow: 0 0 30px rgba(255,107,53,0.4);
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

.card, .metric {
  animation: fadeInUp 0.6s ease-out;
}

.card:hover {
  animation: glow 2s ease-in-out infinite;
}

.badge {
  animation: slideIn 0.8s ease-out;
}

.anomaly-icon {
  animation: float 3s ease-in-out infinite;
}

/* Responsividade */
@media (max-width: 768px) {
  .main .block-container {
    padding: 1rem !important;
  }
  
  .main-hero {
    font-size: 2rem;
  }
  
  .card {
    padding: 1rem;
  }
}

/* Efeitos de glassmorphism avançados */
.glass {
  background: rgba(255, 107, 53, 0.05);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 107, 53, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glass-fire {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(247, 147, 30, 0.05) 100%);
  backdrop-filter: blur(25px);
  border: 1px solid rgba(255, 107, 53, 0.2);
  box-shadow: 0 0 40px rgba(255, 107, 53, 0.1);
}

.glass-success {
  background: linear-gradient(135deg, rgba(6, 214, 160, 0.1) 0%, rgba(6, 214, 160, 0.05) 100%);
  backdrop-filter: blur(25px);
  border: 1px solid rgba(6, 214, 160, 0.2);
  box-shadow: 0 0 40px rgba(6, 214, 160, 0.1);
}

.glass-danger {
  background: linear-gradient(135deg, rgba(255, 51, 102, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%);
  backdrop-filter: blur(25px);
  border: 1px solid rgba(255, 51, 102, 0.2);
  box-shadow: 0 0 40px rgba(255, 51, 102, 0.1);
}

/* Loading states */
.loading {
  position: relative;
  overflow: hidden;
}

.loading::after {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { left: -100%; }
  100% { left: 100%; }
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_training_history():
    """Carrega o histórico de treinamento do modelo U-Net localmente"""
    HISTORY_PATH = "training_history/bottle_unet_history.json"
    
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Erro ao carregar histórico de treinamento: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_example_images():
    """Carrega imagens de exemplo localmente"""
    # Busca todos os arquivos de imagem na pasta IMAGES_DIR
    # Usa glob para encontrar .png, .jpg, .jpeg, etc.
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    # Ordena as imagens para uma exibição consistente
    image_files.sort()
    
    # Verifica se as imagens existem localmente
    existing_images = []
    for image_path in image_files:
        if os.path.exists(image_path):
            existing_images.append(image_path)
        else:
            st.warning(f"Imagem não encontrada: {image_path}")
    
    return existing_images

def get_env_status():
    """Retorna informações do ambiente"""
    gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu else platform.processor() or "CPU"
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if gpu else "-"
    return {
        "device": "GPU" if gpu else "CPU",
        "device_name": device_name,
        "torch": torch_ver,
        "cuda": cuda_ver,
        "python": platform.python_version(),
    }

def show_status(model, config, history):
    """Exibe o status do sistema com design premium"""
    st.markdown("""
    <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
      <h2 style="font-size: 1.8rem; font-weight: 600; color: #F8FAFC; 
                 letter-spacing: -0.025em; margin-bottom: 0.5rem;">
        <span class="emoji emoji-success">⚡</span> Status do Sistema
      </h2>
      <div style="width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #FF6B35, transparent); 
                  margin: 0 auto; border-radius: 1px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        model_status = "Carregado" if model else "Indisponível"
        status_class = "badge-success" if model else "badge-danger"
        st.markdown(
            f"""
<div class="metric glass-fire">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: {'#06D6A0' if model else '#FF3366'}; margin-right: 0.5rem;"></div>
    <b style="color: var(--text-primary);">Modelo U-Net</b>
  </div>
  <span class="badge {status_class}">{model_status}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    
    with c2:
        config_status = "OK" if config else "Não encontrado"
        status_class = "badge-success" if config else "badge-danger"
        st.markdown(
            f"""
<div class="metric glass-fire">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: {'#06D6A0' if config else '#FF3366'}; margin-right: 0.5rem;"></div>
    <b style="color: var(--text-primary);">Configuração</b>
  </div>
  <span class="badge {status_class}">{config_status}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    
    with c3:
        history_status = "Disponível" if history else "Sem histórico"
        status_class = "badge-success" if history else "badge-warning"
        st.markdown(
            f"""
<div class="metric glass-fire">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: {'#06D6A0' if history else '#FFB627'}; margin-right: 0.5rem;"></div>
    <b style="color: var(--text-primary);">Histórico de Treino</b>
  </div>
  <span class="badge {status_class}">{history_status}</span>
</div>
""",
            unsafe_allow_html=True,
        )

def page_home(model, config, history):
    """Página inicial do aplicativo"""
    # Header elegante e consistente com paleta quente
    st.markdown("""
    <div class="particles" style="text-align: center; margin-bottom: 2rem; position: relative;">
      <div class="glow-effect">
        <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   margin-bottom: 0.5rem; letter-spacing: -0.025em; text-shadow: 0 0 30px rgba(255,107,53,0.3);">
          <span class="emoji">🔍</span> Anomaly Detection • U-Net
        </h1>
      </div>
      <p style="font-size: 1.1rem; color: #B0B0B0; font-weight: 400; max-width: 500px; margin: 0 auto;">
        <span class="emoji emoji-success">🤖</span> Detecção de anomalias em garrafas usando modelo U-Net treinado no dataset MVTec
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    show_status(model, config, history)
    st.markdown("---")

    # Métricas do modelo expandidas
    st.markdown("""
    <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
      <h2 style="font-size: 1.8rem; font-weight: 600; color: #F8FAFC; 
                 letter-spacing: -0.025em; margin-bottom: 0.5rem;">
        <span class="emoji emoji-warning">📊</span> Métricas do Modelo
      </h2>
      <div style="width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #F7931E, transparent); 
                  margin: 0 auto; border-radius: 1px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        threshold = config.get('classification_threshold', 'N/D') if config else 'N/D'
        st.markdown(f"""
<div class="card glass-fire" style="min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: #FF6B35; margin-right: 0.5rem;"></div>
    <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-primary);">Threshold de Classificação</h4>
  </div>
  <div class="badge" style="font-size: 0.8rem;">{threshold}</div>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        if history:
            last_epoch = len(history)
            final_loss = history[-1].get('val_loss', 'N/D')
        else:
            last_epoch = 0
            final_loss = 'N/D'
        st.markdown(f"""
<div class="card glass-fire" style="min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: #F7931E; margin-right: 0.5rem;"></div>
    <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-primary);">Épocas Treinadas</h4>
  </div>
  <div class="badge" style="font-size: 0.8rem;">{last_epoch}</div>
</div>
""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
<div class="card glass-fire" style="min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: #06D6A0; margin-right: 0.5rem;"></div>
    <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-primary);">Loss Final</h4>
  </div>
  <div class="badge" style="font-size: 0.8rem;">{final_loss}</div>
</div>
""", unsafe_allow_html=True)
    
    with col4:
        env_info = get_env_status()
        st.markdown(f"""
<div class="card glass-fire" style="min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;">
  <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 8px; height: 8px; border-radius: 50%; background: #FFB627; margin-right: 0.5rem;"></div>
    <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-primary);">Dispositivo</h4>
  </div>
  <div class="badge" style="font-size: 0.8rem;">{env_info['device']}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    
    # Gráfico de evolução do treinamento
    if history:
        st.markdown("""
        <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
          <h2 style="font-size: 1.8rem; font-weight: 600; color: #F8FAFC; 
                     letter-spacing: -0.025em; margin-bottom: 0.5rem;">
            <span class="emoji emoji-warning">📈</span> Evolução do Treinamento
          </h2>
          <div style="width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #FFD23F, transparent); 
                      margin: 0 auto; border-radius: 1px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        df_history = pd.DataFrame(history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['epoch'], 
            y=df_history['train_loss'], 
            mode='lines+markers', 
            name='Train Loss',
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=6, color='#FF6B35')
        ))
        fig.add_trace(go.Scatter(
            x=df_history['epoch'], 
            y=df_history['val_loss'], 
            mode='lines+markers', 
            name='Validation Loss',
            line=dict(color='#F7931E', width=3),
            marker=dict(size=6, color='#F7931E')
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E6E6E6',
            xaxis_title="Época",
            yaxis_title="Loss",
            legend_title="Tipo",
            title="Evolução das Losses durante o Treinamento",
            title_font_color='#FF6B35',
            title_font_size=18
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cards informativos adicionais
    st.markdown("""
    <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
      <h2 style="font-size: 1.8rem; font-weight: 600; color: #F8FAFC; 
                 letter-spacing: -0.025em; margin-bottom: 0.5rem;">
        <span class="emoji emoji-success">ℹ️</span> Informações do Sistema
      </h2>
      <div style="width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #06D6A0, transparent); 
                  margin: 0 auto; border-radius: 1px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        <div class="card glass-success particles" style="min-height: 120px;">
          <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #06D6A0; margin-right: 0.5rem;"></div>
            <h4 style="font-size: 0.9rem; color: var(--text-primary);"><span class="emoji emoji-success">📁</span> Dataset</h4>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
            MVTec Bottle Dataset<br/>
            <span style="color: #06D6A0;"><span class="emoji emoji-success">✅</span> 209 imagens normais</span><br/>
            <span style="color: #FF3366;"><span class="emoji emoji-danger">⚠️</span> 42 imagens anômalas</span>
          </p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="card glass-fire particles" style="min-height: 120px;">
          <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #FF6B35; margin-right: 0.5rem;"></div>
            <h4 style="font-size: 0.9rem; color: var(--text-primary);"><span class="emoji">🏗️</span> Arquitetura</h4>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
            U-Net com encoder-decoder<br/>
            <span style="color: #FF6B35;"><span class="emoji">🔗</span> Skip connections</span><br/>
            <span style="color: #F7931E;"><span class="emoji">🔄</span> Reconstruction loss</span>
          </p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
        <div class="card glass-danger particles" style="min-height: 120px;">
          <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #FF3366; margin-right: 0.5rem;"></div>
            <h4 style="font-size: 0.9rem; color: var(--text-primary);"><span class="emoji emoji-warning">⚡</span> Performance</h4>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">
            Threshold otimizado<br/>
            <span style="color: #FF3366;"><span class="emoji emoji-danger">🎯</span> 0.000205</span><br/>
            <span style="color: #06D6A0;"><span class="emoji emoji-success">🚀</span> Alta precisão</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

def page_detect(model, config):
    """Página de detecção de anomalias"""
    # Header elegante e limpo com paleta quente
    st.markdown("""
    <div class="particles" style="text-align: center; margin-bottom: 2rem; position: relative;">
      <div class="glow-effect">
        <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   margin-bottom: 0.5rem; letter-spacing: -0.025em; text-shadow: 0 0 30px rgba(255,107,53,0.3);">
          <span class="emoji">🔬</span> Detecção de Anomalias
        </h1>
      </div>
      <p style="font-size: 1.1rem; color: #B0B0B0; font-weight: 400; max-width: 500px; margin: 0 auto;">
        <span class="emoji emoji-success">🧠</span> Sistema de análise de imagens com IA para identificação de irregularidades
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not model or not config:
        st.error("Modelo ou configuração não disponível. Verifique se os arquivos estão presentes.")
        return

    # Usar configurações do model_utils.py
    classification_threshold = config.get('classification_threshold', 0.01)
    pixel_threshold = config.get('pixel_visualization_threshold', 0.5)

    # Seção de seleção simples
    st.markdown("""
    <div class="section-header">
        <span class='emoji emoji-success'>🖼️</span> Seleção de Imagem
    </div>
    <div class="section-subtitle">
        <span class='emoji emoji-warning'>👆</span> Escolha uma imagem de exemplo para análise detalhada
    </div>
    """, unsafe_allow_html=True)

    def run_prediction(image: Image.Image, key_prefix: str = "single"):
        """Executa a predição e exibe os resultados usando funções do model_utils.py"""
        start_time = time.time()
        
        # Executa predição com configuração do model_utils.py
        results = predict(model, config, image)
        latency = (time.time() - start_time) * 1000
        
        # Determina se é anomalia
        is_anomaly = results["prediction"] == "Anomalia Detectada"
        
        # Seção de resultados elegante
        st.markdown("""
        <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
          <h2 style="font-size: 2rem; font-weight: 600; color: #F8FAFC; 
                     letter-spacing: -0.025em; margin-bottom: 0.5rem;">
            <span class="emoji emoji-success">📊</span> Resultado da Análise
          </h2>
          <div style="width: 60px; height: 2px; background: linear-gradient(90deg, transparent, #FF6B35, transparent); 
                      margin: 0 auto; border-radius: 1px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Resultado premium
        if is_anomaly:
            st.markdown(f"""
<div style="background: linear-gradient(145deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03)); 
            border: 1px solid rgba(239,68,68,0.2); 
            border-radius: 20px; 
            padding: 2rem; 
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(239,68,68,0.1), 0 0 0 1px rgba(239,68,68,0.1);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;">
  <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #EF4444, transparent);"></div>
  <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem;">
    <div style="width: 48px; height: 48px; border-radius: 50%; 
                background: linear-gradient(135deg, #EF4444, #F87171); 
                display: flex; align-items: center; justify-content: center;
                margin-right: 1rem; 
                box-shadow: 0 8px 25px rgba(239,68,68,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
      <span style="font-size: 20px; color: white;">⚠</span>
    </div>
    <h2 style="margin: 0; color: #EF4444; font-size: 2rem; font-weight: 700; letter-spacing: -0.025em;">
      Anomalia Detectada
    </h2>
  </div>
  <p style="color: #CBD5E1; font-size: 1.1rem; margin: 0 0 2rem 0; font-weight: 400;">
    Irregularidade identificada na imagem analisada
  </p>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; max-width: 400px; margin: 0 auto;">
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 12px; border: 1px solid rgba(239,68,68,0.1);">
      <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500;">Erro de Reconstrução</div>
      <div style="color: #EF4444; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{results["error"]:.6f}</div>
    </div>
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 12px; border: 1px solid rgba(0,212,255,0.1);">
      <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500;">Tempo de Análise</div>
      <div style="color: #00D4FF; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{latency:.0f}ms</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background: linear-gradient(145deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03)); 
            border: 1px solid rgba(16,185,129,0.2); 
            border-radius: 20px; 
            padding: 2rem; 
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(16,185,129,0.1), 0 0 0 1px rgba(16,185,129,0.1);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;">
  <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #10B981, transparent);"></div>
  <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem;">
    <div style="width: 48px; height: 48px; border-radius: 50%; 
                background: linear-gradient(135deg, #10B981, #34D399); 
                display: flex; align-items: center; justify-content: center;
                margin-right: 1rem; 
                box-shadow: 0 8px 25px rgba(16,185,129,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
      <span style="font-size: 20px; color: white;">✓</span>
    </div>
    <h2 style="margin: 0; color: #10B981; font-size: 2rem; font-weight: 700; letter-spacing: -0.025em;">
      Imagem Normal
    </h2>
  </div>
  <p style="color: #CBD5E1; font-size: 1.1rem; margin: 0 0 2rem 0; font-weight: 400;">
    Nenhuma anomalia detectada na imagem
  </p>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; max-width: 400px; margin: 0 auto;">
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 12px; border: 1px solid rgba(16,185,129,0.1);">
      <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500;">Erro de Reconstrução</div>
      <div style="color: #10B981; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{results["error"]:.6f}</div>
    </div>
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 12px; border: 1px solid rgba(0,212,255,0.1);">
      <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500;">Tempo de Análise</div>
      <div style="color: #00D4FF; font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{latency:.0f}ms</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        
        # Seção de análise visual elegante
        st.markdown("""
        <div class="fog-effect" style="text-align: center; margin: 2rem 0; position: relative;">
          <h2 style="font-size: 1.8rem; font-weight: 600; color: #F8FAFC; 
                     letter-spacing: -0.025em; margin-bottom: 0.5rem;">
            <span class="emoji emoji-warning">🔍</span> Análise Visual Detalhada
          </h2>
          <div style="width: 50px; height: 2px; background: linear-gradient(90deg, transparent, #F7931E, transparent); 
                      margin: 0 auto; border-radius: 1px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Primeira linha: Imagem original/bbox + Mapa de calor
        col_img, col_heat = st.columns(2)
        
        with col_img:
            st.markdown("""
            <div class="glass-premium particles" style="border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; position: relative; overflow: hidden;">
              <h4 style="color: #FF6B35; margin: 0 0 0.5rem 0; text-align: center; font-size: 0.8rem; font-weight: 500; letter-spacing: -0.025em;">
                <span class="emoji">🖼️</span> Imagem Analisada
              </h4>
            </div>
            """, unsafe_allow_html=True)
            # Se for anomalia, exibe com bounding box
            if is_anomaly:
                bbox_image = display_bounding_box(results, config)
            else:
                # Caso contrário, exibe a imagem original redimensionada
                bbox_image = np.array(results["original_image"].resize((256, 256)))
            st.image(bbox_image, use_container_width=True)
        
        with col_heat:
            st.markdown("""
            <div class="glass-premium particles" style="border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; position: relative; overflow: hidden;">
              <h4 style="color: #F7931E; margin: 0 0 0.5rem 0; text-align: center; font-size: 0.8rem; font-weight: 500; letter-spacing: -0.025em;">
                <span class="emoji emoji-warning">🔥</span> Mapa de Calor
              </h4>
            </div>
            """, unsafe_allow_html=True)
            heatmap_fig = get_heatmap_image(results)
            st.pyplot(heatmap_fig)
        
        # Segunda linha: Mapa de anomalia + Máscara binária
        col_anomaly, col_mask = st.columns(2)
        
        with col_anomaly:
            st.markdown("""
            <div class="glass-premium particles" style="border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; position: relative; overflow: hidden;">
              <h4 style="color: #FF6B35; margin: 0 0 0.5rem 0; text-align: center; font-size: 0.8rem; font-weight: 500; letter-spacing: -0.025em;">
                <span class="emoji emoji-danger">⚠️</span> Mapa de Anomalia
              </h4>
            </div>
            """, unsafe_allow_html=True)
            anomaly_fig = get_anomaly_map_image(results)
            st.pyplot(anomaly_fig)
        
        with col_mask:
            st.markdown("""
            <div class="glass-premium particles" style="border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; position: relative; overflow: hidden;">
              <h4 style="color: #F7931E; margin: 0 0 0.5rem 0; text-align: center; font-size: 0.8rem; font-weight: 500; letter-spacing: -0.025em;">
                <span class="emoji emoji-success">🎭</span> Máscara Binária
              </h4>
            </div>
            """, unsafe_allow_html=True)
            mask_fig = get_binary_mask_image(results)
            st.pyplot(mask_fig)
        
        # Botão de download estilizado
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        
        # Container para centralizar o botão
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <style>
            .stDownloadButton > button {
                background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
                box-shadow: 0 4px 15px rgba(255,107,53,0.3) !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                position: relative !important;
                overflow: hidden !important;
                width: 100% !important;
            }
            
            .stDownloadButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(255,107,53,0.4) !important;
                background: linear-gradient(135deg, #E55A2B 0%, #E8851A 100%) !important;
            }
            
            .stDownloadButton > button:active {
                transform: translateY(0) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.download_button(
                "Baixar Imagem Original", 
                data=buf.getvalue(), 
                file_name=f"imagem_{key_prefix}.png", 
                mime="image/png",
                use_container_width=True
            )

    # Seleção de exemplos simples
    example_images = get_example_images()
    if example_images:
        # Usar streamlit-image-select para seleção
        try:
            # Carrega as imagens dos arquivos temporários
            loaded_images = []
            captions = []
            
            for img_path in example_images:
                try:
                    img = Image.open(img_path)
                    loaded_images.append(img)
                    # Usa o nome do arquivo diretamente
                    caption = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                    captions.append(caption)
                except Exception as e:
                    st.warning(f"Erro ao carregar imagem {img_path}: {e}")
            
            if loaded_images:
                selected_image = image_select(
                    "",
                    images=loaded_images,
                    captions=captions
                )
            else:
                selected_image = None
                
        except Exception as e:
            st.error(f"Erro ao carregar imagens de exemplo: {e}")
            selected_image = None
        
        if selected_image is not None:
            # Botão centralizado com key única para evitar duplicação
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <style>
                .stButton > button[kind="primary"] {
                    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 12px !important;
                    padding: 0.75rem 1.5rem !important;
                    font-weight: 600 !important;
                    font-size: 0.95rem !important;
                    box-shadow: 0 4px 15px rgba(255,107,53,0.3) !important;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                    position: relative !important;
                    overflow: hidden !important;
                    width: 100% !important;
                }
                
                .stButton > button[kind="primary"]:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 25px rgba(255,107,53,0.4) !important;
                    background: linear-gradient(135deg, #E55A2B 0%, #E8851A 100%) !important;
                }
                
                .stButton > button[kind="primary"]:active {
                    transform: translateY(0) !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                if st.button("Analisar Imagem", type="primary", use_container_width=True, key="analyze_button"):
                    run_prediction(selected_image, "exemplo")
    else:
        st.info("Nenhuma imagem de exemplo encontrada. Verifique a conexão com o GitHub.")

def page_training():
    """Página de análise do treinamento"""
    st.markdown("## Análise do Treinamento")
    
    history = load_training_history()
    
    if history:
        df_history = pd.DataFrame(history)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Épocas Totais", len(df_history))
        with col2:
            st.metric("Loss Final (Train)", f"{df_history['train_loss'].iloc[-1]:.6f}")
        with col3:
            st.metric("Loss Final (Val)", f"{df_history['val_loss'].iloc[-1]:.6f}")
        with col4:
            best_epoch = df_history['val_loss'].idxmin() + 1
            st.metric("Melhor Época", best_epoch)
        
        st.markdown("---")
        
        # Gráficos de evolução
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### <span class='emoji emoji-warning'>📊</span> Evolução das Losses")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=df_history['epoch'], 
                y=df_history['train_loss'], 
                mode='lines+markers', 
                name='Train Loss',
                line=dict(color='#FF6B35', width=3),
                marker=dict(size=6, color='#FF6B35')
            ))
            fig_loss.add_trace(go.Scatter(
                x=df_history['epoch'], 
                y=df_history['val_loss'], 
                mode='lines+markers', 
                name='Validation Loss',
                line=dict(color='#F7931E', width=3),
                marker=dict(size=6, color='#F7931E')
            ))
            
            fig_loss.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E6E6E6',
                xaxis_title="Época",
                yaxis_title="Loss",
                legend_title="Tipo"
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.markdown("### <span class='emoji emoji-success'>📊</span> Distribuição das Losses")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df_history['train_loss'], 
                name='Train Loss',
                opacity=0.7,
                marker_color='#FF6B35'
            ))
            fig_dist.add_trace(go.Histogram(
                x=df_history['val_loss'], 
                name='Validation Loss',
                opacity=0.7,
                marker_color='#F7931E'
            ))
            
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E6E6E6',
                xaxis_title="Loss",
                yaxis_title="Frequência",
                legend_title="Tipo"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Tabela de dados
        st.markdown("### <span class='emoji'>📋</span> Dados de Treinamento")
        st.dataframe(df_history, use_container_width=True)
        
    else:
        st.warning("Histórico de treinamento não encontrado.")

def page_about():
    """Página sobre o projeto"""
    st.markdown("## <span class='emoji emoji-success'>ℹ️</span> Sobre o Projeto")
    st.markdown(
        """
<div class="card">
<p>Aplicativo profissional em Streamlit para <b>detecção de anomalias em garrafas</b> usando modelo U-Net treinado no dataset MVTec, com interface dark premium e análise visual detalhada.</p>

<h4>Tecnologias Utilizadas:</h4>
<ul>
<li><b>Modelo:</b> U-Net para reconstrução de imagens</li>
<li><b>Dataset:</b> MVTec AD - Bottle</li>
<li><b>Framework:</b> PyTorch</li>
<li><b>Interface:</b> Streamlit com design premium</li>
<li><b>Visualização:</b> Matplotlib, Plotly, OpenCV</li>
</ul>

<h4>Funcionalidades:</h4>
<ul>
<li>Classificação de imagens como normal ou anômala</li>
<li>Visualização de mapa de anomalia</li>
<li>Máscara binária de pixels anômalos</li>
<li>Mapa de calor sobreposto</li>
<li>Análise do histórico de treinamento</li>
</ul>

<p><b>Autor:</b> <a href="https://github.com/sidnei-almeida" target="_blank">sidnei-almeida</a><br/>
<b>Contato:</b> <a href="mailto:sidnei.almeida1806@gmail.com">sidnei.almeida1806@gmail.com</a></p>
</div>
""",
        unsafe_allow_html=True,
    )

def main():
    """Função principal do aplicativo"""
    
    # Sidebar com navegação
    with st.sidebar:
        st.markdown("<h2 style='color:#FF6B35; text-shadow: 0 0 10px rgba(255,107,53,0.3);'><span class='emoji'>🧭</span> Navegação</h2>", unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Início", "Detecção", "Treinamento", "Sobre"],
            icons=["house", "search", "graph-up", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#FF6B35"},
                "nav-link": {"color": "#E6E6E6", "--hover-color": "#171A23"},
                "nav-link-selected": {"background-color": "rgba(255,107,53,0.12)", "color": "#FF6B35", "border-left": "4px solid #FF6B35", "border-radius": "6px"},
            },
        )

        # Status do ambiente
        st.markdown("---")
        st.markdown("<h4 style='margin-bottom:0.5rem; color:#F7931E;'><span class='emoji emoji-warning'>⚙️</span> Ambiente</h4>", unsafe_allow_html=True)
        env = get_env_status()
        st.markdown(
            f"""
<div class="card glass-fire">
  <div style="color: var(--text-primary);"><b>Dispositivo:</b> <span style="color: #FF6B35;">{env['device']}</span></div>
  <div style="color: var(--text-primary);"><b>Nome:</b> <span style="color: #F7931E;">{env['device_name']}</span></div>
  <div style="color: var(--text-primary);"><b>Python:</b> <span style="color: #06D6A0;">{env['python']}</span></div>
  <div style="color: var(--text-primary);"><b>Torch:</b> <span style="color: #FFB627;">{env['torch']}</span> (CUDA: <span style="color: #FF3366;">{env['cuda']}</span>)</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Carrega modelo e configuração
    try:
        model, config = setup_model_and_config()
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        model, config = None, None
    
    # Carrega histórico de treinamento
    history = load_training_history()

    # Navegação entre páginas
    if selected == "Início":
        page_home(model, config, history)
    elif selected == "Detecção":
        page_detect(model, config)
    elif selected == "Treinamento":
        page_training()
    else:
        page_about()

if __name__ == "__main__":
    main()
