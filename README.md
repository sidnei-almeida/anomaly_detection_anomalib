# 🔍 Anomaly Detection • U-Net

Aplicativo profissional em Streamlit para detecção de anomalias em garrafas usando modelo U-Net treinado no dataset MVTec AD.

## 🚀 Características

- **Interface Premium**: Design dark elegante inspirado em aplicativos profissionais
- **Detecção de Anomalias**: Classificação de imagens como normal ou anômala
- **Visualizações Detalhadas**: Mapa de anomalia, máscara binária e mapa de calor
- **Análise de Treinamento**: Gráficos de evolução das métricas
- **Seleção de Imagens**: Interface intuitiva com `streamlit-image-select`
- **Otimizado para Cloud**: Execução sem GPU, compatível com Streamlit Cloud

## 📁 Estrutura do Projeto

```
anomaly_detection_anomalib/
├── app.py                    # Aplicativo principal Streamlit
├── model_utils.py           # Utilitários do modelo U-Net
├── requirements.txt         # Dependências Python (CPU only)
├── requirements-gpu.txt     # Dependências Python (com GPU)
├── run_app.sh              # Script de execução
├── models/                 # Modelos treinados
│   ├── bottle_unet_best.pth
│   └── bottle_unet_config.json
├── imagem/                 # Imagens de exemplo
│   ├── bottle_normal_1.jpg
│   ├── bottle_anomaly_1.jpg
│   └── ...
├── notebooks/              # Notebooks de treinamento
│   ├── 1_Data_Analysis_And_Manipulation.ipynb
│   ├── 2_Model_Construction_And_Training.ipynb
│   ├── 3_Model_Valuation.ipynb
│   └── 4_Modelo_UNET_Training.ipynb
└── training_history/       # Histórico de treinamento
    └── bottle_unet_history.json
```

## 🛠️ Instalação e Execução

### 1. Instalar Dependências

```bash
# Para Streamlit Cloud (CPU only)
pip install -r requirements.txt

# Para instalação local com GPU (opcional)
pip install -r requirements-gpu.txt
```

### 2. Executar o Aplicativo

```bash
# Método 1: Script automatizado
./run_app.sh

# Método 2: Comando direto
streamlit run app.py
```

### 3. Acessar o Aplicativo

Abra seu navegador e acesse: `http://localhost:8501`

## 🎯 Funcionalidades

### 📊 Página Inicial
- Status do sistema (modelo, configuração, histórico)
- Métricas principais do modelo
- Gráfico de evolução do treinamento

### 🔍 Detecção de Anomalias
- **Exemplos**: Selecione imagens de exemplo com interface visual
- **Configurações**: Thresholds definidos no model_utils.py
- **Resultados**: 
  - Classificação (Normal/Anômala)
  - Erro de reconstrução
  - Mapa de anomalia em escala de cinza
  - Máscara binária de pixels anômalos
  - Mapa de calor sobreposto

### 📈 Análise de Treinamento
- Evolução das losses (train/validation)
- Distribuição das métricas
- Tabela completa de dados
- Identificação da melhor época

## ⚙️ Configuração do Modelo

O arquivo `models/bottle_unet_config.json` contém:

```json
{
    "classification_threshold": 0.01,
    "pixel_visualization_threshold": 0.5
}
```

- **classification_threshold**: Valor acima do qual a imagem é considerada anômala
- **pixel_visualization_threshold**: Threshold para destacar pixels anômalos na visualização

## 🎨 Design e Interface

- **Tema Dark Premium**: Cores elegantes com gradientes
- **Tipografia**: Fonte Inter para melhor legibilidade
- **Componentes**: Cards, badges e métricas estilizados
- **Responsivo**: Adaptável a diferentes tamanhos de tela
- **Navegação**: Menu lateral com `streamlit-option-menu`

## 🔧 Tecnologias Utilizadas

- **Streamlit**: Framework web para aplicações de ML
- **PyTorch**: Framework de deep learning
- **U-Net**: Arquitetura de rede neural para reconstrução
- **OpenCV**: Processamento de imagens
- **Matplotlib/Plotly**: Visualizações e gráficos
- **PIL**: Manipulação de imagens

## 📱 Compatibilidade

- ✅ **Streamlit Cloud**: Totalmente compatível (CPU only)
- ✅ **Docker**: Pode ser containerizado
- ✅ **Local**: Execução em ambiente local (CPU/GPU)
- ✅ **CPU Only**: Otimizado para execução sem GPU
- ⚠️ **GPU**: Para uso com GPU, instale versões específicas do PyTorch

## 👨‍💻 Autor

**sidnei-almeida**
- GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)
- Email: sidnei.almeida1806@gmail.com

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

*Desenvolvido com ❤️ para detecção de anomalias em imagens industriais*
