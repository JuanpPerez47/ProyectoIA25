import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import warnings
from gtts import gTTS
import base64

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Reconocimiento de Objetos",
    page_icon=":smile:",
    initial_sidebar_state='auto'
)

# Ocultar elementos de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Cargar el modelo con cache para evitar recargas innecesarias
@st.cache_resource
def load_model():
    model_path = "./modelo_entrenado.h5"
    if not os.path.exists(model_path):
        st.error("Error: No se encontró el modelo entrenado. Verifica la ruta.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

with st.spinner('Cargando modelo...'):
    model = load_model()

# Cargar nombres de clases
class_names = []
try:
    with open("claseIA.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    if not class_names:
        st.error("El archivo claseIA.txt está vacío. Asegúrese de que contiene los nombres de las clases.")
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt. Verifique la ruta.")

# Configuración de la barra lateral
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen")
    st.subheader("Identificación de objetos con VGG16")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

st.image('smartregionlab2.jpeg')
st.title("Modelo de Identificación de Objetos - Smart Regions Center")
st.write("Desarrollo del Proyecto de Ciencia de Datos con Redes Convolucionales")

# Función para preprocesar la imagen
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Expandir dimensión batch
    image_array = preprocess_input(image_array)  # Normalización de VGG16
    return image_array

# Función de predicción
def import_and_predict(image, model, class_names):
    if model is None:
        return "Modelo no cargado", 0.0

    image = preprocess_image(image)
    prediction = model.predict(image)

    index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Desconocido"

    return class_name, confidence

# Función para generar audio
def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# Función para reproducir el audio en Streamlit
def reproducir_audio(mp3_fp):
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Captura de imagen desde la cámara o carga desde archivo
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto") or \
                  st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

resultado = "No se ha procesado ninguna imagen."

if img_file_buffer is None:
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Procesar la imagen y realizar la predicción
if img_file_buffer and model:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)

        # Realizar la predicción
        class_name, confidence_score = import_and_predict(image, model, class_names)

        # Mostrar el resultado y generar audio
        if confidence_score > confianza:
            resultado = f"Tipo de Objeto: {class_name}\nPuntuación de confianza: {100 * confidence_score:.2f}%"
            st.subheader(f"🔍 Tipo de Objeto: {class_name}")
            st.text(f"Puntuación de confianza: {100 * confidence_score:.2f}%")
        else:
            resultado = "No se pudo determinar el tipo de objeto"
            st.text(resultado)

        # Generar y reproducir el audio
        mp3_fp = generar_audio(resultado)
        reproducir_audio(mp3_fp)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.text("Por favor, cargue una imagen usando una de las opciones anteriores.")
