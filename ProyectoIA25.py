import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
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

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'modelo_entrenado.h5')
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Modelo está cargando...'):
    model = load_model()

with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen")
    st.subheader("Reconocimiento de imagen para objetos")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

st.image('smartregionlab2.jpeg')
st.title("Modelo de Identificación de Objetos dentro del Laboratorio Smart Regions Center")
st.write("Desarrollo del Proyecto de Ciencia de Datos : Aplicando modelos de Redes Convolucionales e Imágenes")
st.write("# Detección de Objetos")

# Cargar nombres de clases correctamente
try:
    with open("./claseIA.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    if not class_names:
        st.error("El archivo claseIA.txt está vacío. Asegúrese de que contiene los nombres de las clases.")
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt. Verifique la ruta.")

# Función de preprocesamiento corregida
def preprocess_image(image):
    image = tf.keras.utils.img_to_array(image)  # Convertir a array de TensorFlow
    image = tf.image.resize(image, (224, 224))  # Redimensionar a 224x224
    image = image / 255.0  # Normalizar
    image = tf.expand_dims(image, 0)  # Expandir dimensiones para batch
    return image

def import_and_predict(image_data, model, class_names):
    if image_data.mode != 'RGB':
        image_data = image_data.convert('RGB')

    image_data = image_data.resize((224, 224))  
    image = preprocess_image(image_data)  

    prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0]).numpy()  # Aplicar softmax correctamente
    index = np.argmax(score)
    
    class_name = class_names[index] if index < len(class_names) else "Desconocido"
    return class_name, score[index]

def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Capturar o cargar imagen
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto") or \
                  st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

if img_file_buffer is None:
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Procesar la imagen y realizar la predicción
if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)

        # Realizar la predicción
        class_name, confidence_score = import_and_predict(image, model, class_names)

        # Mostrar el resultado y generar audio
        if confidence_score > confianza:
            resultado = f"Tipo de Objeto: {class_name}\nPuntuación de confianza: {100 * confidence_score:.2f}%"
            st.subheader(f"Tipo de Objeto: {class_name}")
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
