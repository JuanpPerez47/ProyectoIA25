import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from gtts import gTTS
import base64
import os

# Configuración de la página
st.set_page_config(
    page_title="Identificación de Productos",
    page_icon=":package:",
    layout="centered"
)

st.title("Identificación de Productos por Imagen")

# Ocultar menú y footer de Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'modelo_entrenado.h5')
    return tf.keras.models.load_model(model_path)

model = load_model()

# Cargar clases desde el archivo clasesIA.txt
@st.cache_resource
def load_classes():
    with open("clasesIA.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_classes()

# Función para preprocesar imagen y predecir
def import_and_predict(image_data, model, class_names):
    image_data = image_data.convert('RGB')
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Crear batch
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    return class_names[index], 100 * np.max(score)

# Función para generar audio del resultado
def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# Función para reproducir el audio
def reproducir_audio(mp3_fp):
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Opción para subir imágenes
img_file_buffer = st.camera_input("Capturar una imagen del producto")
if img_file_buffer is None:
    img_file_buffer = st.file_uploader("O subir una imagen", type=["jpg", "jpeg", "png"])

if img_file_buffer is None:
    image_url = st.text_input("O ingrese una URL de imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except:
            st.error("Error al cargar la imagen desde la URL.")

if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)
        
        # Realizar predicción
        class_name, confidence = import_and_predict(image, model, class_names)
        resultado = f"Producto identificado: {class_name} (Confianza: {confidence:.2f}%)"
        st.subheader(resultado)
        
        # Generar y reproducir audio
        mp3_fp = generar_audio(resultado)
        reproducir_audio(mp3_fp)
    except:
        st.error("No se pudo procesar la imagen.")
