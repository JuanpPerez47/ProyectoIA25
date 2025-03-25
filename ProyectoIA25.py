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

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """ Carga el modelo entrenado """
    model_path = "modelo_entrenado.h5"
    if not os.path.exists(model_path):
        st.error("Error: No se encontró el modelo entrenado. Verifica la ruta.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

with st.spinner('Cargando modelo...'):
    model = load_model()

# Cargar nombres de clases correctamente
class_names = []
try:
    with open("./claseIA.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    if not class_names:
        st.error("El archivo claseIA.txt está vacío. Asegúrese de que contiene los nombres de las clases.")
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt. Verifique la ruta.")

def preprocess_image(image):
    """ Preprocesa la imagen para el modelo VGG16 """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Expandir dimensión para batch
    image_array = preprocess_input(image_array)  # Normalización específica de VGG16
    return image_array

def import_and_predict(image_data, model, class_names):
    """ Realiza la predicción con el modelo """
    if model is None:
        return "Modelo no cargado", 0.0

    image = preprocess_image(image_data)
    prediction = model.predict(image)

    index = np.argmax(prediction[0])  # Obtener índice de la clase con mayor probabilidad
    confidence = np.max(prediction[0])  # Obtener la probabilidad máxima

    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Desconocido"
    
    return class_name, confidence

def generar_audio(texto):
    """ Genera un archivo de audio a partir del texto """
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    """ Reproduce el audio generado """
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Cargar imagen desde cámara, archivo o URL
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto") or \
                  st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

resultado = "No se ha procesado ninguna imagen."  # Se define antes para evitar el error

if img_file_buffer is None:
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Realizar la predicción si hay una imagen
if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)

        # Realizar la predicción
        class_name, confidence_score = import_and_predict(image, model, class_names)

        # Mostrar el resultado y generar audio
        if confidence_score > 0.5:  # Umbral de confianza del 50%
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
