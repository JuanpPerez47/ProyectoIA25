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
    with open("claseIA.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip().lower() for line in f.readlines()]
    if not class_names:
        st.error("El archivo claseIA.txt está vacío.")
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt.")

# Cargar descripciones desde `proma.txt`
descripcion_dict = {}
try:
    with open("proma.txt", "r", encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split(":", 1)
            if len(partes) == 2:
                clave = partes[0].strip().lower()
                descripcion = partes[1].strip()
                descripcion_dict[clave] = descripcion
except FileNotFoundError:
    st.error("No se encontró el archivo proma.txt.")

# Configuración de la barra lateral
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen")
    st.subheader("Identificación de objetos con VGG16")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

st.image('smartregionlab2.jpeg')
st.title("Modelo de Identificación de Objetos - Smart Regions Center")
st.write("Desarrollo del Proyecto de Ciencia de Datos con Redes Convolucionales")

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

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

def generar_audio(texto):
    """Genera audio asegurando que siempre haya contenido."""
    if not texto.strip():
        texto = "No se encontró información para este objeto."
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    """Reproduce el audio generado en Streamlit."""
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

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

if img_file_buffer and model:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)

        class_name, confidence_score = import_and_predict(image, model, class_names)

        descripcion = descripcion_dict.get(class_name, "No hay información disponible para este objeto.")

        if confidence_score > confianza:
            resultado = f"🔍 Objeto Detectado: {class_name.capitalize()}\n"
            resultado += f"🎯 Confianza: {100 * confidence_score:.2f}%\n\n"
            resultado += f"📌 **Descripción:** {descripcion}"
            
            st.subheader(f"🔍 Tipo de Objeto: {class_name.capitalize()}")
            st.text(f"🎯 Confianza: {100 * confidence_score:.2f}%")
            st.write(f"📌 **Descripción:** {descripcion}")

        else:
            resultado = "No se pudo determinar el tipo de objeto"
            st.text(resultado)

        # Generar y reproducir el audio con la descripción
        mp3_fp1 = generar_audio(resultado)
        mp3_fp = generar_audio(descripcion)
        reproducir_audio(mp3_fp1)
        reproducir_audio(mp3_fp)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.text("Por favor, cargue una imagen usando una de las opciones anteriores.")
