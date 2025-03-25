# streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# En VsC seleccione la version de Python (recomiendo 3.9) 
#CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv 

#o puede usar el siguiente comando en el shell
#Vaya a "view" en el menú y luego a terminal y lance un terminal.
#python -m venv env

#Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
#cd D:\flores\env\Scripts\
#.\activate 

#Debe quedar asi: (.venv) D:\proyectos_ia\Flores>

#Puedes verificar que no tenga ningun libreria preinstalada con
#pip freeze
#Actualicie pip con pip install --upgrade pip

#pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
#Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
#pip install streamlit
#Verifique se se instaló no trante de instalar con pip install pillow
#Esta instalacion se hace si la requiere pip install opencv-python

#Descargue una foto de una flor que le sirva de ícono 

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
import os

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

with st.spinner('Modelo está cargando..'):
    model = load_model()

with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen")
    st.subheader("Reconocimiento de imagen para objetos")
    confianza = st.slider("Seleccione el nivel de Confianza", 0, 100, 50) / 100

st.image('productose.jpg')
st.title("Modelo de Identificación de Imagenes")
st.write("Desarrollo Proyecto Final de Inteligencia Artificial : Aplicando modelos de Redes Convolucionales e Imagenes")
st.write("""
         # Detección de Objetos
         """
         )

def import_and_predict(image_data, model, class_names):
    if image_data.mode != 'RGB':
        image_data = image_data.convert('RGB')
        
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Create a batch
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index].strip()
    return class_name, score

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

class_names = open("./claseIA.txt", "r").readlines()

# Opción para capturar una imagen desde la cámara
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto")

# Opción para cargar una imagen desde un archivo local
if img_file_buffer is None:
    img_file_buffer = st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

# Opción para cargar una imagen desde una URL
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
        class_name, score = import_and_predict(image, model, class_names)
        max_score = np.max(score)

        # Mostrar el resultado y generar audio
        if max_score > confianza:
            resultado = f"Tipo de Objeti: {class_name}\nPuntuación de confianza: {100 * max_score:.2f}%"
            st.subheader(f"Tipo de Objeto: {class_name}")
            st.text(f"Puntuación de confianza: {100 * max_score:.2f}%")
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
