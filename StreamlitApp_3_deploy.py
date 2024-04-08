import streamlit as st
import requests
import base64
from PIL import Image

API_URL_DICT = {
"efficientnet-b7":"https://api-inference.huggingface.co/models/google/efficientnet-b7",
"resnet-18":"https://api-inference.huggingface.co/models/microsoft/resnet-18",
"vit-base-patch16-224":"https://api-inference.huggingface.co/models/google/vit-base-patch16-224"}

headers = {"Authorization": "Bearer input_your_token"}

def input_features():
    model = st.selectbox("Модель", ("vit-base-patch16-224",
                                           "efficientnet-b7",
                                            'resnet-18'))
    flow = st.selectbox("Инференс", ("Huggingface",))
    return model, inference

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def query(model, data):
    response = requests.post(API_URL_DICT[model], headers=headers, data=data)
    return response.json()

def predict(model, flow, data):
    result = query(model, data)
    return result

def inference(model, flow, upload):
    c1, c2= st.columns(2)
    if upload is not None:
        upload_copy = upload
        output = predict(model, flow, upload)
        im= Image.open(upload_copy)        
        c1.header('Input Image')
        c1.image(im)        
        c2.header(f'Output {model}')
        c2.subheader('Predicted class :')
        c2.write(output[0]['label'])        
    
def show_main_page():    
    st.set_page_config(
        layout="wide",        
        page_title="Many Models Inference",    
    )
    set_png_as_page_bg('backgrounds/geometry-geometric-web-dark-wallpaper-38267d8820d01cd810ac21ae7822848a.jpg')
    st.title("Распознаем объект на изображении при помощи различных моделей.")
    st.header("Так же можем выбрать где поднять инференс.")
    st.header('Выберите стэк')
    model, flow = input_features()
    st.header('Загрузите картинку')
    upload = st.file_uploader(':red[Insert image for classification]', type=['png','jpg'])
    inference(model, flow, upload)

def process_main_page():
    show_main_page()


if __name__ == '__main__':
    process_main_page()