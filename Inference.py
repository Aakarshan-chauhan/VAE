import torch as t 
import numpy as np
import torch.nn as nn
from torch.serialization import load 
import model_training as mt
from model_training import VAE, Encoder, Decoder, load_data, show
import streamlit as st

def infer(model,img):
    
    img = t.reshape(img, (1, 1, 28, 28))
    preds = model(img.to(mt.device))[0].detach().cpu()
    

@st.cache
def load_models():
    model = VAE(10).to(mt.device)
    model.load_model()
    return model

@st.cache(allow_output_mutation=True)
def get_means(model:VAE, img):
    
    means = model.encode(img.to(mt.device)).detach().numpy()[0]
    return means

def reset_get_means(model:VAE, img):
    
    means = model.encode(img.to(mt.device)).detach().numpy()[0]
    return means

def imgshow(img, caption, col):
    print(img.shape)
    img = t.clamp(img.permute(1,2,0)/255, -1, 1)
    img = img.numpy()
    col.image(img, caption=caption, width=300, clamp=True)

def sliderMeans(inputs_means):
    output_means = np.empty(10, dtype=np.float32)
    for i in range(10):
        print(inputs_means[i])
        output_means[i] = st.sidebar.slider(
            f"mean: {i}",
             -10., 
             10., 
             value=float(inputs_means[i]))
    
    return output_means
    

if __name__ == "__main__":

    st.title("Variational Auto Encoder")
    model = load_models()
    print(model)
    ximgs, _ = load_data(550)
    
    st.sidebar.write("Press C to reset the sliders")

    index = st.sidebar.number_input("Enter an image index (0 - 548)", 0, 548, 0)
    means = get_means(model, ximgs[index:index+1])
    og_means = np.array(list(means.copy()))
    

    output_means = sliderMeans(means)    

    output_means = t.tensor(output_means, dtype=t.float32).reshape(1, -1)
    outs = model.decoder(output_means)[0].detach()
    
    left, mid, right = st.columns(3)
    imgshow(outs, "Recreated", left)
    imgshow(ximgs[index], "Original", right)