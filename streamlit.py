from io import BytesIO
import streamlit as st
from pyngrok import ngrok
from PIL import Image
import PIL
import os
from torchvision import transforms
import torch
from cnn_for_covid_xray_public import CNN


#path = st.text_input('Enter the x-ray title to be classified:')
file_up = st.file_uploader("Upload an image", type=["jpeg","jpg","png"])
def predict(image):
    """Return the 2 predictions ranked by highest probability.
    Parameters:
    :param image: uploaded image
    :type image: png,jpg,jpeg
    :rtype: list
    :returns: predictions ranked by highest probability
    """
    pytorch_model = CNN(2)

    # transforms
    transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()])
    # loading image 
    img = Image.open(image)
    rgb_im = img.convert('RGB')
    #converting and transforming rgb image to tensor
    img2 = transform(rgb_im)
    batch_t= torch.unsqueeze(img2,dim=0)  
    pytorch_model.eval()
    out = pytorch_model(batch_t)

    
    classes = ['COVID','NORMAL']

    # return the prediction ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:2]]

if file_up is not None:
    # displays uploaded image
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the predictions with scores, highest probability value is the most likely case
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])


