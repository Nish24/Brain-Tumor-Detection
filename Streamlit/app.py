import streamlit as st
import numpy as np

from io import BytesIO
from PIL import Image
import tensorflow as tf

st.header(":red[Brain Tumor] Detection :brain:", divider="rainbow")
upload_file = st.file_uploader("Upload an Image")

if upload_file is not None:
    if st.button("Predict"):
        col1, col2 = st.columns(2)
        with col1:
            MODEL = tf.keras.models.load_model("Models/2")
            CLASS_NAMES = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
            st.image(upload_file, caption="Uploaded Image", width=300)
            img = np.array(Image.open(BytesIO(upload_file.getvalue())))
            img = tf.image.resize(img, [256, 256])
            img_bt = np.expand_dims(img, 0)
            prediction = MODEL.predict(img_bt)[0]
            prediction_class = CLASS_NAMES[np.argmax(prediction)] 
            confidence = round(np.max(prediction) * 100, 2)
        with col2:
            if(prediction_class == "No Tumor"):
                st.title(f":green[{prediction_class}]")
                st.title(f"({confidence}% confidence)")
            else:
                tab1,tab2 = st.tabs(["🧠Result", "📝Info"])
                with tab1:
                    st.title(f":red[{prediction_class}]")
                    st.title(f"({confidence}% confidence)")
                with tab2:
                    if prediction_class == "Glioma Tumor":
                        st.write("Glioma is a type of brain tumor that originates in the glial cells, which support and protect the nerve cells in the brain. These tumors can vary in aggressiveness and are classified by their cell type and location in the brain. Symptoms of gliomas can vary but often include headaches, seizures, and neurological deficits.  \n\n  Treatment options depend on the tumor's grade and location and may include surgery, radiation therapy, and chemotherapy. Gliomas are typically challenging to treat due to their tendency to invade surrounding brain tissue, and their prognosis varies widely depending on various factors, including the tumor's type and stage.")
                    elif prediction_class == "Meningioma Tumor":
                        st.write("Meningioma is a type of tumor that arises from the meninges, the layers of tissue covering the brain and spinal cord. These tumors are usually benign (non-cancerous) and slow-growing, often causing no symptoms. However, if they grow large enough or press on nearby structures, they can lead to symptoms such as headaches, seizures, and changes in vision or personality.  \n\n  Treatment options include observation, surgery, radiation therapy, and in some cases, medication. The prognosis for meningioma is generally favorable, especially for benign tumors that are completely removed. Regular monitoring is recommended for those with asymptomatic or small tumors.")
                    else:
                        st.write("A pituitary tumor is an abnormal growth in the pituitary gland, a pea-sized gland located at the base of the brain. These tumors can be benign (non-cancerous) or, rarely, malignant (cancerous). Depending on their size and type, they can cause hormonal imbalances, leading to a variety of symptoms such as headaches, vision problems, fatigue, and changes in weight or appetite.  \n\n  Treatment options include medication to reduce tumor size, surgery to remove the tumor, and radiation therapy. Regular monitoring and management by healthcare professionals are crucial to prevent complications and maintain hormonal balance.")