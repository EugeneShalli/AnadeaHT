import streamlit as st
from PIL import Image
import time
import os
import torch
from inference import MaskRCNNInference
import cv2
import matplotlib.pyplot as plt
from utils import combine_masks
import config


def app():
    base_folder = os.path.dirname(os.path.dirname(__file__))
    folder_path_original = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/original')
    device = torch.device('cpu')
    model = MaskRCNNInference(device=device, checkpoint='pytorch_model-e12.bin')

    list_imgs_path = os.listdir(folder_path_original)

    img_selected = st.selectbox("You can choose file from the server (it is from TEST set).", options=list_imgs_path)

    uploaded_file = st.file_uploader("Or you can upload your own file", type=['jpg', 'png', 'tif'])
    path_original = ''

    with st.sidebar:
        device_res = st.radio(
            "Run inference with GPU or CPU",
            ('CPU', 'GPU'))

        # load model
        # deprecated since no GPU on server
        if device_res == 'CPU':
            pass
        elif device_res == 'GPU':
            st.error("Unfortunately GPU is not supported yet, but we're working on it.")

        print(type(config.mask_threshold_dict[1]))
        A172 = st.slider('A172 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[1])
        BT474 = st.slider('BT474 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[2])
        BV2 = st.slider('BV2 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[3])
        Huh7 = st.slider('Huh7 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[4])
        MCF7 = st.slider('MCF7 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[5])
        SHSY5Y = st.slider('SHSY5Y threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[6])
        SKOV3 = st.slider('SKOV3 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[7])
        SkBr3 = st.slider('SkBr3 threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.mask_threshold_dict[8])

        A172_s = st.slider('A172 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[1])
        BT474_s = st.slider('BT474 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[2])
        BV2_s = st.slider('BV2 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[3])
        Huh7_s = st.slider('Huh7 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[4])
        MCF7_s = st.slider('MCF7 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[5])
        SHSY5Y_s = st.slider('SHSY5Y score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[6])
        SKOV3_s = st.slider('SKOV3 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[7])
        SkBr3_s = st.slider('SkBr3 score threshold', min_value=0.0, max_value=1.0, step=0.05, value=config.min_score_dict[8])

        mask_threshold_dict = {1: A172, 2: BT474, 3: BV2, 4: Huh7, 5: MCF7, 6: SHSY5Y, 7: SKOV3, 8: SkBr3}
        min_score_dict = {1: A172_s, 2: BT474_s, 3: BV2_s, 4: Huh7_s, 5: MCF7_s, 6: SHSY5Y_s, 7: SKOV3_s, 8: SkBr3_s}

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        name = uploaded_file.name
        path_original = os.path.join(folder_path_original, name)
        img.save(path_original)
        print(path_original)
        st.image(img, caption='This is your picture', width=350)
        # col1, col2 = st.columns(2)

    if img_selected:
        img_selected_path = os.path.join(folder_path_original, img_selected)
        img = Image.open(img_selected_path)
        path_original = img_selected_path
        st.image(img, caption='This is your picture', width=350)

    start_time = time.time()

    if st.button('Run Inference'):
        with st.spinner('The process may take some time. Wait for it...'):
            img = cv2.imread(path_original, cv2.IMREAD_COLOR)
            preds = model.predict(img, mask_threshold_dict, min_score_dict)

            ig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 60), facecolor="#fefefe")
            ax[0].imshow(img)
            ax[0].set_title(f"cell type {1}")
            ax[0].axis("off")

            masks = combine_masks(preds, 0.5)
            ax[1].imshow(masks)
            ax[1].set_title(f"Predicted number of cells, {len(preds)} cells")
            ax[1].axis("off")
            plt.tight_layout()
            plt.savefig('temporary.png', bbox_inches='tight')

            st.image('temporary.png', caption='These are the segmented instances', use_column_width=True)
            st.success('Done!')

            st.info(f'It took the model {time.time() - start_time:7.3f} seconds to process your image.')
            print("--- %s seconds ---" % (time.time() - start_time))


def main():
    st.markdown("# ▶ Inference")
    st.sidebar.markdown("# ▶ Inference️")
    app()


if __name__ == "__main__":
    main()
