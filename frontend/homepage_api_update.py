# write a streamlit app to upload an image and apply image segmentation in it as per detectron2_tutorial.py

import streamlit as st
import cv2
import numpy as np
import requests
import io

st.title("House Wall Detection")

st.write("Upload an image to detect House Walls & ceilings.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    st.write("")
    st.write("Detecting Walls & ceilings...")

    # Convert the image to a byte stream
    retval, buffer = cv2.imencode(".png", image)
    byte_stream = io.BytesIO(buffer)

    # sending request to the API endpoint to segment the image
    api_url = "https://7b89-103-123-226-98.ngrok-free.app/segment"

    # Send the API request with the image as form data
    files = {"image": ("image.png", byte_stream, "image/png")}
    response = requests.post(api_url, files=files)

    # Display the API response
    if response.status_code == 200:
        # print(response.content)
        st.image(response.content)
        st.write("API Response:", (response))
    else:
        st.write("Error:", response.status_code)

    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.DEVICE = 'cpu'
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(image)
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # st.image(out.get_image()[:, :, ::-1], caption='Output Image.', use_column_width=True)

    # st.write("")
    # st.write("Applying Segmentation for Walls & ceilings...")
    # #Also apply panoptic segmentation
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    # cfg.MODEL.DEVICE = 'cpu'
    # predictor = DefaultPredictor(cfg)
    # panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    # st.image(out.get_image()[:, :, ::-1], caption='Output Image.', use_column_width=True)

    # #add a multiselect to select the color of the wall or ceiling
    # option = ['wall', 'ceiling']
    # choice = st.multiselect('Select the color of the wall or ceiling', option)

    # #if the user selects the wall
    # if 'wall' in choice:

    #     st.write("")
    #     st.write("Changing color of Wall...")

    #     #take an user input for color as RGB
    #     r = st.slider('Red', 0, 255, 0)
    #     g = st.slider('Green', 0, 255, 0)
    #     b = st.slider('Blue', 0, 255, 0)
    #     new_wall_color = (r,g,b)

    #     #change the color of the wall
    #     im = image
    #     panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    #     metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    #     # Find the index of the 'wall' label
    #     wall_index = -1
    #     for idx, item in enumerate(metadata.stuff_classes):
    #         if item == "wall":
    #             wall_index = idx
    #             break

    #     # If the label is found, change the color of the 'wall' label
    #     if wall_index != -1:
    #         # Change the color of 'wall' to your desired RGB value
    #         new_wall_color = (r,g,b)
    #         metadata.stuff_colors[wall_index] = new_wall_color

    #     # Pass the modified metadata to the Visualizer
    #     v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    #     out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    #     st.image(out.get_image()[:, :, ::-1], caption='Output Image.', use_column_width=True)

    # elif 'ceiling' in choice:
    #     st.write("")
    #     st.write("Changing color of ceiling...")

    #     #take an user input for color as RGB
    #     r = st.slider('Red', 0, 255, 0)
    #     g = st.slider('Green', 0, 255, 0)
    #     b = st.slider('Blue', 0, 255, 0)
    #     new_ceiling_color = (r,g,b)

    #     #change the color of the ceiling
    #     im = image
    #     panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    #     metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    #     # Find the index of the 'ceiling' label
    #     ceiling_index = -1
    #     for idx, item in enumerate(metadata.stuff_classes):
    #         if item == "ceiling":
    #             ceiling_index = idx
    #             break

    #     # If the label is found, change the color of the 'ceiling' label
    #     if ceiling_index != -1:
    #         # Change the color of 'ceiling' to your desired RGB value
    #         new_ceiling_color = (r,g,b)
    #         metadata.stuff_colors[ceiling_index] = new_ceiling_color

    #     # Pass the modified metadata to the Visualizer
    #     v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    #     out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    #     st.image(out.get_image()[:, :, ::-1], caption='Output Image.', use_column_width=True)

    # #add a button to download the image
    # if st.button('Download Image'):
    #     cv2.imwrite('output.jpg', out.get_image()[:, :, ::-1])
    #     st.write('Image downloaded')
