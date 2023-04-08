import streamlit as st
import cv2
import numpy as np
from uuid import uuid4
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

setup_logger()

def get_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if st.session_state.session_id not in st.session_state:
        st.session_state[st.session_state.session_id] = {}
    return st.session_state[st.session_state.session_id]

def set_session_state(state):
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if st.session_state.session_id not in st.session_state:
        st.session_state[st.session_state.session_id] = {}
    st.session_state[st.session_state.session_id] = state

def is_user_logged_in():
    return get_session_state().get("is_logged_in", False)

if not is_user_logged_in():
    st.error("You need to log in to access this page.")
    st.stop()

st.title("House Wall Detection")
st.write("Upload an image to detect House Walls & ceilings.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("Applying Segmentation for Walls & ceilings...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    only_segmented_image_op = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info, alpha=0.01)

    choice_1 = st.sidebar.checkbox('wall', value=True)
    choice_2 = st.sidebar.checkbox('ceiling')
    if choice_1:
        choice = ['wall']
    if choice_2:
        choice = ['ceiling']
    if choice_1 and choice_2:
        choice = ['wall', 'ceiling']
    new_wall_color = st.sidebar.color_picker('Choose a color', '#FFFFFF')

    if 'wall' and 'ceiling' in choice:
        b, g, r = tuple(int(new_wall_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        im = image
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        wall_index = -1
        for idx, item in enumerate(metadata.stuff_classes):
            if item == "wall":
                wall_index = idx
                break
        if wall_index != -1:
            metadata.stuff_colors[wall_index] = (r, g, b)

        ceiling_index = -1
        for idx, item in enumerate(metadata.stuff_classes):
            if item == "ceiling":
                ceiling_index = idx
                break
        if ceiling_index != -1:
            metadata.stuff_colors[ceiling_index] = (r, g, b)

        wall_segment_id = None
        for segment_info in segments_info:
            if segment_info["category_id"] == wall_index:
                wall_segment_id = segment_info["id"]
                break

        if wall_segment_id is not None:
            wall_mask = (panoptic_seg == wall_segment_id).numpy()
            wall_color = (r, g, b)
            im[wall_mask] = wall_color
            st.image(im[:, :, ::-1], caption='Output Image.', use_column_width=True)

    elif 'wall' in choice:
        b, g, r = tuple(int(new_wall_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        im = image
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        wall_index = -1
        for idx, item in enumerate(metadata.stuff_classes):
            if item == "wall":
                wall_index = idx
                break
        if wall_index != -1:
            metadata.stuff_colors[wall_index] = (r, g, b)

        wall_segment_id = None
        for segment_info in segments_info:
            if segment_info["category_id"] == wall_index:
                wall_segment_id = segment_info["id"]
                break

        if wall_segment_id is not None:
            wall_mask = (panoptic_seg == wall_segment_id).numpy()
            wall_color = (r, g, b)
            im[wall_mask] = wall_color
            st.image(im[:, :, ::-1], caption='Output Image.', use_column_width=True)

    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    with st.expander("Show detected objects"):
        col1, col2 = st.columns(2)
        col1.image(out.get_image()[:, :, ::-1], caption='Output Image.', use_column_width=True)
        col2.image(only_segmented_image_op.get_image()[:, :, ::-1], caption='segmented image', use_column_width=True)

    if st.button('Download Image'):
        cv2.imwrite('output.jpg', out.get_image()[:, :, ::-1])
        st.write('Image downloaded')
