# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to create a color palette
def create_palette(image, num_colors=5):
    # Resize the image to a smaller size for faster processing
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_height = 100
    new_width = int(aspect_ratio * new_height)
    small_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2LAB)

    # Reshape the image to a 2D array
    pixel_values = lab_image.reshape(-1, 3)

    # Cluster the pixel values using K-Means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixel_values)
    cluster_centers = kmeans.cluster_centers_

    # Convert the cluster centers to the RGB color space
    cluster_centers = np.array([cluster_centers[:, i][np.newaxis, :] for i in range(3)])
    rgb_centers = cv2.cvtColor(cluster_centers.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Display the color palette
    fig, ax = plt.subplots(1, num_colors, figsize=(num_colors*2, 2), dpi=100)
    for i in range(num_colors):
        color = rgb_centers[i]
        ax[i].imshow(np.array([[color]]), extent=[0, 1, 0, 1], aspect='auto')
        ax[i].set_axis_off()
        ax[i].set_title('Color {}'.format(i+1))
    st.pyplot(fig)

# Check if image is uploaded
if uploaded_file is not None:
    # Read and decode the uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Create a color palette
    create_palette(image)
