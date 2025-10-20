import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import torch
import io
import os

# Import the functions from the current directory
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import the functions
from image_crops import overlap_crop_image, reconstruct_from_crops

# Set page config
st.set_page_config(
    page_title="Image Processing Application",
    page_icon="🖼️",
    layout="wide"
)

def apply_filter(crop, filter_name):
    """Apply the selected filter to an image crop"""
    if filter_name == "Normal":
        return crop
    elif filter_name == "Black & White":
        return crop.convert("L").convert("RGB")
    elif filter_name == "Blur":
        return crop.filter(ImageFilter.BLUR)
    elif filter_name == "Contour":
        return crop.filter(ImageFilter.CONTOUR)
    elif filter_name == "Sharpen":
        return crop.filter(ImageFilter.SHARPEN)
    return crop

def main():
    st.title("Advanced Image Processing Application")
    st.write("Process large images by tiling, editing, and stitching them back together.")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        st.subheader("Image Processing Options")
        filter_option = st.selectbox(
            "Choose a filter:",
            ["Normal", "Black & White", "Blur", "Contour", "Sharpen"]
        )

        overlap = st.slider("Overlap (pixels):", 0, 20, 4, 1)
        max_crops = st.slider("Maximum Number of Tiles:", 4, 16, 9, 1)

        process_btn = st.button("Process Image")
    
    if uploaded_file is not None and process_btn:
        try:
            # Load and display original image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Convert to numpy array for processing
            image_np = np.array(image)
            
            # Split into tiles
            with st.spinner("Splitting image into tiles..."):
                output = overlap_crop_image(
                    image_np, 
                    overlap_margin=overlap, 
                    max_crops=max_crops
                )
                
                # Get local crops (skip the global crop)
                local_crops = output["crops"][1:]
                
                # Process each crop
                processed_crops = []
                for i, crop_np in enumerate(local_crops):
                    # Convert numpy array to PIL Image
                    crop_img = Image.fromarray(crop_np)
                    
                    # Apply selected filter
                    processed_crop = apply_filter(crop_img, filter_option)
                    processed_crops.append(processed_crop)
            
            # Convert processed crops back to numpy arrays
            processed_np = [np.array(img) for img in processed_crops]
            
            # Reconstruct the image
            with st.spinner("Reconstructing image..."):
                crops_tensor = torch.from_numpy(np.array(processed_np)).float()
                reconstructed = reconstruct_from_crops(
                    crops_tensor, 
                    output["tiling"], 
                    overlap_margin=overlap
                )
                
                # Convert back to PIL Image for display
                result_img = Image.fromarray(reconstructed.cpu().numpy().astype(np.uint8))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Processed Image")
                st.image(result_img, use_container_width=True)

                # Download button
                buffered = io.BytesIO()
                result_img.save(buffered, format="JPEG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/jpeg"
                )

            # Show crop grid
            st.subheader("Processed Tiles")
            cols = st.columns(3)  # 3 columns for the grid
            for idx, crop in enumerate(processed_crops):
                with cols[idx % 3]:
                    st.image(crop, caption=f"Tile {idx+1}", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    elif uploaded_file is None and process_btn:
        st.warning("Please upload an image first.")

if __name__ == "__main__":
    main()
