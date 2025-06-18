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
    page_title="Görüntü İşleme Uygulaması",
    page_icon="🖼️",
    layout="wide"
)

def apply_filter(crop, filter_name):
    """Apply the selected filter to an image crop"""
    if filter_name == "Normal":
        return crop
    elif filter_name == "Siyah-Beyaz":
        return crop.convert("L").convert("RGB")
    elif filter_name == "Blur":
        return crop.filter(ImageFilter.BLUR)
    elif filter_name == "Kontur":
        return crop.filter(ImageFilter.CONTOUR)
    elif filter_name == "Keskinleştir":
        return crop.filter(ImageFilter.SHARPEN)
    return crop

def main():
    st.title("Gelişmiş Görüntü İşleme Uygulaması")
    st.write("Büyük görüntüleri parçalara ayırıp işleyen ve tekrar birleştiren uygulama")

    # Sidebar controls
    with st.sidebar:
        st.header("Ayarlar")
        uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["png", "jpg", "jpeg"])
        
        st.subheader("Görüntü İşleme Ayarları")
        filter_option = st.selectbox(
            "Filtre Seçin:",
            ["Normal", "Siyah-Beyaz", "Blur", "Kontur", "Keskinleştir"]
        )
        
        overlap = st.slider("Örtüşme Payı (piksel):", 0, 20, 4, 1)
        max_crops = st.slider("Maksimum Parça Sayısı:", 4, 16, 9, 1)
        
        process_btn = st.button("Görüntüyü İşle")
    
    if uploaded_file is not None and process_btn:
        try:
            # Load and display original image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Convert to numpy array for processing
            image_np = np.array(image)
            
            # Split into tiles
            with st.spinner("Görüntü parçalara ayrılıyor..."):
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
            with st.spinner("Görüntü yeniden oluşturuluyor..."):
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
                st.subheader("Orijinal Görüntü")
                st.image(image, use_container_width=True)
                
            with col2:
                st.subheader("İşlenmiş Görüntü")
                st.image(result_img, use_container_width=True)
                
                # Download button
                buffered = io.BytesIO()
                result_img.save(buffered, format="JPEG")
                st.download_button(
                    label="İşlenmiş Görüntüyü İndir",
                    data=buffered,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/jpeg"
                )
            
            # Show crop grid
            st.subheader("İşlenen Parçalar")
            cols = st.columns(3)  # 3 columns for the grid
            for idx, crop in enumerate(processed_crops):
                with cols[idx % 3]:
                    st.image(crop, caption=f"Parça {idx+1}", use_container_width=True)
                    
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
    elif uploaded_file is None and process_btn:
        st.warning("Lütfen önce bir görüntü yükleyin.")

if __name__ == "__main__":
    main()
