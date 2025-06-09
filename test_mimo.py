#!/usr/bin/env python3
"""
Simple test script for MiMo multimodal model
"""
import torch
from PIL import Image
import numpy as np

# Import our MiMo model
import sys
sys.path.append('.')

# Import components from mimo-multimodal.py
exec(open('mimo-multimodal.py').read().split('# Example usage')[0])

def test_multimodal_model():
    print("🚀 Testing MiMo Multimodal Model")
    print("=" * 50)
    
    # Initialize model
    print("📦 Loading model...")
    model = MiMo7B()
    
    # Load test image
    try:
        image = Image.open("unnamed.png")
        print(f"📸 Loaded image: {image.size}, mode: {image.mode}")
        
        # Show image stats
        img_array = np.array(image)
        print(f"🎨 Image stats - Mean RGB: {img_array.mean(axis=(0,1))}")
        
    except Exception as e:
        print(f"⚠️  Could not load image: {e}")
        print("🎭 Using test pattern instead")
        image = Image.new("RGB", (224, 224))
        # Create a simple test pattern
        img_array = np.array(image)
        img_array[:112, :] = [255, 0, 0]  # Red top half
        img_array[112:, :] = [0, 0, 255]  # Blue bottom half
        image = Image.fromarray(img_array)
    
    # Test simple prompt
    print("\n🤖 Testing simple multimodal prompt...")
    prompt = "Describe this image: <image>"
    
    try:
        response = model.generate(
            prompt=prompt,
            images=[image],
            max_length=256,
            temperature=0.8
        )
        
        print(f"💬 Response: {response}")
        
        # Test another prompt
        print("\n🔍 Testing another prompt...")
        prompt2 = "What colors do you see in this image: <image>?"
        response2 = model.generate(
            prompt=prompt2,
            images=[image],
            max_length=128,
            temperature=0.7
        )
        print(f"💬 Response 2: {response2}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_multimodal_model()
