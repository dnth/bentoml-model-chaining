import streamlit as st
import requests
import base64
import json
import logging
from PIL import Image, ImageDraw

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def run_tag_image(image_file):
    base_url = "http://localhost:3000"
    api_url = f"{base_url}/tag_image_file"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    
    response = requests.post(api_url, files=files)
    return response.json()

def run_detect_objects(image_file):
    base_url = "http://localhost:3001"
    api_url = f"{base_url}/detect_objects_file"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    
    response = requests.post(api_url, files=files)
    return response.json()

def run_caption_image(image_file):
    base_url = "http://localhost:3002"
    api_url = f"{base_url}/caption_image_file"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    
    response = requests.post(api_url, files=files)
    return response.text.strip()

def draw_bounding_boxes_yolov8(image, detections, confidence_threshold=0.0):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        score = detection['confidence']
        if score < confidence_threshold:
            continue
        box = detection['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        label = f"{detection['name']} {score:.2f}"
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")
    return image

def draw_bounding_boxes_owlv2(image, detections, confidence_threshold=0.0):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        score = detection['score']
        if score < confidence_threshold:
            continue
        bbox = detection['box']
        x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        label = f"{detection['label']} {score:.2f}"
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")
    return image

def run_owlv2_inference(image_file, queries, confidence_threshold):
    base_url = "http://localhost:3003"
    api_url = f"{base_url}/detect_objects"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    data = {
        'queries': queries,
        'confidence_threshold': confidence_threshold
    }
    
    response = requests.post(api_url, files=files, data=data)
    
    return response.json()

def run_phi35_inference(image_file, prompt):
    base_url = "http://localhost:3004"
    api_url = f"{base_url}/caption_image"
    
    files = {
        'image': (image_file.name, image_file, 'image/jpeg')
    }
    data = {
        'prompt': prompt
    }
    
    response = requests.post(api_url, files=files, data=data)
    return response.text.strip()

def main():
    logging.basicConfig(level=logging.INFO)
    
    st.title("Image Inference Demo")
    
    # Create tabs
    tabs = st.tabs(["Image Tagging", "Object Detection", "Image Captioning", "Zero-Shot Detection", "VQA"])
    
    st.sidebar.title("Input image")
    
    # File uploader (shared between tabs)
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Image Tagging tab
        with tabs[0]:
            st.header("Image Tagging - Recognize Anything Model Plus")
            st.markdown("[Swagger UI](http://localhost:3000)")
            if st.button('Run Image Tagging'):
                with st.spinner('Running image tagging...'):
                    try:
                        result = run_tag_image(uploaded_file)
                        st.success('Image tagging complete!')
                        st.json(result)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
        # Object Detection tab
        with tabs[1]:
            st.header("Object Detection - YOLOv8x")
            st.markdown("[Swagger UI](http://localhost:3001)")
            # Add confidence threshold slider
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
            if st.button('Run Object Detection'):
                with st.spinner('Running object detection...'):
                    try:
                        result = run_detect_objects(uploaded_file)
                        st.success('Object detection complete!')
                        result_json = result
                        
                        # Display all detections in JSON
                        st.json(result_json)
                        
                        # Plot the detection result
                        image = Image.open(uploaded_file)
                        image_with_boxes = draw_bounding_boxes_yolov8(image, result_json, confidence_threshold)
                        st.image(image_with_boxes, caption='Object Detection Result', use_column_width=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Full error details:")
                        st.write(e)
        
        # Image Captioning tab
        with tabs[2]:
            st.header("Image Captioning - BLIP2")
            st.markdown("[Swagger UI](http://localhost:3002)")
            if st.button('Run Image Captioning'):
                with st.spinner('Running image captioning...'):
                    try:
                        result = run_caption_image(uploaded_file)
                        st.success('Image captioning complete!')
                        result_json = {"caption": result}
                        st.json(result_json)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Full error details:")
                        st.write(e)
        
        # OWLv2 Zero-Shot Detection tab
        with tabs[3]:
            st.header("Zero-Shot Detection - OWLv2")
            st.markdown("[Swagger UI](http://localhost:3003)")
            queries = st.text_input("Enter object queries (comma-separated)", "car")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
            if st.button('Run OWLv2 Zero-Shot Detection'):
                with st.spinner('Running OWLv2 zero-shot detection...'):
                    try:
                        result = run_owlv2_inference(uploaded_file, queries, confidence_threshold)
                        st.success('OWLv2 zero-shot detection complete!')
                        
                        st.json(result)
                        
                        # Plot the detection result
                        image = Image.open(uploaded_file)
                        image_with_boxes = draw_bounding_boxes_owlv2(image, result, confidence_threshold)
                        st.image(image_with_boxes, caption='OWLv2 Detection Result', use_column_width=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Full error details:")
                        st.write(e)

        # Phi 3.5 Vision tab
        with tabs[4]:
            st.header("Phi 3.5 Vision")
            st.markdown("[Swagger UI](http://localhost:3004)")
            prompt = st.text_area("Enter your prompt", "Describe the image in detail, focusing on the main subjects, their actions, and the overall setting. Include information about colors, textures, and any notable objects or elements in the background.")
            if st.button('Run Phi 3.5 Vision'):
                with st.spinner('Running Phi 3.5 Vision...'):
                    try:
                        result = run_phi35_inference(uploaded_file, prompt)
                        st.success('Phi 3.5 Vision analysis complete!')
                        st.write(result)
                        result_json = {"phi3.5_vision": result}
                        st.json(result_json)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("Full error details:")
                        st.write(e)

if __name__ == "__main__":
    main()