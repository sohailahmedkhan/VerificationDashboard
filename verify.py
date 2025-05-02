import io
import cv2
import numpy as np
import streamlit as st
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
from streamlit_folium import folium_static
import folium
from PIL import Image
import xyzservices.providers as xyz
import json
import base64
from openai import OpenAI

# Set page style
st.set_page_config(page_title='VerificationDash', page_icon='📷', layout='wide')

# Load logo image
try:
    logo_image = Image.open("mf.png")  # Replace with the actual path to your logo image
except FileNotFoundError:
    logo_image = None
    st.warning("Logo image not found.  Make sure 'path/to/your/logo.png' is correct.")

# Display logo and title in a horizontal layout
col1, col2 = st.columns([1, 4])  # Adjust column widths as needed for logo/title balance
with col1:
    if logo_image is not None:
        st.image(logo_image, width=165)  # Adjust width as desired
with col2:
    st.title("The Verification Dashboard")

# Set sidebar title and description
st.sidebar.title('ℹ️ About')
st.sidebar.info(
    'The Verification Dashboard is designed to streamline multiple verification steps in one convenient location. It integrates features such as reverse image search, OCR, landmark detection, and automated geolocation. Leveraging the power of the Google Cloud Vision API and the OpenAI API, the dashboard enables fast, efficient access to valuable information, all in a single interface.')
st.sidebar.markdown('----')

# Add a button to upload a config file
config_slot = st.empty()
config_file = config_slot.file_uploader('Upload a config file', type=['json'])

# Load the credentials from the config file
if config_file is not None:
    content = config_file.read()

    # get openai key
    json_str = content.decode('utf-8')
    config_2 = json.loads(json_str)
    openai_api_key = config_2.get("openai_api_key")

    try:
        credentials = service_account.Credentials.from_service_account_info(json.loads(content))
        client = vision.ImageAnnotatorClient(credentials=credentials)
        config_slot.empty()
        # st.sidebar.subheader('🖼️ Supported image formats:')
        # st.sidebar.markdown("""
        #         - JPG
        #         - JPEG
        #         - PNG
        #     """)
        # st.sidebar.markdown('----')
        # st.sidebar.subheader('⚠️ Free: first 1000 units/month')
        st.sidebar.markdown('----')
        st.sidebar.subheader('📘 More resources:')
        st.sidebar.markdown("""
            - [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
            - [Cloud Vision API Documentation](https://cloud.google.com/vision/docs)
            ----
            """)
        st.sidebar.button('Reset app')

        # Upload image
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

        def create_folium_map(landmarks):
            providers = xyz.flatten()
            selection = [
                'OpenTopoMap',
                'Stadia.AlidadeSmooth',
                'Stadia.AlidadeSmoothDark',
                'Stadia.OSMBright',
                'CartoDB.Positron',
                'CartoDB.Voyager',
                'WaymarkedTrails.hiking',
                'WaymarkedTrails.cycling',
                'WaymarkedTrails.mtb',
                'WaymarkedTrails.slopes',
                'WaymarkedTrails.riding',
                'WaymarkedTrails.skating',
                'OpenRailwayMap'
            ]

            m = folium.Map(
                location=[landmarks[0].locations[0].lat_lng.latitude,
                          landmarks[0].locations[0].lat_lng.longitude],
                zoom_start=15
            )

            for landmark in landmarks:
                tooltip = landmark.description
                folium.Marker(
                    location=[landmark.locations[0].lat_lng.latitude,
                              landmark.locations[0].lat_lng.longitude],
                    tooltip=tooltip
                ).add_to(m)

            for tiles_name in selection:
                tiles = providers[tiles_name]
                folium.TileLayer(
                    tiles=tiles.build_url(),
                    attr=tiles.html_attribution,
                    name=tiles.name,
                ).add_to(m)

            folium.LayerControl().add_to(m)
            return m


        if uploaded_file is not None:
            # Encode as base64
            file_bytes = uploaded_file.read()
            base64_encoded_openai = base64.b64encode(file_bytes).decode('utf-8')
            
            with st.spinner('Analyzing the image...'):
                content = file_bytes
                image = types.Image(content=content)

                # LANDMARK DETECTION
                response = client.landmark_detection(image=image)
                landmarks = response.landmark_annotations

                st.write('-------------------')
                st.subheader('📤 Uploaded image and detected location:')
                col1, col2 = st.columns(2)
                with col1:
                    image = Image.open(io.BytesIO(content))
                    st.image(image, use_container_width=True, caption='')
                if landmarks:
                    with col2:
                        folium_map = create_folium_map(landmarks)
                        folium_static(folium_map)
                    st.write('-------------------')
                    st.subheader('📍 Location information:')
                    for landmark in landmarks:
                        st.write('- **Coordinates**: ' + str(
                            landmark.locations[0].lat_lng.latitude) + ', ' + str(
                            landmark.locations[0].lat_lng.longitude))
                        st.write('- **Location**: ' + landmark.description)
                        st.write('')
                    st.write('-------------------')
                else:
                    st.write('❌ No landmarks detected.')
                    st.write('-------------------')

                # LOGO DETECTION
                image = types.Image(content=content)
                response = client.logo_detection(image=image)
                logos_detected = response.logo_annotations

                if logos_detected:
                    st.subheader('👓 Logos Detected:')
                    for logo in logos_detected:
                        st.markdown(f'''- {logo.description}''')
                else:
                    st.write('❌ No Logos Detected.')
                st.write('-------------------')

                # OBJECT DETECTION
                image = types.Image(content=content)
                response = client.object_localization(image=image)
                object_annotations = response.localized_object_annotations

                if object_annotations:
                    st.subheader('🧳 Objects Detected:')
                    annotated_image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                    if annotated_image is not None:
                        for object_found in object_annotations:
                            vertices = [(int(vertex.x * annotated_image.shape[1]),
                                         int(vertex.y * annotated_image.shape[0]))
                                        for vertex in object_found.bounding_poly.normalized_vertices]
                            for i in range(len(vertices)):
                                cv2.line(annotated_image, vertices[i],
                                         vertices[(i + 1) % len(vertices)], color=(0, 255, 0),
                                         thickness=2)
                            cv2.putText(annotated_image,
                                        f"{object_found.name} ({round(object_found.score * 100, 1)}% Confidence)",
                                        (vertices[0][0], vertices[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 2)

                        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image, channels="RGB")
                    else:
                        st.write('❌ Error loading image for object detection.')
                else:
                    st.write('❌ No Objects Detected.')
                st.write('-------------------')

                # WEB ENTITY DETECTION
                image = types.Image(content=content)
                response = client.web_detection(image=image)
                web_entities = response.web_detection.web_entities
                pages_with_matching_images = response.web_detection.pages_with_matching_images
                visually_similar_images = response.web_detection.visually_similar_images

                if web_entities or pages_with_matching_images or visually_similar_images:
                    st.subheader('🌐 Detected web entities:')
                    entity_rows = [entity.description for entity in web_entities if entity.description]
                    if entity_rows:
                        st.write(entity_rows)
                    else:
                        st.write('❌ No web entities detected.')
                    st.write('-------------------')

                    st.subheader('🔗 Pages with matching images:')
                    page_rows = [page.url for page in pages_with_matching_images]
                    if page_rows:
                        st.write(page_rows)
                    else:
                        st.write('❌ No pages with matching images found.')

                    # OCR TEXT DETECTION
                    image = types.Image(content=content)
                    response = client.text_detection(image=image)  # Use text_detection instead
                    texts = response.text_annotations
                    st.write('-------------------')
                    st.subheader('📝 Text (OCR):')
                    if texts:
                        ocr_text = texts[0].description  # Full text is in the first element
                        st.write(ocr_text)
                    else:
                        st.write('❌ No text detected.')
                    st.write('-------------------')

                    st.subheader('🖼️ Visually similar images:')
                    similar_images = [image for image in visually_similar_images if image.url]
                    num_images = len(similar_images)
                    if num_images > 0:
                        cols = st.columns(3)
                        for i, image in enumerate(similar_images):
                            if i % 3 == 0:
                                cols = st.columns(3)
                            with cols[i % 3]:
                                st.image(image.url, use_container_width=True, caption=image.url)
                    else:
                        st.write('❌ No visually similar images found.')
                else:
                    st.write('❌ No web entities detected.')

                st.write('-------------------')
        
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze the image and describe what it primarily depicts: if it shows a person, give details about their appearance, identity (if known), attire, and activity; if it shows a place or landmark, describe it with relevant context and provide estimated coordinates in this format: Coordinates: latitude, longitude; if it shows another type of subject, describe it and provide relevant information."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_encoded_openai}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )

                # Display the AI's response
                st.subheader("OpenAI Response:")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"API Error: {e}")
        else:
            st.write('📁 Please upload an image.')
        config_slot.empty()
    except json.JSONDecodeError as e:
        st.error("Invalid JSON syntax in config file: {}".format(e))
    except Exception as e:
        st.error("Error while loading config file: {}".format(e))
    config_slot.empty()
else:
    st.warning('Please upload a config file.')