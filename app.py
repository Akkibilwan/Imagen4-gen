import streamlit as st
from PIL import Image
import io
import json
import google.generativeai as genai
from google.cloud import aiplatform
from google.oauth2 import service_account
import os
import time # For unique keys / basic progress

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="Multi-Image Thumbnail Analyzer & Generator",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #121212; color: #e0e0e0; }
    .stApp { background-color: #121212; }
    h1, h2, h3 { color: #ffffff; font-family: 'Roboto', sans-serif; }
    p, li, div[data-baseweb="base-input"], div[data-baseweb="textarea"] { color: #cccccc; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stFileUploader label { color: #e0e0e0 !important; }
    .stMultiSelect div[data-baseweb="select"] > div { background-color: #333333; color: #e0e0e0; }
    .stImage { border: 2px solid #444444; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Helper: Credentials & Client Initialization ---
def initialize_clients(gcp_project_id, gcp_location, creds_json_content):
    """Initializes Google AI clients using service account JSON content."""
    try:
        credentials_info = json.loads(creds_json_content)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

        # Initialize Gemini (google-generativeai)
        genai.configure(credentials=credentials) # Or use API key if preferred and configured

        # Initialize Vertex AI (google-cloud-aiplatform)
        aiplatform.init(project=gcp_project_id, location=gcp_location, credentials=credentials)
        
        # Specific models
        # For multimodal analysis and prompt generation
        gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or gemini-pro-vision
        # For text refinement (optional)
        gemini_text_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or gemini-pro
        # For Imagen generation (via Vertex AI)
        imagen_model = aiplatform.ImageGenerationModel.from_pretrained("imagegeneration@006")


        st.session_state.clients_initialized = True
        st.session_state.gemini_vision_model = gemini_vision_model
        st.session_state.gemini_text_model = gemini_text_model
        st.session_state.imagen_model = imagen_model
        st.success("Google AI Clients Initialized Successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing Google AI Clients: {e}")
        st.session_state.clients_initialized = False
        return False

# --- Core Functions ---

# 2. Analyze Image and Create Breakdown Prompts (Gemini)
BREAKDOWN_CATEGORIES_PROMPT_SYSTEM = """
You are an expert YouTube thumbnail analyst. Your task is to analyze the provided image and break it down into distinct visual and conceptual components.
For each component category I list, provide:
1. 'analysis': A brief description of that component in the image.
2. 'prompt_suggestion': A concise, descriptive phrase (3-10 words) that could be used to recreate *only that specific component* in a new image. Make it sound like a part of an image generation prompt.

The categories for breakdown are:
- Main Subject: The primary focus (person, object, character).
- Action/Activity: What the main subject is doing.
- Setting/Background: The environment or backdrop.
- Key Objects (excluding main subject): Other notable items.
- Dominant Colors/Palette: Main colors and overall color scheme.
- Artistic Style: e.g., photorealistic, cartoon, 3D render, illustration, graphic design.
- Text Overlay: If text is present, its content and style.
- Overall Vibe/Emotion: The feeling the image conveys (e.g., exciting, calm, mysterious).
- Compositional Focus: How elements are arranged, main focal point.

Return your response ONLY as a valid JSON object where keys are the category names (e.g., "Main Subject")
and values are objects containing 'analysis' and 'prompt_suggestion'.
If a category is not applicable (e.g., no text), provide a relevant 'analysis' (e.g., "No text visible") and a null or empty 'prompt_suggestion'.
"""

def analyze_image_and_create_prompts(image_bytes):
    """Analyzes image with Gemini, breaks it down, and generates prompt suggestions."""
    if 'gemini_vision_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Gemini Vision client not initialized.")
        return None

    model = st.session_state.gemini_vision_model
    image_part = {"mime_type": "image/png", "data": image_bytes} # Assuming PNG, adjust if needed

    try:
        full_prompt = [BREAKDOWN_CATEGORIES_PROMPT_SYSTEM, image_part]
        response = model.generate_content(full_prompt)
        
        # Clean the response text to ensure it's valid JSON
        # Gemini can sometimes wrap JSON in ```json ... ```
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        analysis_result = json.loads(cleaned_response_text.strip())
        return analysis_result
    except Exception as e:
        st.error(f"Error during Gemini analysis: {e}")
        st.error(f"Gemini raw response: {getattr(response, 'text', 'N/A')}")
        return None

# 4. Generate Image (Imagen via Vertex AI)
def generate_image_with_imagen(prompt_text):
    """Generates an image using Imagen on Vertex AI."""
    if 'imagen_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Imagen client not initialized.")
        return None

    model = st.session_state.imagen_model
    try:
        st.info(f"Generating image with prompt: '{prompt_text[:100]}...' (This may take a minute)")
        # Add YouTube specific enhancements to the prompt
        enhanced_prompt = (
            f"Create a hyper-realistic YouTube thumbnail with a 16:9 aspect ratio. "
            f"The image should be extremely high quality, photorealistic, and follow YouTube thumbnail best practices "
            f"with vibrant colors and clear focal points. Make it look professional and eye-catching. "
            f"Content details: {prompt_text}"
        )

        images = model.generate_images(
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio="16:9", # "1:1", "9:16", "16:9", "4:3", "3:4"
            # You can add seed, safety_filter_level, person_generation, etc.
        )
        if images and images.images:
            # The `_image_bytes` attribute is internal, but commonly used.
            # Alternatively, save to a temp file and read if this is problematic.
            img_bytes = images.images[0]._image_bytes
            return img_bytes
        else:
            st.error("Imagen did not return any images.")
            return None
    except Exception as e:
        st.error(f"Error during Imagen generation: {e}")
        return None

# --- Streamlit App UI & Logic ---
st.title("🖼️ YouTube Thumbnail Analyzer & Creative Prompt Generator")
st.subheader("Upload, Analyze, Select Prompts, and Generate with Google AI")

# --- Initialize Session State Variables ---
if 'clients_initialized' not in st.session_state:
    st.session_state.clients_initialized = False
if 'uploaded_image_analyses' not in st.session_state:
    st.session_state.uploaded_image_analyses = [] # List of dicts: {id, name, image_obj, analysis_data}
if 'all_selectable_prompts' not in st.session_state:
    st.session_state.all_selectable_prompts = [] # List of tuples: (unique_id, display_label, prompt_text, source_image_name)
if 'selected_prompt_texts_for_generation' not in st.session_state:
    st.session_state.selected_prompt_texts_for_generation = []
if 'final_combined_prompt' not in st.session_state:
    st.session_state.final_combined_prompt = ""
if 'generated_image_bytes' not in st.session_state:
    st.session_state.generated_image_bytes = None

# --- Credentials Section ---
with st.sidebar:
    st.header("🔑 Google Cloud Credentials")
    gcp_project_id = st.text_input("GCP Project ID", st.secrets.get("GCP_PROJECT_ID", ""))
    gcp_location = st.text_input("GCP Location (e.g., us-central1)", st.secrets.get("GCP_LOCATION", "us-central1"))
    
    # Option 1: Use Streamlit Secrets for credentials JSON string
    creds_json_str_secret = st.secrets.get("GOOGLE_CREDENTIALS_JSON_STR")

    # Option 2: Upload credentials JSON file
    uploaded_sa_file = st.file_uploader("Upload Service Account JSON", type=['json'])
    
    creds_json_content = None
    if uploaded_sa_file:
        creds_json_content = uploaded_sa_file.read().decode("utf-8")
    elif creds_json_str_secret:
        creds_json_content = creds_json_str_secret

    if st.button("Initialize Google AI Clients", disabled=st.session_state.clients_initialized):
        if gcp_project_id and gcp_location and creds_json_content:
            initialize_clients(gcp_project_id, gcp_location, creds_json_content)
        else:
            st.warning("Please provide Project ID, Location, and Service Account JSON.")

    if st.session_state.clients_initialized:
        st.success("Clients ready!")
    else:
        st.info("Please provide credentials and initialize clients to proceed.")

# --- Main App Sections ---
if not st.session_state.clients_initialized:
    st.warning("Please initialize Google AI Clients using the sidebar to use the app.")
    st.stop()

# 1. Multi-Image Upload
st.header("1. Upload Thumbnail Images")
uploaded_files = st.file_uploader(
    "Choose images to analyze (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    new_files_to_process = []
    processed_ids = [item['id'] for item in st.session_state.uploaded_image_analyses]
    for uploaded_file in uploaded_files:
        if uploaded_file.file_id not in processed_ids:
            new_files_to_process.append(uploaded_file)

    if new_files_to_process:
        if st.button(f"Analyze {len(new_files_to_process)} New Image(s)"):
            with st.spinner("Analyzing images with Gemini... please wait."):
                for uploaded_file in new_files_to_process:
                    image_id = uploaded_file.file_id
                    image_name = uploaded_file.name
                    img = Image.open(uploaded_file)
                    
                    # Convert to PNG bytes for consistency
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()

                    st.write(f"Processing: {image_name}")
                    analysis_data = analyze_image_and_create_prompts(image_bytes)
                    
                    if analysis_data:
                        st.session_state.uploaded_image_analyses.append({
                            "id": image_id,
                            "name": image_name,
                            "image_obj": img, # Store PIL image for display
                            "analysis_data": analysis_data
                        })
                        st.success(f"Analyzed: {image_name}")
                    else:
                        st.error(f"Failed to analyze: {image_name}")
                # After processing, update all_selectable_prompts
                st.session_state.all_selectable_prompts = []
                for analysis_item in st.session_state.uploaded_image_analyses:
                    for category, data in analysis_item['analysis_data'].items():
                        prompt_text = data.get('prompt_suggestion')
                        if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                            unique_prompt_id = f"{analysis_item['id']}_{category}"
                            display_label = f"({analysis_item['name']}) {category}: {prompt_text}"
                            st.session_state.all_selectable_prompts.append(
                                (unique_prompt_id, display_label, prompt_text, analysis_item['name'])
                            )
            st.rerun() # Rerun to update UI with new analyses and selectable prompts

# Display Analyzed Images and their Breakdowns
if st.session_state.uploaded_image_analyses:
    st.header("Analyzed Images & Breakdowns")
    for item in st.session_state.uploaded_image_analyses:
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(item['image_obj'], caption=item['name'], use_column_width=True)
        with cols[1]:
            st.subheader(f"Breakdown for: {item['name']}")
            if item['analysis_data']:
                for category, data in item['analysis_data'].items():
                    st.markdown(f"**{category}**: ")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Analysis*: {data.get('analysis', 'N/A')}")
                    if data.get('prompt_suggestion'):
                         st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Prompt Idea*: `{data.get('prompt_suggestion')}`")
            else:
                st.write("No analysis data available.")
        st.divider()
else:
    st.info("Upload images and click 'Analyze' to see breakdowns here.")


# 3. User Selection of Prompts
if st.session_state.all_selectable_prompts:
    st.header("2. Select Prompt Components for New Thumbnail")
    
    # Store selections in session state to persist across reruns
    if 'current_selected_labels' not in st.session_state:
        st.session_state.current_selected_labels = []

    # Use display labels for multiselect options
    options = [item[1] for item in st.session_state.all_selectable_prompts]
    
    # Update selected_labels from session_state before rendering multiselect
    selected_labels = st.multiselect(
        "Choose prompt segments from the analyses above:",
        options=options,
        default=st.session_state.current_selected_labels, # Use stored selections
        key=f"prompt_multiselect_{len(st.session_state.all_selectable_prompts)}" # Dynamic key if options change
    )
    # Update session state with current selections
    st.session_state.current_selected_labels = selected_labels

    # Map selected display labels back to actual prompt texts
    st.session_state.selected_prompt_texts_for_generation = []
    for label in selected_labels:
        for item in st.session_state.all_selectable_prompts:
            if item[1] == label:
                st.session_state.selected_prompt_texts_for_generation.append(item[2])
                break
    
    if st.session_state.selected_prompt_texts_for_generation:
        st.subheader("Selected Prompt Segments:")
        for seg_idx, segment in enumerate(st.session_state.selected_prompt_texts_for_generation):
            st.markdown(f"- `{segment}`")

        if st.button("Combine Selected Segments into Final Prompt"):
            st.session_state.final_combined_prompt = ", ".join(st.session_state.selected_prompt_texts_for_generation)
            # Optional: Refine with Gemini text model here if desired
            # e.g., prompt_to_refine = "Refine these image prompt segments into a coherent paragraph: " + st.session_state.final_combined_prompt
            # refined_prompt = st.session_state.gemini_text_model.generate_content(prompt_to_refine).text
            # st.session_state.final_combined_prompt = refined_prompt
            st.rerun() # Update the displayed final prompt

    if st.session_state.final_combined_prompt:
        st.subheader("Final Combined Prompt for Generation:")
        st.text_area("Edit if needed:", value=st.session_state.final_combined_prompt, height=100, key="final_prompt_edit_area")
        st.session_state.final_combined_prompt = st.session_state.final_prompt_edit_area # Allow user edits

        if st.button("🚀 Generate Thumbnail with Imagen", type="primary"):
            with st.spinner("Generating your new thumbnail with Imagen... This might take a moment!"):
                generated_bytes = generate_image_with_imagen(st.session_state.final_combined_prompt)
                if generated_bytes:
                    st.session_state.generated_image_bytes = generated_bytes
                    st.success("Thumbnail Generated!")
                else:
                    st.error("Thumbnail generation failed. Check logs if any.")
            st.rerun() # To display the image

# Display Generated Image
if st.session_state.generated_image_bytes:
    st.header("🎉 Your Generated Thumbnail 🎉")
    st.image(st.session_state.generated_image_bytes, caption="Generated by Imagen", use_column_width=True)
    st.download_button(
        label="Download Thumbnail",
        data=st.session_state.generated_image_bytes,
        file_name="generated_thumbnail.png",
        mime="image/png"
    )

st.markdown("---")
st.caption("Built with Streamlit and Google Generative AI")
