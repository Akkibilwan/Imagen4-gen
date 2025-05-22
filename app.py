# app.py
import streamlit as st
from PIL import Image
import io
import json
import google.generativeai as genai
from google.cloud import aiplatform
from google.oauth2 import service_account
import os # Not strictly needed with st.secrets but good for local fallback if desired

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
    p, li, div[data-baseweb="base-input"], div[data-baseweb="textarea"], .stFileUploader label { color: #cccccc !important; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; border: none; padding: 10px 15px; }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .stMultiSelect div[data-baseweb="select"] > div { background-color: #333333; color: #e0e0e0; }
    .stImage > img { border: 2px solid #444444; border-radius: 8px; }
    div[data-testid="stSpinner"] > div { color: #007bff !important; } /* Spinner color */
</style>
""", unsafe_allow_html=True)

# --- Helper: Credentials & Client Initialization ---
def initialize_clients(gcp_project_id, gcp_location, creds_json_content):
    """Initializes Google AI clients using service account JSON content."""
    try:
        credentials_info = json.loads(creds_json_content)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

        # Initialize Gemini (google-generativeai)
        genai.configure(credentials=credentials) 

        # Initialize Vertex AI (google-cloud-aiplatform)
        # This init is crucial and must happen before accessing aiplatform.ImageGenerationModel
        aiplatform.init(project=gcp_project_id, location=gcp_location, credentials=credentials)
        
        # Specific models
        # Using a recent, capable model for Gemini
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Imagen model (using a known stable identifier)
        imagen_model_instance = aiplatform.ImageGenerationModel.from_pretrained("imagegeneration@006")

        st.session_state.clients_initialized = True
        st.session_state.gemini_model = gemini_model # Consolidated Gemini model
        st.session_state.imagen_model = imagen_model_instance
        st.success("Google AI Clients Initialized Successfully!")
        return True
    except json.JSONDecodeError:
        st.error("Error: The provided Google Credentials JSON is not valid.")
        st.session_state.clients_initialized = False
        return False
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
If a category is not applicable (e.g., no text), provide a relevant 'analysis' (e.g., "No text visible") and an empty string or null for 'prompt_suggestion'.
"""

def analyze_image_and_create_prompts(image_bytes):
    """Analyzes image with Gemini, breaks it down, and generates prompt suggestions."""
    if 'gemini_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Gemini client not initialized. Please check credentials and initialization status.")
        return None

    model = st.session_state.gemini_model
    # Ensure image_bytes is in a format Gemini expects (raw bytes for common types)
    image_part = {"mime_type": "image/png", "data": image_bytes} 

    try:
        full_prompt = [BREAKDOWN_CATEGORIES_PROMPT_SYSTEM, image_part]
        response = model.generate_content(full_prompt)
        
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        analysis_result = json.loads(cleaned_response_text.strip())
        return analysis_result
    except json.JSONDecodeError as e:
        st.error(f"Error parsing Gemini's JSON response: {e}. Ensure the model is providing valid JSON.")
        st.error(f"Gemini raw response: {getattr(response, 'text', 'N/A')}")
        return None
    except Exception as e:
        st.error(f"Error during Gemini analysis: {e}")
        st.error(f"Gemini raw response details (if available): {getattr(response, 'text', 'N/A')}")
        return None

# 4. Generate Image (Imagen via Vertex AI)
def generate_image_with_imagen(prompt_text):
    """Generates an image using Imagen on Vertex AI."""
    if 'imagen_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Imagen client not initialized. Please check credentials and initialization status.")
        return None

    model_instance = st.session_state.imagen_model
    try:
        st.info(f"Generating image with prompt: '{prompt_text[:150]}...' (This may take a minute or two)")
        enhanced_prompt = (
            f"Create a hyper-realistic YouTube thumbnail with a 16:9 aspect ratio. "
            f"The image should be extremely high quality, photorealistic, and follow YouTube thumbnail best practices "
            f"with vibrant colors and clear focal points. Make it look professional and eye-catching. "
            f"Content details: {prompt_text}"
        )

        images_response = model_instance.generate_images(
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            # seed=12345 # Optional: for reproducibility if desired
        )
        if images_response and images_response.images:
            # The `_image_bytes` attribute is internal; ensure SDK version supports this or use documented method
            # For recent SDKs, this is a common way to get bytes for a single generated image.
            img_bytes = images_response.images[0]._image_bytes 
            return img_bytes
        else:
            st.error("Imagen did not return any images. The prompt might be too restrictive or an issue occurred.")
            return None
    except Exception as e:
        st.error(f"Error during Imagen generation: {e}")
        return None

# --- Streamlit App UI & Logic ---
st.title("🖼️ YouTube Thumbnail Analyzer & Creative Prompt Generator")
st.markdown("Upload images, get AI-powered analysis and prompt ideas, then generate new thumbnails!")

# --- Initialize Session State Variables ---
if 'clients_initialized' not in st.session_state:
    st.session_state.clients_initialized = False
if 'uploaded_image_analyses' not in st.session_state:
    st.session_state.uploaded_image_analyses = [] 
if 'all_selectable_prompts' not in st.session_state:
    st.session_state.all_selectable_prompts = [] 
if 'current_selected_labels' not in st.session_state: # For multiselect state
    st.session_state.current_selected_labels = []
if 'final_combined_prompt' not in st.session_state:
    st.session_state.final_combined_prompt = ""
if 'generated_image_bytes' not in st.session_state:
    st.session_state.generated_image_bytes = None

# --- Credentials Section ---
with st.sidebar:
    st.header("🔑 Google Cloud Setup")
    # Pre-fill with a sensible default for users in India, but allow changes
    gcp_project_id = st.text_input("GCP Project ID", st.secrets.get("GCP_PROJECT_ID", ""), help="Your Google Cloud Project ID.")
    gcp_location = st.text_input("GCP Location", st.secrets.get("GCP_LOCATION", "asia-south1"), help="e.g., us-central1, asia-south1. Must be supported by Vertex AI.")
    
    creds_json_content = None
    # Option 1: Use Streamlit Secrets for credentials JSON string (recommended for deployment)
    creds_json_str_secret = st.secrets.get("GOOGLE_CREDENTIALS_JSON_STR")
    if creds_json_str_secret:
        creds_json_content = creds_json_str_secret
        st.caption("Using credentials from Streamlit Secrets.")
    else:
        # Option 2: Upload credentials JSON file (for local development)
        st.info("For deployed apps, use Streamlit Secrets. For local use, you can upload your Service Account JSON.")
        uploaded_sa_file = st.file_uploader("Upload Service Account JSON", type=['json'])
        if uploaded_sa_file:
            try:
                creds_json_content = uploaded_sa_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Could not read uploaded JSON: {e}")
                creds_json_content = None
    
    if st.button("Initialize Google AI Clients", disabled=st.session_state.clients_initialized):
        if gcp_project_id and gcp_location and creds_json_content:
            with st.spinner("Initializing clients..."):
                initialize_clients(gcp_project_id, gcp_location, creds_json_content)
        else:
            st.warning("Please provide Project ID, Location, and Service Account JSON content/upload.")

    if st.session_state.clients_initialized:
        st.success("✅ Clients Ready!")
    else:
        st.info("ⓘ Please provide credentials and initialize clients to proceed.")

# --- Main App Sections ---
if not st.session_state.clients_initialized:
    st.warning("🚦 Please initialize Google AI Clients using the sidebar to use the app.")
    st.stop()

# 1. Multi-Image Upload
st.header("1. Upload Thumbnail Images")
uploaded_files = st.file_uploader(
    "Choose images to analyze (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="thumbnail_uploader"
)

if uploaded_files:
    new_files_to_process = []
    processed_ids = [item['id'] for item in st.session_state.uploaded_image_analyses]
    for uploaded_file in uploaded_files:
        # Use a combination of name and size for a more robust ID if file_id is not always unique across reruns with same file
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in processed_ids:
            new_files_to_process.append(uploaded_file)

    if new_files_to_process:
        if st.button(f"Analyze {len(new_files_to_process)} New Image(s) ✨", key="analyze_button"):
            with st.spinner("Analyzing images with Gemini... this may take a few moments per image."):
                current_analyses = list(st.session_state.uploaded_image_analyses) # Make a mutable copy
                for uploaded_file in new_files_to_process:
                    file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
                    st.write(f"Processing: {uploaded_file.name}")
                    try:
                        img = Image.open(uploaded_file)
                        # Convert to PNG bytes for consistency (Gemini can handle various, but PNG is lossless)
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        image_bytes = img_byte_arr.getvalue()

                        analysis_data = analyze_image_and_create_prompts(image_bytes)
                        
                        if analysis_data:
                            current_analyses.append({
                                "id": file_identifier,
                                "name": uploaded_file.name,
                                "image_obj": img, 
                                "analysis_data": analysis_data
                            })
                            st.success(f"Analyzed: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to get analysis data for: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing file {uploaded_file.name}: {e}")
                
                st.session_state.uploaded_image_analyses = current_analyses
                # Rebuild selectable prompts
                st.session_state.all_selectable_prompts = []
                for analysis_item in st.session_state.uploaded_image_analyses:
                    if analysis_item.get('analysis_data'):
                        for category, data in analysis_item['analysis_data'].items():
                            prompt_text = data.get('prompt_suggestion')
                            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                                unique_prompt_id = f"{analysis_item['id']}_{category}" # Unique key for potential use
                                display_label = f"({analysis_item['name']}) {category}: {prompt_text}"
                                st.session_state.all_selectable_prompts.append(
                                    (unique_prompt_id, display_label, prompt_text, analysis_item['name'])
                                )
            st.rerun()

# Display Analyzed Images and their Breakdowns
if st.session_state.uploaded_image_analyses:
    st.header("📊 Analyzed Images & Breakdowns")
    for item in st.session_state.uploaded_image_analyses:
        with st.expander(f"View Analysis for: {item['name']}", expanded=False):
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(item['image_obj'], caption=item['name'], use_column_width="auto")
            with cols[1]:
                st.subheader(f"Breakdown for: {item['name']}")
                if item.get('analysis_data'):
                    for category, data in item['analysis_data'].items():
                        st.markdown(f"**{category}**:")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;*Analysis*: {data.get('analysis', 'N/A')}")
                        prompt_sugg = data.get('prompt_suggestion')
                        if prompt_sugg and prompt_sugg.strip():
                             st.markdown(f"&nbsp;&nbsp;&nbsp;*Prompt Idea*: `{prompt_sugg}`")
                else:
                    st.write("No analysis data available for this image.")
        st.divider()

# 3. User Selection of Prompts
if st.session_state.all_selectable_prompts:
    st.header("📝 Select Prompt Components for New Thumbnail")
    
    options = [item[1] for item in st.session_state.all_selectable_prompts] # Use display_label for options
    
    st.session_state.current_selected_labels = st.multiselect(
        "Choose prompt segments from the analyses above:",
        options=options,
        default=st.session_state.current_selected_labels, 
        key=f"prompt_multiselect_{len(st.session_state.all_selectable_prompts)}" 
    )
    
    selected_prompt_texts_for_generation = []
    for label in st.session_state.current_selected_labels:
        for item in st.session_state.all_selectable_prompts:
            if item[1] == label:
                selected_prompt_texts_for_generation.append(item[2]) # Add the actual prompt_text
                break
    
    if selected_prompt_texts_for_generation:
        st.subheader("Selected Prompt Segments:")
        for seg_idx, segment in enumerate(selected_prompt_texts_for_generation):
            st.markdown(f"- `{segment}`")

        current_prompt_text = ", ".join(selected_prompt_texts_for_generation)
        st.session_state.final_combined_prompt = st.text_area(
            "Combined Prompt (edit if needed):", 
            value=current_prompt_text, 
            height=150, 
            key="final_prompt_edit_area"
        )

        if st.button("🚀 Generate Thumbnail with Imagen", type="primary", key="generate_image_button"):
            if not st.session_state.final_combined_prompt.strip():
                st.error("The combined prompt is empty. Please select or write a prompt.")
            else:
                with st.spinner("Generating your new thumbnail with Imagen... This can take up to a minute!"):
                    generated_bytes = generate_image_with_imagen(st.session_state.final_combined_prompt)
                    if generated_bytes:
                        st.session_state.generated_image_bytes = generated_bytes
                        st.success("Thumbnail Generated!")
                    else:
                        st.session_state.generated_image_bytes = None # Clear previous if failed
                        st.error("Thumbnail generation failed. Check logs if any or try a different prompt.")
                st.rerun()

# Display Generated Image
if st.session_state.generated_image_bytes:
    st.header("🎉 Your Generated Thumbnail 🎉")
    st.image(st.session_state.generated_image_bytes, caption="Generated by Imagen on Vertex AI", use_column_width="auto")
    st.download_button(
        label="Download Thumbnail",
        data=st.session_state.generated_image_bytes,
        file_name="generated_thumbnail.png",
        mime="image/png",
        key="download_button"
    )

st.markdown("---")
st.caption("Built with Streamlit and Google Generative AI. For educational and experimental purposes.")
