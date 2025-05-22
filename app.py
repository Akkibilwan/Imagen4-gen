# app.py
# Updated: May 22, 2025 - Includes fix for UnboundLocalError

import streamlit as st
from PIL import Image
import io
import json
import google.generativeai as genai
from google.oauth2 import service_account

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="Multi-Image Thumbnail Analyzer & Gemini Generator",
    page_icon="✨",
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
    div[data-testid="stSpinner"] > div { color: #007bff !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper: Credentials & Client Initialization ---
def initialize_gemini_client(creds_json_content):
    """Initializes the Google Gemini AI client."""
    try:
        credentials_info = json.loads(creds_json_content)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        genai.configure(credentials=credentials)
        
        analysis_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        try:
            # HYPOTHETICAL: Model for Image Generation via Gemini API
            image_generation_model = genai.GenerativeModel('gemini-1.5-pro-imagegen') # PURELY HYPOTHETICAL model name
            st.session_state.gemini_image_gen_model = image_generation_model
            st.info("Note: Using a hypothetical Gemini model for image generation.")
        except Exception as img_model_e:
            st.warning(f"Could not initialize hypothetical Gemini image generation model: {img_model_e}.")
            st.warning("Direct image generation feature might not be available or model name is incorrect.")
            st.session_state.gemini_image_gen_model = None

        st.session_state.clients_initialized = True
        st.session_state.gemini_analysis_model = analysis_model
        st.success("Google Gemini Client Initialized Successfully!")
        return True
    except json.JSONDecodeError:
        st.error("Error: The provided Google Credentials JSON is not valid.")
        st.session_state.clients_initialized = False
        return False
    except Exception as e:
        st.error(f"Error initializing Google Gemini Client: {e}")
        st.session_state.clients_initialized = False
        return False

# --- Core Functions ---
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
    if 'gemini_analysis_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Gemini analysis client not initialized.")
        return None
    
    model = st.session_state.gemini_analysis_model
    image_part = {"mime_type": "image/png", "data": image_bytes}
    
    response = None  # Initialize response to None here (FIX for UnboundLocalError)
    
    try:
        full_prompt = [BREAKDOWN_CATEGORIES_PROMPT_SYSTEM, image_part]
        response = model.generate_content(full_prompt) 
        
        if not response or not hasattr(response, 'text'):
            st.error("Gemini analysis call succeeded but returned an unexpected response structure.")
            try:
                st.error(f"Problematic Gemini response object: {response}")
            except Exception as log_e:
                st.error(f"Could not even log the problematic response: {log_e}")
            return None

        cleaned_response_text = response.text.strip()
        
        if not cleaned_response_text:
            st.error("Gemini analysis returned an empty text response.")
            return None

        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        analysis_result = json.loads(cleaned_response_text.strip())
        return analysis_result
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing Gemini's JSON response: {e}. Ensure the model is providing valid JSON.")
        st.error(f"Gemini raw response (if available): {getattr(response, 'text', 'Response object not available or lacks text attribute')}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini analysis: {e}")
        st.error(f"Gemini raw response (if available) at time of error: {getattr(response, 'text', 'Response object not available or lacks text attribute')}")
        return None

# HYPOTHETICAL: Image Generation with Gemini API
def generate_image_with_gemini(prompt_text):
    """
    Generates an image using the Gemini API (HYPOTHETICAL direct image generation).
    """
    if 'gemini_image_gen_model' not in st.session_state or not st.session_state.gemini_image_gen_model:
        st.error("Gemini image generation model not initialized or available. Cannot generate image.")
        st.error("This feature relies on direct image generation capabilities in the Gemini API, which may require specific SDK versions or model names if available.")
        return None

    model_instance = st.session_state.gemini_image_gen_model
    st.info(f"Attempting image generation with Gemini using prompt: '{prompt_text[:150]}...'")
    
    try:
        # This is a PURELY HYPOTHETICAL API call structure for image generation.
        # The actual method, parameters, and response would depend on Google's official implementation.
        # Replace this with the correct API call if/when it exists.
        image_generation_prompt = f"Generate a hyper-realistic YouTube thumbnail, 16:9 aspect ratio, based on this: {prompt_text}. The image should be high quality, photorealistic, vibrant, and eye-catching."
        
        # Example: response = model_instance.generate_image_content(prompt=image_generation_prompt, output_mime_type="image/png")
        # if response.image_bytes:
        #     return response.image_bytes

        st.error("Placeholder: Direct image generation with 'google-generativeai' SDK is not implemented in this example due to lack of a confirmed public API for it. This section needs the official method if/when Google releases it.")
        return None  # Return None as the placeholder
        
    except AttributeError as ae:
        st.error(f"SDK Error: The method for image generation might be incorrect or not available: {ae}")
        return None
    except Exception as e:
        st.error(f"Error during hypothetical Gemini image generation: {e}")
        return None

# --- Streamlit App UI & Logic ---
st.title("🖼️ YouTube Thumbnail Analyzer & Creative Prompt Generator (Gemini Focus)")
st.markdown("Upload images, get AI-powered analysis, and attempt to generate new thumbnails directly with Gemini API.")
st.markdown("**Disclaimer:** Direct image *generation* with the `google-generativeai` SDK is a feature assumed for this example (May 2025). If not officially supported by Google with the model and SDK version you are using, image generation will fail or not work as expected.")

if 'clients_initialized' not in st.session_state: st.session_state.clients_initialized = False
if 'uploaded_image_analyses' not in st.session_state: st.session_state.uploaded_image_analyses = [] 
if 'all_selectable_prompts' not in st.session_state: st.session_state.all_selectable_prompts = [] 
if 'current_selected_labels' not in st.session_state: st.session_state.current_selected_labels = []
if 'final_combined_prompt' not in st.session_state: st.session_state.final_combined_prompt = ""
if 'generated_image_bytes' not in st.session_state: st.session_state.generated_image_bytes = None

with st.sidebar:
    st.header("🔑 Google Cloud Setup")
    gcp_project_id = st.text_input("GCP Project ID", st.secrets.get("GCP_PROJECT_ID", ""), help="Your Google Cloud Project ID.")
    
    creds_json_content = None
    creds_json_str_secret = st.secrets.get("GOOGLE_CREDENTIALS_JSON_STR")
    if creds_json_str_secret:
        creds_json_content = creds_json_str_secret
        st.caption("Using credentials from Streamlit Secrets.")
    else:
        st.info("For deployed apps, use Streamlit Secrets. For local use, upload Service Account JSON.")
        uploaded_sa_file = st.file_uploader("Upload Service Account JSON", type=['json'])
        if uploaded_sa_file:
            try:
                creds_json_content = uploaded_sa_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Could not read uploaded JSON: {e}")
    
    if st.button("Initialize Gemini Client", disabled=st.session_state.clients_initialized):
        if gcp_project_id and creds_json_content:
            with st.spinner("Initializing Gemini client..."):
                initialize_gemini_client(creds_json_content)
        else:
            st.warning("Please provide GCP Project ID and Service Account JSON.")

    if st.session_state.clients_initialized:
        st.success("✅ Gemini Client Ready!")
    else:
        st.info("ⓘ Please provide credentials and initialize client.")

if not st.session_state.clients_initialized:
    st.warning("🚦 Please initialize the Gemini Client using the sidebar.")
    st.stop()

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
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in processed_ids:
            new_files_to_process.append(uploaded_file)

    if new_files_to_process:
        if st.button(f"Analyze {len(new_files_to_process)} New Image(s) ✨", key="analyze_button"):
            with st.spinner("Analyzing images with Gemini... this may take a few moments per image."):
                current_analyses = list(st.session_state.uploaded_image_analyses)
                for uploaded_file in new_files_to_process:
                    file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
                    st.write(f"Processing: {uploaded_file.name}")
                    try:
                        img = Image.open(uploaded_file)
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        image_bytes = img_byte_arr.getvalue()
                        analysis_data = analyze_image_and_create_prompts(image_bytes)
                        
                        if analysis_data:
                            current_analyses.append({
                                "id": file_identifier, "name": uploaded_file.name,
                                "image_obj": img, "analysis_data": analysis_data
                            })
                            st.success(f"Analyzed: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to get analysis data for: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing file {uploaded_file.name}: {e}")
                st.session_state.uploaded_image_analyses = current_analyses
                st.session_state.all_selectable_prompts = []
                for analysis_item in st.session_state.uploaded_image_analyses:
                    if analysis_item.get('analysis_data'):
                        for category, data in analysis_item['analysis_data'].items():
                            prompt_text = data.get('prompt_suggestion')
                            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                                unique_prompt_id = f"{analysis_item['id']}_{category}"
                                display_label = f"({analysis_item['name']}) {category}: {prompt_text}"
                                st.session_state.all_selectable_prompts.append(
                                    (unique_prompt_id, display_label, prompt_text, analysis_item['name'])
                                )
            st.rerun()

if st.session_state.uploaded_image_analyses:
    st.header("📊 Analyzed Images & Breakdowns")
    for item in st.session_state.uploaded_image_analyses:
        with st.expander(f"View Analysis for: {item['name']}", expanded=False):
            cols = st.columns([1, 2])
            with cols[0]: st.image(item['image_obj'], caption=item['name'], use_column_width="auto")
            with cols[1]:
                st.subheader(f"Breakdown for: {item['name']}")
                if item.get('analysis_data'):
                    for category, data in item['analysis_data'].items():
                        st.markdown(f"**{category}**:")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;*Analysis*: {data.get('analysis', 'N/A')}")
                        prompt_sugg = data.get('prompt_suggestion')
                        if prompt_sugg and prompt_sugg.strip():
                             st.markdown(f"&nbsp;&nbsp;&nbsp;*Prompt Idea*: `{prompt_sugg}`")
                else: st.write("No analysis data available.")
        st.divider()

if st.session_state.all_selectable_prompts:
    st.header("📝 Select Prompt Components for New Thumbnail")
    options = [item[1] for item in st.session_state.all_selectable_prompts]
    st.session_state.current_selected_labels = st.multiselect(
        "Choose prompt segments:", options=options,
        default=st.session_state.current_selected_labels, 
        key=f"prompt_multiselect_{len(st.session_state.all_selectable_prompts)}" 
    )
    selected_prompt_texts_for_generation = []
    for label in st.session_state.current_selected_labels:
        for item in st.session_state.all_selectable_prompts:
            if item[1] == label:
                selected_prompt_texts_for_generation.append(item[2])
                break
    
    if selected_prompt_texts_for_generation:
        st.subheader("Selected Prompt Segments:")
        for seg in selected_prompt_texts_for_generation: st.markdown(f"- `{seg}`")
        current_prompt_text = ", ".join(selected_prompt_texts_for_generation)
        st.session_state.final_combined_prompt = st.text_area(
            "Combined Prompt (edit if needed):", value=current_prompt_text, 
            height=150, key="final_prompt_edit_area"
        )

        if st.button("🚀 Generate Thumbnail with Gemini (Experimental)", type="primary", key="generate_gemini_image_button"):
            if not st.session_state.final_combined_prompt.strip():
                st.error("The combined prompt is empty.")
            else:
                with st.spinner("Attempting direct Gemini image generation... (Experimental Feature)"):
                    generated_bytes = generate_image_with_gemini(st.session_state.final_combined_prompt) 
                    if generated_bytes:
                        st.session_state.generated_image_bytes = generated_bytes
                        st.success("Image data received from Gemini (Experimental)!")
                    else:
                        st.session_state.generated_image_bytes = None
                        st.error("Gemini image generation did not return image data or failed. This is an experimental feature based on assumed SDK capabilities.")
                st.rerun()

if st.session_state.generated_image_bytes:
    st.header("🎉 Your Generated Thumbnail (via Gemini Experimental) 🎉")
    st.image(st.session_state.generated_image_bytes, caption="Generated by Gemini API (Experimental)", use_column_width="auto")
    st.download_button(
        label="Download Thumbnail", data=st.session_state.generated_image_bytes,
        file_name="gemini_generated_thumbnail.png", mime="image/png", key="download_gemini_button"
    )

st.markdown("---")
st.caption("Built with Streamlit and Google Generative AI. Image generation with Gemini direct API is experimental and based on assumed capabilities.")
