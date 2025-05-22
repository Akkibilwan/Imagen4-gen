# app.py
# Updated: May 22, 2025
# Uses ONLY Gemini API (google-generativeai) for image analysis and 
# (hypothetically) for image generation. Credentials from Streamlit Secrets.

import streamlit as st
from PIL import Image
import io
import json
import google.generativeai as genai
from google.oauth2 import service_account # For parsing service account from secrets

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="Multi-Image Analyzer & Gemini Generator",
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
def initialize_gemini_client(creds_json_content_str):
    """Initializes the Google Gemini AI client using credentials from secrets."""
    try:
        # Parse the JSON string from secrets into a dictionary
        credentials_info = json.loads(creds_json_content_str)
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        # Configure the genai library with these credentials
        genai.configure(credentials=credentials)
        
        # Model for image analysis and text-based tasks
        # Using a known multimodal model from the Gemini family
        analysis_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-pro-vision' if preferred
        
        # --- HYPOTHETICAL: Model for Image Generation via Gemini API ---
        # This assumes a specific model name and capability exists in the SDK for direct image generation.
        # 'gemini-experimental-imagegen' is a PURELY HYPOTHETICAL model name.
        # You would need to replace this with the actual model name if Google releases such a feature.
        image_generation_model_name = 'gemini-experimental-imagegen' # Placeholder
        try:
            image_generation_model = genai.GenerativeModel(image_generation_model_name)
            st.session_state.gemini_image_gen_model = image_generation_model
            st.info(f"Note: Attempting to use hypothetical Gemini model '{image_generation_model_name}' for image generation.")
        except Exception as img_model_e:
            st.warning(f"Could not initialize hypothetical Gemini image generation model ('{image_generation_model_name}'): {img_model_e}")
            st.warning("Direct image generation may fail or not be supported by your SDK version/model.")
            st.session_state.gemini_image_gen_model = None
        # --- END HYPOTHETICAL ---

        st.session_state.clients_initialized = True
        st.session_state.gemini_analysis_model = analysis_model
        st.success("Google Gemini Client Initialized Successfully!")
        return True
    except json.JSONDecodeError:
        st.error("Error: The GOOGLE_CREDENTIALS_JSON_STR secret is not valid JSON.")
        st.session_state.clients_initialized = False
        return False
    except Exception as e:
        st.error(f"Error initializing Google Gemini Client: {e}")
        st.session_state.clients_initialized = False
        return False

# --- Core Functions ---
BREAKDOWN_CATEGORIES_PROMPT_SYSTEM = """
You are an expert image analyst. Your task is to analyze the provided image and break it down into distinct visual and conceptual components.
For each component category I list, provide:
1. 'analysis': A detailed description of that component in the image. Be specific about colors, shapes, textures, and any text content including its appearance (font, color, size if discernible).
2. 'prompt_suggestion': A concise, highly descriptive phrase (5-15 words) that could be used as part of a larger prompt to an image generation model to recreate *only that specific component* accurately. For example, if there's red text saying 'SALE!', the prompt suggestion might be 'Bold red "SALE!" text, sans-serif font, slight shadow'.

The categories for breakdown are:
- Main Subject: The primary focus (person, object, character). Include details like pose, expression, clothing.
- Action/Activity: What the main subject is doing or implying.
- Setting/Background: The environment or backdrop. Describe colors, textures, and key elements.
- Key Objects (excluding main subject): Other notable items, their appearance, and placement.
- Dominant Colors/Palette: List the main colors and describe the overall color scheme (e.g., vibrant, muted, monochrome with blue accents).
- Artistic Style: e.g., photorealistic, watercolor illustration, 3D cartoon render, flat graphic design, pixel art.
- Text Elements: For each piece of text, describe: content, font style (e.g., serif, sans-serif, script), color, approximate size relative to image, and any effects (e.g., outline, shadow, glow).
- Overall Vibe/Emotion: The feeling the image conveys (e.g., exciting and energetic, calm and serene, mysterious and intriguing).
- Compositional Focus: How elements are arranged, main focal point, rule of thirds, leading lines, etc.

Return your response ONLY as a valid JSON object where keys are the category names (e.g., "Main Subject")
and values are objects containing 'analysis' and 'prompt_suggestion'.
If a category is not applicable (e.g., no text), provide 'analysis': "Not applicable" and an empty string for 'prompt_suggestion'.
"""

def analyze_image_and_create_prompts(image_bytes):
    """Analyzes image with Gemini, breaks it down, and generates prompt suggestions for each component."""
    if 'gemini_analysis_model' not in st.session_state or not st.session_state.clients_initialized:
        st.error("Gemini analysis client not initialized.")
        return None
    
    model = st.session_state.gemini_analysis_model
    # Gemini SDK expects image parts in a specific format for multimodal input
    image_part = {"mime_type": "image/png", "data": image_bytes} # Assuming PNG, adjust if needed
    
    response = None  # Initialize for safe access in except block
    
    try:
        # Construct the prompt for the Gemini model
        full_prompt_parts = [BREAKDOWN_CATEGORIES_PROMPT_SYSTEM, image_part]
        
        response = model.generate_content(full_prompt_parts)
        
        if not response or not hasattr(response, 'text'):
            st.error("Gemini analysis call succeeded but returned an unexpected response structure.")
            try: st.error(f"Problematic Gemini response object: {response}")
            except Exception: pass
            return None

        cleaned_response_text = response.text.strip()
        if not cleaned_response_text:
            st.error("Gemini analysis returned an empty text response.")
            return None

        # Gemini can sometimes wrap JSON in ```json ... ```
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

# --- HYPOTHETICAL: Image Generation with Gemini API ---
def generate_image_with_gemini(prompt_text):
    """
    Generates an image using the Gemini API (HYPOTHETICAL direct image generation).
    This function assumes that the 'google-generativeai' SDK and a specific
    Gemini model support direct image generation. This is a placeholder.
    """
    if 'gemini_image_gen_model' not in st.session_state or not st.session_state.gemini_image_gen_model:
        st.error("Gemini image generation model not initialized or available. Cannot generate image.")
        st.error("This feature relies on direct image generation capabilities in the Gemini API, which may require specific SDK versions or model names if available.")
        return None

    model_instance = st.session_state.gemini_image_gen_model
    st.info(f"Attempting image generation with Gemini using prompt: '{prompt_text[:150]}...'")
    
    try:
        # --- This is a PURELY HYPOTHETICAL API call structure ---
        # The actual method name, parameters, and response structure would depend on
        # Google's official implementation if this feature exists.
        # You would need to replace this with the correct API call.
        
        # Example of what such a call *might* look like if the model directly outputs image bytes
        # or a structure containing them.
        # This is highly speculative.
        
        # response = model_instance.generate_content(
        #     f"Generate an image based on this detailed prompt: {prompt_text}",
        #     generation_config=genai.types.GenerationConfig(
        #         candidate_count=1
        #         # Hypothetical parameter to request image output, e.g., response_mime_type="image/png"
        #     )
        # )
        #
        # if response.parts and response.parts[0].inline_data and response.parts[0].inline_data.mime_type.startswith("image/"):
        #    return response.parts[0].inline_data.data # bytes

        st.error("CRITICAL: The 'generate_image_with_gemini' function is a placeholder.")
        st.error("Direct image generation with 'google-generativeai' SDK requires an official API and model.")
        st.error("Please consult the latest Google documentation for 'google-generativeai' for image generation capabilities and update this function accordingly.")
        # To prevent actual errors from this hypothetical function during testing of other parts:
        return None 
        
    except AttributeError as ae:
        st.error(f"SDK Error (Hypothetical Call): The method for image generation might be incorrect or not available: {ae}")
        st.error("This indicates the assumed API for direct Gemini image generation does not exist as implemented.")
        return None
    except Exception as e:
        st.error(f"Error during hypothetical Gemini image generation: {e}")
        return None
    # --- END HYPOTHETICAL ---

# --- Streamlit App UI & Logic ---
st.title("🖼️ Image Analyzer & Creative Prompt Generator (Gemini API)")
st.markdown("Upload images for detailed AI breakdown. Select components to build a new prompt, then (theoretically) generate a new image using Gemini.")
st.markdown("**Disclaimer:** Direct image *generation* with the `google-generativeai` SDK is a feature assumed for this example. If not officially supported by Google with the model and SDK version you are using, image generation will fail or not work as expected.")

# Session State Initializations
if 'clients_initialized' not in st.session_state: st.session_state.clients_initialized = False
if 'uploaded_image_analyses' not in st.session_state: st.session_state.uploaded_image_analyses = [] 
if 'all_selectable_prompts' not in st.session_state: st.session_state.all_selectable_prompts = [] 
if 'current_selected_labels' not in st.session_state: st.session_state.current_selected_labels = []
if 'final_combined_prompt' not in st.session_state: st.session_state.final_combined_prompt = ""
if 'generated_image_bytes' not in st.session_state: st.session_state.generated_image_bytes = None

# Sidebar for Credentials from Streamlit Secrets
with st.sidebar:
    st.header("🔑 Google Gemini API Setup")
    st.info("This app expects Google Cloud credentials to be set in Streamlit Secrets as `GOOGLE_CREDENTIALS_JSON_STR`.")

    # GCP Project ID might still be useful for context or if certain Gemini features are project-scoped
    gcp_project_id_secret = st.secrets.get("GCP_PROJECT_ID")
    if gcp_project_id_secret:
        st.caption(f"Using Project ID: {gcp_project_id_secret} (from secrets)")
    else:
        st.caption("GCP_PROJECT_ID not found in secrets (optional for some Gemini API uses).")

    creds_json_str = st.secrets.get("GOOGLE_CREDENTIALS_JSON_STR")

    if st.button("Initialize Gemini Client", disabled=st.session_state.clients_initialized):
        if creds_json_str:
            with st.spinner("Initializing Gemini client..."):
                initialize_gemini_client(creds_json_str)
        else:
            st.error("`GOOGLE_CREDENTIALS_JSON_STR` not found in Streamlit Secrets. Please configure it.")
            st.markdown("See [Streamlit Secrets Management](https://docs.streamlit.io/develop/concepts/connections/secrets-management) for help.")
            
    if st.session_state.clients_initialized:
        st.success("✅ Gemini Client Ready!")
    else:
        st.info("ⓘ Please initialize the client using your configured secrets.")

# Main App Flow
if not st.session_state.clients_initialized:
    st.warning("🚦 Please initialize the Gemini Client using the sidebar.")
    st.stop()

# 1. Multi-Image Upload
st.header("1. Upload Images for Analysis")
uploaded_files = st.file_uploader(
    "Choose images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="thumbnail_uploader"
)

if uploaded_files:
    new_files_to_process = []
    # Use a more robust way to track processed files if file_id is not persistent
    processed_file_ids = {item['id'] for item in st.session_state.uploaded_image_analyses}

    for uploaded_file in uploaded_files:
        # Create a unique identifier for the file based on name and size
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in processed_file_ids:
            new_files_to_process.append(uploaded_file)

    if new_files_to_process:
        if st.button(f"Analyze {len(new_files_to_process)} New Image(s) with Gemini ✨", key="analyze_button"):
            with st.spinner("Analyzing images with Gemini... this may take a few moments per image."):
                # Iterate over a copy if modifying the list, or build a new list
                current_analyses = list(st.session_state.uploaded_image_analyses)
                for uploaded_file in new_files_to_process:
                    file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
                    st.write(f"Processing: {uploaded_file.name}")
                    try:
                        img = Image.open(uploaded_file)
                        # Convert to PNG bytes for consistency
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        image_bytes = img_byte_arr.getvalue()

                        analysis_data = analyze_image_and_create_prompts(image_bytes)
                        
                        if analysis_data:
                            current_analyses.append({
                                "id": file_identifier, # Use the unique identifier
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
                # Rebuild selectable prompts from all analyses
                st.session_state.all_selectable_prompts = []
                for analysis_item in st.session_state.uploaded_image_analyses:
                    if analysis_item.get('analysis_data'):
                        for category, data in analysis_item['analysis_data'].items():
                            prompt_text = data.get('prompt_suggestion')
                            # Ensure prompt_text is valid and not just whitespace
                            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                                # Create a unique ID for each prompt suggestion for the multiselect key
                                unique_prompt_id = f"{analysis_item['id']}_{category.replace(' ', '_')}"
                                display_label = f"({analysis_item['name']}) {category}: {prompt_text}"
                                st.session_state.all_selectable_prompts.append(
                                    (unique_prompt_id, display_label, prompt_text, analysis_item['name'])
                                )
            st.rerun() # Rerun to update UI with new analyses and selectable prompts

# Display Analyzed Images & Breakdowns
if st.session_state.uploaded_image_analyses:
    st.header("📊 Analyzed Images & Prompt Ideas (from Gemini)")
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
                        if prompt_sugg and prompt_sugg.strip(): # Check if prompt suggestion is not empty
                             st.markdown(f"&nbsp;&nbsp;&nbsp;*Prompt Idea*: `{prompt_sugg}`")
                else:
                    st.write("No analysis data available for this image.")
        st.divider()

# 3. User Selection of Prompts & Image Generation
if st.session_state.all_selectable_prompts:
    st.header("📝 Select Prompt Components & Generate New Image")
    
    # Options for multiselect are the display_labels
    options = [item[1] for item in st.session_state.all_selectable_prompts]
    
    # Use st.session_state to preserve multiselect state across reruns
    st.session_state.current_selected_labels = st.multiselect(
        "Choose prompt segments from the analyses above:",
        options=options,
        default=st.session_state.current_selected_labels, 
        key=f"prompt_multiselect_{len(st.session_state.all_selectable_prompts)}" # Dynamic key if options change
    )
    
    selected_prompt_texts_for_generation = []
    for label in st.session_state.current_selected_labels:
        for item in st.session_state.all_selectable_prompts:
            if item[1] == label: # Match based on display_label
                selected_prompt_texts_for_generation.append(item[2]) # Add the actual prompt_text
                break
    
    if selected_prompt_texts_for_generation:
        st.subheader("Selected Prompt Segments:")
        for seg_idx, segment in enumerate(selected_prompt_texts_for_generation):
            st.markdown(f"- `{segment}`")

        # Combine selected segments into a single prompt string
        current_prompt_text = ", ".join(selected_prompt_texts_for_generation)
        # Allow user to edit the final combined prompt
        st.session_state.final_combined_prompt = st.text_area(
            "Combined Prompt for Image Generation (edit if needed):", 
            value=current_prompt_text, 
            height=150, 
            key="final_prompt_edit_area" # Ensure key is consistent
        )

        if st.button("🚀 Generate Image with Gemini (Experimental)", type="primary", key="generate_gemini_image_button"):
            if not st.session_state.final_combined_prompt.strip():
                st.error("The combined prompt is empty. Please select or write a prompt.")
            else:
                with st.spinner("Attempting direct Gemini image generation... (Experimental Feature)"):
                    generated_bytes = generate_image_with_gemini(st.session_state.final_combined_prompt) 
                    if generated_bytes:
                        st.session_state.generated_image_bytes = generated_bytes
                        st.success("Image data (hypothetically) received from Gemini!")
                    else:
                        st.session_state.generated_image_bytes = None # Clear previous if failed
                        st.error("Gemini image generation did not return image data or failed. This is an experimental/placeholder feature.")
                st.rerun() # Rerun to display the image or error

# Display Generated Image
if st.session_state.generated_image_bytes:
    st.header("🎉 Your Generated Image (via Gemini Experimental) 🎉")
    st.image(st.session_state.generated_image_bytes, caption="Generated by Gemini API (Experimental)", use_column_width="auto")
    st.download_button(
        label="Download Image",
        data=st.session_state.generated_image_bytes,
        file_name="gemini_generated_image.png",
        mime="image/png",
        key="download_gemini_button"
    )

st.markdown("---")
st.caption("Built with Streamlit and Google Generative AI. Image generation with Gemini direct API is experimental and based on assumed capabilities.")
