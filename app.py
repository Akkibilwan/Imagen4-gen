import streamlit as st
import os
import io
import json
import base64
import requests
from PIL import Image
import openai
import google.generativeai as genai

# --- Configuration & Credentials ---
st.set_page_config(
    page_title="Multi-Thumbnail Analyzer & Prompt Generator",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS (Dark Mode - similar to YouTube)
st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1, h2, h3, h4, h5, h6 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p, li, div, label { color: #aaaaaa; }
    .stButton>button {
        background-color: #303030;
        color: #f1f1f1;
        border: 1px solid #505050;
    }
    .stButton>button:hover {
        background-color: #505050;
        border: 1px solid #707070;
    }
    .stFileUploader label, .stTextInput label, .stTextArea label {
        color: #f1f1f1 !important;
    }
    .uploaded-image-container {
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #181818;
    }
    .prompt-section {
        background-color: #202020;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .prompt-section h4 {
        margin-top: 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def setup_clients():
    """Sets up and returns OpenAI and Gemini clients."""
    openai_client_instance = None
    gemini_model_instance = None

    # OpenAI API
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", key="openai_key_input")
    elif "OPENAI_API_KEY" in os.environ:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key:
        try:
            openai.api_key = openai_api_key
            # Check if using new OpenAI client structure or old
            if hasattr(openai, 'OpenAI'):
                openai_client_instance = openai.OpenAI(api_key=openai_api_key)
            else: # Fallback for older openai library version
                openai_client_instance = openai
        except Exception as e:
            st.sidebar.error(f"OpenAI Initialization Error: {e}")
    else:
        st.sidebar.warning("OpenAI API Key not found. Image analysis will be unavailable.")

    # Gemini API
    gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not gemini_api_key and "GOOGLE_API_KEY" not in os.environ:
        gemini_api_key = st.sidebar.text_input("Enter your Google Gemini API key:", type="password", key="gemini_key_input")
    elif "GOOGLE_API_KEY" in os.environ:
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-pro'
        except Exception as e:
            st.sidebar.error(f"Gemini Initialization Error: {e}")
    else:
        st.sidebar.warning("Google Gemini API Key not found. Prompt breakdown will be unavailable.")

    return openai_client_instance, gemini_model_instance

# --- Image Processing Functions ---
def encode_image_to_base64(image_bytes):
    """Encodes image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

# --- AI Analysis Functions ---
def analyze_image_with_openai(client, image_base64, filename="image"):
    """
    Analyzes an image using OpenAI's GPT-4 Vision model.
    Returns a textual description of the image.
    """
    if not client:
        return "OpenAI client not initialized."
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze this YouTube thumbnail image named '{filename}' in detail. Describe its main subject, background, style, colors, any text present, and overall composition and mood. Focus on elements that would be important for recreating it."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'): # New OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o", # Or gpt-4-vision-preview
                messages=messages,
                max_tokens=600
            )
            return response.choices[0].message.content
        else: # Old OpenAI client
             response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=600
            )
             return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error analyzing image '{filename}' with OpenAI: {e}")
        return f"Unable to analyze '{filename}' with OpenAI."

def generate_breakdown_prompts_with_gemini(model, openai_analysis, filename="image"):
    """
    Takes OpenAI's analysis and uses Gemini to generate specific breakdown prompts.
    Returns a dictionary of prompts.
    """
    if not model:
        return {"error": "Gemini model not initialized."}

    prompt_system = """
You are an expert prompt engineer. Based on the provided image analysis,
generate distinct, descriptive text prompts for an AI image generator (like DALL-E, Midjourney, or Stable Diffusion)
that would help recreate specific aspects of the original image.

For each category listed below, provide a concise and actionable prompt.
If a category is not relevant or information is insufficient from the analysis, state "Not clearly applicable" or provide a best-guess prompt.
Output the result as a JSON object with the following keys:
"overall_scene_composition", "main_subject_details", "background_elements",
"color_palette_mood", "lighting_style", "artistic_style_medium", "text_elements_style".
Each value should be a string containing the generated prompt for that category.

Example for a single category if the analysis was "A cat wearing a hat":
"main_subject_details": "Photorealistic close-up of a fluffy ginger cat wearing a tiny blue party hat, looking curious."
"""

    user_prompt = f"""
Image Analysis for '{filename}':
---
{openai_analysis}
---

Generate the breakdown prompts in the specified JSON format.
"""
    try:
        full_prompt = [
            {"role": "user", "parts": [prompt_system]},
            {"role": "model", "parts": ["Okay, I understand. I will take the image analysis and generate a JSON object with specific prompts for each category."]}, # Few-shot priming
            {"role": "user", "parts": [user_prompt]}
        ]
        response = model.generate_content(full_prompt)
        
        # Clean the response to extract JSON part
        text_response = response.text.strip()
        # Find the start and end of the JSON block
        json_start = text_response.find('{')
        json_end = text_response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = text_response[json_start:json_end]
            return json.loads(json_str)
        else:
            st.warning(f"Could not parse JSON from Gemini response for {filename}. Response was: {text_response}")
            return {"error": f"Could not parse JSON from Gemini for {filename}."}

    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error for '{filename}' from Gemini: {e}. Response text: {response.text}")
        return {"error": f"JSON decoding error for {filename}."}
    except Exception as e:
        st.error(f"Error generating prompts for '{filename}' with Gemini: {e}")
        return {"error": f"Unable to generate prompts for '{filename}' with Gemini."}

# --- Main Application ---
def main():
    st.title("🖼️ Multi-Thumbnail Analyzer & Prompt Generator")
    st.markdown("Upload one or more thumbnail images. The app will use OpenAI to analyze them, and then Gemini to generate breakdown prompts for recreating similar visuals.")

    openai_client, gemini_model = setup_clients()

    # Initialize session state
    if 'image_analyses' not in st.session_state:
        st.session_state.image_analyses = [] # List of dicts: {filename, bytes, b64, openai_desc, gemini_prompts}

    uploaded_files = st.file_uploader(
        "Upload thumbnail images (JPG, PNG)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        new_files_to_process = []
        for uploaded_file in uploaded_files:
            # Check if this file (by name and size) has already been processed to avoid duplicates on re-upload
            is_new = True
            for existing_analysis in st.session_state.image_analyses:
                if existing_analysis['filename'] == uploaded_file.name and len(existing_analysis['bytes']) == uploaded_file.size:
                    is_new = False
                    break
            if is_new:
                new_files_to_process.append(uploaded_file)

        if new_files_to_process:
            st.info(f"Found {len(new_files_to_process)} new images to add to the processing queue.")

        if st.button(f"Process {len(new_files_to_process) if new_files_to_process else len(st.session_state.image_analyses)} Image(s)", key="analyze_button", disabled=not (openai_client and gemini_model)):
            if not openai_client:
                st.error("OpenAI client is not initialized. Please check your API key.")
            if not gemini_model:
                st.error("Gemini model is not initialized. Please check your API key.")
            
            if openai_client and gemini_model:
                current_analyses = [] # Store analyses for this run
                
                # If new files were just uploaded, prioritize them
                files_for_processing_run = new_files_to_process if new_files_to_process else \
                                        [f for f in uploaded_files if not any(a['filename'] == f.name for a in st.session_state.image_analyses)]


                with st.spinner("Analyzing images... This may take a while for multiple images."):
                    for uploaded_file in files_for_processing_run:
                        image_bytes = uploaded_file.getvalue()
                        image_base64 = encode_image_to_base64(image_bytes)
                        filename = uploaded_file.name
                        st.write(f"Processing: {filename}")

                        # 1. Analyze with OpenAI
                        openai_desc = analyze_image_with_openai(openai_client, image_base64, filename)
                        if "Unable to analyze" in openai_desc or "OpenAI client not initialized" in openai_desc :
                             st.warning(f"Skipping Gemini for {filename} due to OpenAI analysis issue.")
                             gemini_prompts = {"error": "Skipped due to OpenAI analysis failure."}
                        else:
                            # 2. Generate breakdown prompts with Gemini
                            gemini_prompts = generate_breakdown_prompts_with_gemini(gemini_model, openai_desc, filename)

                        current_analyses.append({
                            "filename": filename,
                            "bytes": image_bytes,
                            "b64": image_base64, # Storing b64 can consume memory, consider if needed long term
                            "openai_desc": openai_desc,
                            "gemini_prompts": gemini_prompts
                        })
                
                # Add new results to session state, avoiding duplicates if any slipped through
                for new_analysis in current_analyses:
                    if not any(existing['filename'] == new_analysis['filename'] for existing in st.session_state.image_analyses):
                        st.session_state.image_analyses.append(new_analysis)
                
                st.success("Analysis complete!")
                # Clear the uploader after processing to avoid reprocessing the same files by default
                # This can be tricky if the user wants to re-analyze after changing API keys, for example
                # For now, let's clear it. User can re-upload if needed.
                # st.experimental_rerun() # This might be too aggressive, let's see


    if st.session_state.image_analyses:
        st.markdown("---")
        st.subheader("Analysis Results & Breakdown Prompts")

        if st.button("Clear All Results", key="clear_results"):
            st.session_state.image_analyses = []
            st.rerun()


        for i, analysis_data in enumerate(st.session_state.image_analyses):
            with st.container():
                st.markdown(f"<div class='uploaded-image-container'>", unsafe_allow_html=True)
                st.subheader(f"🖼️ {analysis_data['filename']}")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(analysis_data['bytes'], use_column_width=True, caption="Uploaded Image")

                with col2:
                    st.markdown("#### 📝 OpenAI Vision Analysis:")
                    with st.expander("Show/Hide OpenAI Analysis", expanded=False):
                        st.markdown(f"<small>{analysis_data['openai_desc']}</small>", unsafe_allow_html=True)

                    st.markdown("#### ✨ Gemini Breakdown Prompts (for GPT/DALL-E etc.):")
                    if "error" in analysis_data["gemini_prompts"]:
                        st.error(f"Could not generate prompts: {analysis_data['gemini_prompts']['error']}")
                    elif isinstance(analysis_data["gemini_prompts"], dict):
                        for category, prompt_text in analysis_data["gemini_prompts"].items():
                            clean_category_name = category.replace("_", " ").title()
                            st.markdown(f"<div class='prompt-section'><h4>{clean_category_name}</h4>", unsafe_allow_html=True)
                            st.code(prompt_text, language=None) # `None` for plain text, allows easy copy
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Gemini prompts are not in the expected format.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("Upload some images to get started.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses AI to analyze YouTube thumbnail images (or any image) "
        "and generate detailed prompts for recreating similar visuals with text-to-image models."
    )

if __name__ == "__main__":
    main()
