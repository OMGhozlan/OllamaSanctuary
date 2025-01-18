import os
import base64
import subprocess
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from pathlib import Path
import ollama
from difflib import get_close_matches

app = Flask(__name__)

def is_ollama_running():
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        return 'ollama.exe' in result.stdout
    except Exception as e:
        return False

def start_ollama():
    try:
        subprocess.Popen(['ollama', 'serve'])
    except Exception as e:
        raise RuntimeError(f"Error starting Ollama: {e}")

def encode_image_to_base64(filepath):
    """Encode an image to a base64 string.

    Args:
        filepath: Path to the image file.

    Returns:
        Base64-encoded image string.
    """
    try:
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"Error encoding image {filepath}: {e}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    ollama_client = ollama.Client()
    model_list = ollama_client.list()['models']
    model_names = [model['name'] for model in model_list]
    return jsonify(model_names)

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory('uploads', filename)  # Serving from the root folder

@app.route('/generate_captions', methods=['POST'])
def generate_captions():
    if not is_ollama_running():
        start_ollama()

    model_name = request.form['model']
    prepend_text = request.form['prependText']
    images = request.files.getlist('images')
    uploads_path = Path('uploads')
    uploads_path.mkdir(parents=True, exist_ok=True)

    system_prompt_path = Path(__file__).parent / "System.prompt"
    if system_prompt_path.is_file():
        try:
            with open(system_prompt_path, 'r') as file:
                prompt = file.read()
        except Exception as e:
            return jsonify({"error": f"Error reading System.prompt file: {e}"}), 500
    else:
        prompt = request.form.get('prompt', '')

    ollama_client = ollama.Client()
    validate_model(model_name, ollama_client)

    captions = []
    for image in images:
        image_path = uploads_path / image.filename
        image.save(image_path)
        caption = process_image(image_path, prompt, model_name, ollama_client, prepend_text)
        captions.append({
            "file_name": image.filename,
            "image_url": url_for('serve_image', filename=image.filename),
            "caption": caption
        })

    return jsonify({"captions": captions})

def validate_model(model_name, ollama_client):
    """Validate if the specified model exists.

    Args:
        model_name: Name of the model to validate.
        ollama_client: Initialized Ollama client.

    Returns:
        None. Raises an error if the model does not exist.
    """
    model_list = ollama_client.list()['models']
    model_names = [model['name'] for model in model_list]

    if model_name not in model_names:
        suggestions = get_close_matches(model_name, model_names, n=3)
        suggestion_message = (f"Model '{model_name}' not found. Did you mean: "
                              f"{', '.join(suggestions)}?" if suggestions else "No similar models found.")
        raise ValueError(suggestion_message)

def process_image(image_path, prompt, model_name, ollama_client, prepend_text):
    image_base64 = encode_image_to_base64(image_path)
    try:
        response = ollama_client.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt, 'images': [image_base64]}])
        response = response['message']['content']
        return f"{prepend_text} {response}"
    except ollama.ResponseError as e:
        return f"LLaVa model error: {e}"

if __name__ == "__main__":
    app.run(debug=True)