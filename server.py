from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys

# Ensure poc/ folder is in path so we can import the pipeline
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "poc"))
import poc_pipeline

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        try:
            # Secure the filename to prevent OS Error [Errno 22] on Windows
            filename = secure_filename(file.filename)
            if not filename:
                filename = "uploaded_image.jpg"
                
            # Save image temporarily
            temp_dir = os.path.join(script_dir, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            # Run the OCR pipeline
            print(f"Running pipeline on {temp_path}...")
            result = poc_pipeline.run_pipeline(temp_path)
            
            # result is a dict with "table" and "grid_image"
            return jsonify(result)
        except Exception as e:
            import traceback
            print("Pipeline Error:")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
