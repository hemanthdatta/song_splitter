import os
import uuid
import shutil
from pathlib import Path
from flask import Flask, request, render_template, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from audio_analyzer import AudioAnalyzer

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
BASE_DIR = Path(os.getcwd())

# Use environment variable for production paths
if os.environ.get('RENDER'):
    # On Render, use the specific storage path we have permission for
    STORAGE_DIR = Path('/opt/render/project/src/storage')
    UPLOAD_FOLDER = STORAGE_DIR / 'uploads'
    RESULTS_FOLDER = STORAGE_DIR / 'results'
    STATIC_FOLDER = STORAGE_DIR / 'static'
    app.static_folder = str(STATIC_FOLDER)
else:
    # Local development paths
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'static' / 'results'
    STATIC_FOLDER = BASE_DIR / 'static'

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
STATIC_FOLDER.mkdir(exist_ok=True, parents=True)

# Initialize the audio analyzer
analyzer = AudioAnalyzer(model_name='htdemucs', target_sr=44100)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up files older than 24 hours."""
    import time
    current_time = time.time()
    max_age = 86400  # 24 hours
    
    # Cleanup uploads
    for file_path in UPLOAD_FOLDER.glob('*'):
        if current_time - file_path.stat().st_mtime > max_age:
            try:
                file_path.unlink()
            except Exception as e:
                app.logger.error(f"Error deleting {file_path}: {e}")
    
    # Cleanup results
    for dir_path in RESULTS_FOLDER.glob('*'):
        if dir_path.is_dir() and current_time - dir_path.stat().st_mtime > max_age:
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                app.logger.error(f"Error deleting {dir_path}: {e}")

@app.route('/')
def index():
    """Render the upload page."""
    cleanup_old_files()  # Clean up old files on page load
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)
        
        # Create unique output directory
        output_dir = RESULTS_FOLDER / str(uuid.uuid4())
        output_dir.mkdir(exist_ok=True)
        
        # Process the audio file
        separated_paths, sr = analyzer.separate_sources(str(upload_path), output_dir=str(output_dir))
        
        # Convert filesystem paths to URL paths
        result_paths = {}
        for stem, path in separated_paths.items():
            if os.environ.get('RENDER'):
                # For Render, serve through the static endpoint
                rel_path = os.path.relpath(Path(path), STATIC_FOLDER)
                result_paths[stem] = url_for('static', filename=rel_path)
            else:
                rel_path = os.path.relpath(path, BASE_DIR / 'static')
                result_paths[stem] = url_for('static', filename=rel_path)
        
        # Clean up the uploaded file
        upload_path.unlink()
        
        return jsonify({
            'success': True,
            'redirect': url_for('results', result_id=output_dir.name)
        })
    
    except Exception as e:
        app.logger.error(f"Error processing file: {e}")
        # Clean up any uploaded/processed files
        if 'upload_path' in locals():
            upload_path.unlink(missing_ok=True)
        if 'output_dir' in locals():
            shutil.rmtree(output_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

@app.route('/results/<result_id>')
def results(result_id):
    """Display the results page."""
    output_dir = RESULTS_FOLDER / result_id
    if not output_dir.exists():
        return "Results not found", 404
    
    # Get all WAV files in the output directory
    stems = {}
    for wav_file in output_dir.glob('*.wav'):
        stem_name = wav_file.stem
        if os.environ.get('RENDER'):
            # For Render, serve through the static endpoint
            rel_path = os.path.relpath(wav_file, STATIC_FOLDER)
            stems[stem_name] = url_for('static', filename=rel_path)
        else:
            rel_path = os.path.relpath(wav_file, BASE_DIR / 'static')
            stems[stem_name] = url_for('static', filename=rel_path)
    
    return render_template('results.html', stems=stems)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    if os.environ.get('RENDER'):
        return send_from_directory(STATIC_FOLDER, filename)
    return app.send_static_file(filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

if __name__ == '__main__':
    # For production, use gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 