import os
import uuid
import shutil
import logging
import gc
import psutil
from pathlib import Path
from flask import Flask, request, render_template, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from audio_analyzer import AudioAnalyzer

app = Flask(__name__)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage - RSS: {mem_info.rss / 1024 / 1024:.2f}MB, VMS: {mem_info.vms / 1024 / 1024:.2f}MB")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
BASE_DIR = Path(os.getcwd())

# Use environment variable for production paths
if os.environ.get('RENDER'):
    # On Render, use a directory we definitely have permission for
    STORAGE_DIR = Path('/opt/render/project/src/data')
    UPLOAD_FOLDER = STORAGE_DIR / 'uploads'
    RESULTS_FOLDER = STORAGE_DIR / 'results'
    STATIC_FOLDER = STORAGE_DIR / 'static'
    app.static_folder = str(STATIC_FOLDER)
    
    # Log environment info
    logger.info(f"Running on Render")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory listing: {os.listdir('.')}")
    logger.info(f"Storage directory: {STORAGE_DIR}")
    log_memory_usage()
else:
    # Local development paths
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'static' / 'results'
    STATIC_FOLDER = BASE_DIR / 'static'

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

def create_directory(directory):
    """Create directory and verify it's writable."""
    try:
        directory.mkdir(exist_ok=True, parents=True, mode=0o755)
        logger.info(f"Created directory: {directory}")
        
        # Verify directory is writable
        test_file = directory / '.test_write'
        test_file.touch()
        test_file.unlink()
        logger.info(f"Successfully verified write permissions for: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error with directory {directory}: {e}", exc_info=True)
        return False

# Ensure directories exist with proper permissions
try:
    directories_ok = all(create_directory(d) for d in [STORAGE_DIR, UPLOAD_FOLDER, RESULTS_FOLDER, STATIC_FOLDER])
    if not directories_ok:
        raise Exception("Failed to create or verify one or more directories")
except Exception as e:
    logger.error(f"Error setting up directories: {e}", exc_info=True)
    raise

# Initialize the audio analyzer with explicit CPU device and chunk processing
try:
    analyzer = AudioAnalyzer(model_name='htdemucs', target_sr=44100, device='cpu', chunk_size=5)
    logger.info("Successfully initialized AudioAnalyzer")
except Exception as e:
    logger.error(f"Error initializing AudioAnalyzer: {e}", exc_info=True)
    raise

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up files older than 24 hours."""
    import time
    current_time = time.time()
    max_age = 86400  # 24 hours
    
    try:
        # Cleanup uploads
        for file_path in UPLOAD_FOLDER.glob('*'):
            if current_time - file_path.stat().st_mtime > max_age:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old upload: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Cleanup results
        for dir_path in RESULTS_FOLDER.glob('*'):
            if dir_path.is_dir() and current_time - dir_path.stat().st_mtime > max_age:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up old results: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting {dir_path}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.route('/')
def index():
    """Render the upload page."""
    cleanup_old_files()  # Clean up old files on page load
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    logger.info("Starting file upload process")
    log_memory_usage()
    
    try:
        if 'audio' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        logger.info(f"Saving uploaded file to: {upload_path}")
        
        # Ensure upload directory exists
        if not create_directory(UPLOAD_FOLDER):
            raise Exception("Failed to create upload directory")
        
        # Save the file
        file.save(upload_path)
        logger.info(f"File saved successfully: {upload_path}")
        file_size = upload_path.stat().st_size
        logger.info(f"File size: {file_size / (1024*1024):.2f}MB")
        
        # Create unique output directory
        output_dir = RESULTS_FOLDER / str(uuid.uuid4())
        if not create_directory(output_dir):
            raise Exception("Failed to create output directory")
        
        # Process the audio file
        logger.info("Starting audio processing")
        log_memory_usage()
        
        # Clear memory before processing
        gc.collect()
        
        separated_paths, sr = analyzer.separate_sources(str(upload_path), output_dir=str(output_dir))
        logger.info("Audio processing completed")
        log_memory_usage()
        
        # Convert filesystem paths to URL paths
        result_paths = {}
        for stem, path in separated_paths.items():
            if os.environ.get('RENDER'):
                rel_path = os.path.relpath(Path(path), STATIC_FOLDER)
                result_paths[stem] = url_for('static', filename=rel_path)
            else:
                rel_path = os.path.relpath(path, BASE_DIR / 'static')
                result_paths[stem] = url_for('static', filename=rel_path)
        
        # Clean up the uploaded file
        upload_path.unlink()
        logger.info("Cleaned up uploaded file")
        
        # Clear memory after processing
        gc.collect()
        log_memory_usage()
        
        return jsonify({
            'success': True,
            'redirect': url_for('results', result_id=output_dir.name)
        })
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        log_memory_usage()
        # Clean up any uploaded/processed files
        if 'upload_path' in locals():
            try:
                upload_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up upload: {cleanup_error}")
        if 'output_dir' in locals():
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up output: {cleanup_error}")
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

@app.route('/results/<result_id>')
def results(result_id):
    """Display the results page."""
    output_dir = RESULTS_FOLDER / result_id
    if not output_dir.exists():
        logger.error(f"Results not found for ID: {result_id}")
        return "Results not found", 404
    
    try:
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
        
        logger.info(f"Found stems for result {result_id}: {list(stems.keys())}")
        return render_template('results.html', stems=stems)
    except Exception as e:
        logger.error(f"Error serving results: {e}", exc_info=True)
        return "Error serving results", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    try:
        if os.environ.get('RENDER'):
            return send_from_directory(STATIC_FOLDER, filename)
        return app.send_static_file(filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}", exc_info=True)
        return "Error serving file", 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.error("File too large error")
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

if __name__ == '__main__':
    # For production, use gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 