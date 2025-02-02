# Song Splitter Web App

A web application that uses the Demucs model to split songs into their individual stems (drums, bass, vocals, and other). Users can upload audio files through a modern web interface and download or listen to the separated tracks.

## Features

- Drag and drop file upload
- Real-time processing progress indication
- In-browser audio playback
- Individual stem download options
- Support for multiple audio formats (MP3, WAV, M4A, FLAC)
- Modern, responsive web interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/song-splitter.git
cd song-splitter
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Development

1. Set up environment variables (create a `.env` file):
```
FLASK_APP=app.py
FLASK_ENV=development
```

2. Run the development server:
```bash
flask run
```

The app will be available at `http://localhost:5000`

## Cloud Deployment

### Deploying to Heroku

1. Install Heroku CLI and login:
```bash
heroku login
```

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Add Heroku-specific files:
   - Create `Procfile` with: `web: gunicorn app:app`
   - Update `requirements.txt` with gunicorn

4. Deploy:
```bash
git push heroku main
```

### Deploying to Google Cloud Platform

1. Install Google Cloud SDK
2. Initialize project:
```bash
gcloud init
```

3. Deploy to App Engine:
```bash
gcloud app deploy
```

## Project Structure

```
song_splitter/
├── app.py              # Flask application
├── audio_analyzer.py   # Audio processing logic
├── requirements.txt    # Project dependencies
├── static/            # Static files (CSS, JS, results)
├── templates/         # HTML templates
├── uploads/          # Temporary upload directory
└── tests/            # Test files
```

## API Reference

### POST /upload
Uploads and processes an audio file.

Request:
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file

Response:
- 200: JSON with paths to separated stems
- 400: Invalid file type or no file
- 500: Processing error

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 