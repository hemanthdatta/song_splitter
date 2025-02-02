import os
import torch
import librosa
import numpy as np
import soundfile as sf
import logging
import gc
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, model_name='htdemucs', target_sr=44100, device=None, chunk_size=10):
        """
        Initialize the audio analyzer with the Demucs model.
        
        Parameters:
        - model_name: the name of the Demucs model to use.
        - target_sr: target sample rate for processing.
        - device: 'cpu' or 'cuda' (if None, will auto-detect)
        - chunk_size: size of audio chunks in seconds
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.target_sr = target_sr
        self.chunk_size = chunk_size
        
        try:
            # Load Demucs model for source separation
            logger.info(f"Loading model: {model_name}")
            self.separator = get_model(model_name)
            self.separator.to(self.device)
            self.separator.eval()  # Set model to evaluation mode
            logger.info("Model loaded successfully")
            
            # Clear any unused memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def process_in_chunks(self, audio, chunk_samples):
        """Process audio in chunks to save memory."""
        logger.info("Processing audio in chunks")
        total_samples = audio.shape[1]
        chunks = []
        
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            logger.info(f"Processing chunk {start//chunk_samples + 1}")
            
            # Extract chunk
            chunk = audio[:, start:end]
            chunk_tensor = torch.tensor(chunk, device=self.device).unsqueeze(0)
            
            # Process chunk
            with torch.no_grad():
                processed_chunk = apply_model(self.separator, chunk_tensor, progress=False)
            
            # Move to CPU and convert to numpy
            processed_chunk = processed_chunk.cpu().numpy()
            chunks.append(processed_chunk)
            
            # Clear memory
            del chunk_tensor, processed_chunk
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate chunks
        return np.concatenate(chunks, axis=2)

    def separate_sources(self, audio_path, output_dir=None):
        """Separate audio into different stems and save them."""
        logger.info(f"Processing audio file: {audio_path}")
        
        if output_dir is None:
            output_dir = Path("separated")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        try:
            # Load audio using librosa at the target sample rate
            logger.info("Loading audio file...")
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=False)
            logger.info(f"Loaded audio shape: {audio.shape}, Sample rate: {sr}")
            
            # Convert mono to stereo if needed
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            logger.info(f"Audio shape after ensuring stereo: {audio.shape}")
            
            # Calculate chunk size in samples
            chunk_samples = self.chunk_size * sr
            
            # Process audio in chunks
            sources = self.process_in_chunks(audio, chunk_samples)
            logger.info(f"Processed audio shape: {sources.shape}")
            
            # Remove the batch dimension if batch size is 1
            if sources.shape[0] == 1:
                sources = np.squeeze(sources, axis=0)
            
            # Define stem names
            stem_names = ['drums', 'bass', 'other', 'vocals']
            separated_paths = {}
            
            logger.info("Saving separated stems...")
            for idx, name in enumerate(stem_names):
                stem = sources[idx]  # shape: (channels, samples)
                if stem.shape[0] < stem.shape[1]:
                    stem = stem.T
                
                # Scale if needed
                max_val = np.abs(stem).max()
                if max_val > 0:
                    stem = stem / max_val * 0.9
                
                out_path = output_dir / f"{name}.wav"
                sf.write(out_path, stem, sr, subtype='PCM_16')
                separated_paths[name] = str(out_path)
                logger.info(f"Saved {name} stem to {out_path}")
                
                # Clear memory
                del stem
                gc.collect()
            
            return separated_paths, sr
            
        except Exception as e:
            logger.error(f"Error in separate_sources: {e}", exc_info=True)
            raise

    def process_audio(self, audio_path):
        """Run the processing pipeline on the given audio file."""
        logger.info(f"Processing audio file: {audio_path}")
        separated_paths, sr = self.separate_sources(audio_path)
        logger.info("Processing completed successfully")
        return separated_paths

def main():
    """Demonstrate usage of AudioAnalyzer."""
    analyzer = AudioAnalyzer(model_name='htdemucs', target_sr=22050)
    
    # Update with your audio file path.
    audio_path = "/kaggle/input/songew/test_song.mp3"
    if os.path.exists(audio_path):
        analyzer.process_audio(audio_path)
    else:
        print("Please provide a valid audio file path")

if __name__ == "__main__":
    main()
