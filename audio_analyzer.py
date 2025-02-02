import os
import torch
import librosa
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, model_name='htdemucs', target_sr=44100, device=None):
        """
        Initialize the audio analyzer with the Demucs model.
        
        Parameters:
        - model_name: the name of the Demucs model to use.
        - target_sr: target sample rate for processing.
        - device: 'cpu' or 'cuda' (if None, will auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.target_sr = target_sr
        
        try:
            # Load Demucs model for source separation
            logger.info(f"Loading model: {model_name}")
            self.separator = get_model(model_name)
            self.separator.to(self.device)
            self.separator.eval()  # Set model to evaluation mode
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def separate_sources(self, audio_path, output_dir=None):
        """Separate audio into different stems and save them."""
        logger.info(f"Processing audio file: {audio_path}")
        
        if output_dir is None:
            output_dir = Path("separated")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        try:
            # Load audio using librosa at the target sample rate (load as stereo if available)
            logger.info("Loading audio file...")
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=False)
            logger.info(f"Loaded audio shape: {audio.shape}, Sample rate: {sr}")
            
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            logger.info(f"Audio shape after ensuring at least 2D: {audio.shape}")
        
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio, device=self.device)
            # If shape is (channels, samples) then add a batch dimension
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            logger.info(f"Audio tensor shape: {audio_tensor.shape}")
        
            # Apply source separation using Demucs
            logger.info("Applying source separation...")
            with torch.no_grad():
                sources = apply_model(self.separator, audio_tensor, progress=True)
            sources = sources.cpu().numpy()
            logger.info(f"Demucs raw output shape: {sources.shape}")
            
            # Remove the batch dimension if batch size is 1
            if sources.shape[0] == 1:
                sources = np.squeeze(sources, axis=0)
                logger.info(f"Squeezed sources shape: {sources.shape}")
                
            # Expected number of stems (typically 4: drums, bass, other, vocals)
            expected_stems = 4

            # Reorder axes according to the output shape
            if sources.ndim == 3:
                if sources.shape[0] == expected_stems:
                    # Already (stems, channels, samples)
                    sources_ordered = sources
                elif sources.shape[-1] == expected_stems:
                    # Convert from (channels, samples, stems) to (stems, channels, samples)
                    sources_ordered = np.moveaxis(sources, -1, 0)
                else:
                    raise ValueError(f"Unexpected sources array shape after squeeze: {sources.shape}")
            else:
                raise ValueError(f"Unexpected sources array shape: {sources.shape}")
            
            # Prepare each stem for writing (ensure shape is (samples, channels))
            processed_sources = []
            for i in range(sources_ordered.shape[0]):
                stem = sources_ordered[i]  # shape: (channels, samples)
                # Transpose to (samples, channels) if needed
                if stem.shape[0] < stem.shape[1]:
                    stem = stem.T
                processed_sources.append(stem)
            processed_sources = np.array(processed_sources)
            
            # Optional: scale amplitude if maximum amplitude is very low
            scale_factor = 1.0
            if processed_sources.max() < 0.001:
                scale_factor = 1000.0
                logger.info(f"Scaling sources by {scale_factor}")
            processed_sources *= scale_factor
            
            # Define stem names (adjust order if needed)
            stem_names = ['drums', 'bass', 'other', 'vocals']
            separated_paths = {}
            
            logger.info("Saving separated stems...")
            for source, name in zip(processed_sources, stem_names):
                out_path = output_dir / f"{name}.wav"
                # Write using 16-bit PCM to reduce file size
                sf.write(out_path, source, sr, subtype='PCM_16')
                separated_paths[name] = str(out_path)
                logger.info(f"Saved {name} stem to {out_path}")
        
            return separated_paths, sr
            
        except Exception as e:
            logger.error(f"Error in separate_sources: {e}", exc_info=True)
            raise

    def process_audio(self, audio_path):
        """Run the processing pipeline on the given audio file."""
        print(f"Processing audio file: {audio_path}")
        
        # Step 1: Source Separation
        separated_paths, sr = self.separate_sources(audio_path)
        
        print("Separated stems saved at:")
        for stem, path in separated_paths.items():
            print(f"{stem}: {path}")
        # Step 2: (Placeholder for additional analysis if needed)

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
