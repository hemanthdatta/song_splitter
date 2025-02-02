from audio_analyzer import AudioAnalyzer
import os

def test_audio_analysis():
    # Initialize the analyzer
    print("Initializing Audio Analyzer...")
    analyzer = AudioAnalyzer()
    
    # Replace this with your audio file path
    audio_file = r"C:\Users\heman\Downloads\gen ai course\Udemy_Complete_Generative_AI_Course_With_Langchain_and_Huggingface_2024_9.part1_Downloadly.ir\genai-me\song_splitter\test_song.mp3"  # Put your audio file path here
    
    if not os.path.exists(audio_file):
        print(f"Please place an audio file at: {audio_file}")
        print("Or modify this script to point to your audio file location")
        return
    
    print(f"\nProcessing audio file: {audio_file}")
    try:
        # Process the audio file
        report_path = analyzer.process_audio(audio_file)
        print(f"\nProcessing complete!")
        print(f"Check the results in: {os.path.dirname(report_path)}")
        print("You will find:")
        print("1. Separated audio tracks in the 'separated' folder")
        print("2. Analysis report in 'analysis_report.json'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_audio_analysis() 