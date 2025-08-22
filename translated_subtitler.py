#This file will take in a Video and will create translated subtitles.
# ====== PARAMETERS =========================================================================================================== #
INPUT_FILENAME = "test.mp4"                             # Input video file

#Select at least one Action
BURN_INTO_VIDEO = False                                 # Create a new video file with subtitles burned in
SAVE_ASS_FILE = True                                    # Save the .ass subtitle file
SAVE_SRT_FILE = True                                    # Save the .srt subtitle file
PRINT_TRANSCRIPT = False                                # Print translated transcript to console

#Subtitle Burning
APPEND_FILENAME = " - AI Translated"                    # Output filename suffix
SUBTITLE_COLOR = "d79e9e"                               # Text color, hex with no '#'
SUBTITLE_FONT_SIZE = 64                                 # Font size in pixels
SUBTITLE_MARGIN_BOTTOM = 50                             # Space from bottom in pixels

#Language
SPEECH_LANGUAGE = 'ja'                                  # Code for the language the video is in. Options: en(English), ja(Japanese), es(Spanish), zh(Chinese), fr(French), de(German), etc.
SUBTITLE_LANGUAGE = 'en'                                # Subtitle language code, same code options as above
TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-ja-en"         # Huggingface Translator Model. Language pair specific, search for the best one: https://huggingface.co/models?pipeline_tag=translation&sort=downloads
                                                        # Spanish to English: Helsinki-NLP/opus-mt-es-en      Japanese to English: Helsinki-NLP/opus-mt-ja-en      Chinese to English: Helsinki-NLP/opus-mt-zh-en
#Whisper
WHISPER_MODEL = "large-v3"                              #The whisper model to use for speech to text. Options: "large-v3"  "turbo" "medium" "small"
PRELOAD_WHISPER = True                                  # Preload whisper. Faster if you have enough RAM
TEMP = 0.05                                             # Transcription temperature, reccomend close to 0 but not 0
CHUNK_SIZE = 300 * 60                                   # Chunk Size in seconds
OVERLAP = 0                                             # Chunk overlap in seconds, for not splitting phrases at the end of the chunk
PREROLL = 0                                             # Chunk pre-roll is seconds, for not splitting phrases at the beginning of the chunk

#Technical
USE_GPU = True                                          # Set False to use CPU
MAX_LENGTH = 150                                        # Max charactor length of a subtitle
MAX_DURATION = 5                                        # Max subtitle duration in seconds 
MIN_DURATION = 0.5                                      # Minimum subtitle duration in seconds
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"    #Required full path of ffmpeg.exe
# =============================================================================================================================== #

# ====== CONFIGURATION ====== #                            
import os, subprocess, torch, gc, whisper, time, re, shlex
from moviepy.editor import VideoFileClip
from transformers import pipeline

# FFmpeg setup
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_PATH}")

# Model locations
WHISPER_MODEL_DIR = "./whisper_models"
os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
# ================================== #

# ====== FUNCTIONS ================= #
def setup_device():
    #Configure GPU/CPU usage based on available hardware
    model_size = WHISPER_MODEL
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU" if not USE_GPU else "!!!!! GPU not available, falling back to CPU !!!!!")
    print(f"Selected Whisper model: {model_size}")
    return device, model_size

def rgb_to_ass_bgr(rgb_hex: str, alpha: int = 0x00) -> str:
    #Convert RRGGBB hex to .ass &HAABBGGRR format.
    r = int(rgb_hex[0:2], 16)
    g = int(rgb_hex[2:4], 16)
    b = int(rgb_hex[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

def preprocess_audio(audio_path):
    #Apply audio processing to improve speech recognition
    print(f"Pre-processing audio...")
    processed_path = audio_path.replace(".wav", "_processed.wav")
    command = [
        FFMPEG_PATH,
        "-y", "-loglevel", "error",
        "-i", audio_path,
        "-af", ("highpass=f=300,lowpass=f=3500,"
                "afftdn=nf=-30,"
                "equalizer=f=800:width_type=h:width=800:g=8,"
                "loudnorm=I=-16:TP=-1.5:LRA=11"),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        processed_path
    ]

    subprocess.run(command, check=True)
    return processed_path

def clean_text(text):
    #Post-process text for better accuracy
    # Remove unwanted punctuation and normalize spacing
    text = re.sub(r'[、。]+\s*', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return
    
def transcribe_audio(audio_path, device, model_size):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing audio file: {os.path.abspath(audio_path)}")

    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size, device=device, download_root=WHISPER_MODEL_DIR, in_memory=PRELOAD_WHISPER)

    try:
        # Get audio duration
        audio_info = subprocess.run(
            [FFMPEG_PATH, "-i", audio_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", audio_info.stderr)
        duration = sum(float(x) * 60 ** i for i, x in enumerate(reversed(duration_match.groups())))

        print(f"Transcribing audio ({duration/60:.1f} min)")
        segments = []
        
        last_text = None
        duplicate_count = 0
        
        for start in range(0, int(duration), CHUNK_SIZE - OVERLAP):
            end = min(start + CHUNK_SIZE, duration)
            chunk_path = f"{audio_path}_chunk_{start}.wav"
            print(f"Processing section: {start/60:.1f}-{end/60:.1f} min")

            subprocess.run([
                FFMPEG_PATH, "-y", "-loglevel", "error",
                "-ss", str(max(0, start - PREROLL)),
                "-accurate_seek",  
                "-i", audio_path,
                "-t", str(CHUNK_SIZE + PREROLL + OVERLAP),
                "-c:a", "pcm_s16le",
                "-ar", "16000", 
                "-ac", "1",
                chunk_path
            ], check=True)
            
            # Transcribe with initial prompt to discourage repetitions
            result = model.transcribe(
                chunk_path,
                language=SPEECH_LANGUAGE,
                temperature=TEMP,  
                initial_prompt="Clear dialogue without repetitions. Proper punctuation.",
                carry_initial_prompt=True,             #Apply throughout, False is only first 30 seconds
                condition_on_previous_text=True,      # carryover previous transcription as context?
                compression_ratio_threshold=2.4,      # Lower is stricter declaring hallucinations
                logprob_threshold=-0.7,               # Close to 0 is stricter generation confidence
                no_speech_threshold=0.5              # Lower is more strict for declaring silence
            )
            
            # Process segments with duplicate detection
            for segment in result["segments"]:
                segment["start"] += start
                segment["end"] += start
                
                # Only keep segments in the non-overlapping portion
                if segment["start"] >= start + OVERLAP/2:
                    current_text = segment["text"].strip().lower()
                    
                    # Skip duplicates and very short segments
                    if (current_text and 
                        current_text != last_text and 
                        len(current_text) > 2):
                        
                        if not re.match(r'^(oh+|ah+|mm+|uh+|hm+)$', current_text):
                            segments.append(segment)
                            last_text = current_text
                        else:
                            duplicate_count += 1
            
            os.remove(chunk_path)
        if duplicate_count > 0:
            print(f"Filtered out {duplicate_count} repetitive segments")
        
        return {"segments": segments}
            
    finally: #Release Reourses used for Speech to text
        del model 
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def create_srt_file(segments, translations, output_path): #If you want to save the .srt file.
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (segment, translation) in enumerate(zip(segments, translations), 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n")
            f.write(f"{translation}\n\n")

def translate_video(input_path): #Main Function
    print('\n','\n',f'Starting Translation of {input_path} from {SPEECH_LANGUAGE} to {SUBTITLE_LANGUAGE}','\n')
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}{APPEND_FILENAME}.mp4"
    audio_path = f"{base_name}_temp.wav"
    ass_path = f"{base_name} subtitles.ass"
    srt_path = f"{base_name} subtitles.srt"

    device, model_size = setup_device()

    try:
        # 1. Extract and preprocess audio
        print("Extracting audio...")
        extract_audio(input_path, audio_path)
        processed_audio = preprocess_audio(audio_path)

        # 2. Transcribe
        result = transcribe_audio(processed_audio, device, model_size)

        # 3. Initialize translator
        print("Initializing translator...")
        translator = pipeline(task=f"translation_{SPEECH_LANGUAGE}_to_{SUBTITLE_LANGUAGE}", model=TRANSLATOR_MODEL, device=0 if device == "cuda" else -1)

        # 4. Generate .ass file
        print("Creating subtitles...")
        PRIMARY_COLOUR_ASS = rgb_to_ass_bgr(SUBTITLE_COLOR, alpha=0x00) #Format colors
        BACK_COLOUR_ASS = rgb_to_ass_bgr("000000", alpha=0x80)

        with open(ass_path, "w", encoding="utf-8") as f:
            # .ass header
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n")
            f.write("Timer: 100.0000\n\n")

            # Styles
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n")
            
            margin_l = 30
            margin_r = 30

            f.write(f"Style: Default,Arial,{SUBTITLE_FONT_SIZE},{PRIMARY_COLOUR_ASS},&H00000000,"
                    f"&H00000000,{BACK_COLOUR_ASS},0,0,0,0,100,100,0,0,1,1,0,2,{margin_l},{margin_r},{SUBTITLE_MARGIN_BOTTOM},1\n\n")

            # Events
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            translations = []
            for segment in result["segments"]: #The actual translation part
                # Base timing
                original_start = segment["start"]
                original_end = segment["end"]
                original_duration = original_end - original_start

                # Set the upper and lower bounds for duration
                adjusted_duration = min(
                    max(MIN_DURATION, original_duration),
                    MAX_DURATION if MAX_DURATION > 0 else float('inf')
                )
                adjusted_end = original_start + adjusted_duration

                clean_text_output = clean_text(segment["text"]) # Handle text
                translation = translator(clean_text_output)[0]['translation_text'] # Translate

                if MAX_LENGTH > 0 and len(translation) > MAX_LENGTH: # Apply uupper charactor limit
                    translation = translation[:MAX_LENGTH]

                translations.append(translation) #Build translation list for .srt file

                if PRINT_TRANSCRIPT:
                    print(f"[{format_time(original_start)}] {translation}")

                # Write to .ass file
                start_time = format_time(original_start, ass=True)
                end_time = format_time(adjusted_end, ass=True)
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{translation}\n")

        if SAVE_SRT_FILE:
            create_srt_file(result["segments"], translations, srt_path)
            print(f".srt Subtitles saved: {srt_path}")

        if BURN_INTO_VIDEO:
            print("Burning subtitles into video...")
            cfr_video = f"{base_name}_cfr.mp4"
            
            # First convert to constant frame rate
            subprocess.run([
                FFMPEG_PATH, "-y",
                "-i", input_path,
                "-vsync", "cfr",
                "-c:v", "libx264",  # ← Re-encode to H.264 (MP4 compatible)
                "-preset", "fast",  # Balance between speed and quality
                "-crf", "23",       # Good quality (lower = better quality, 18-28 range)
                "-c:a", "aac",      # Re-encode audio to AAC (MP4 compatible)
                "-b:a", "128k",     # Audio bitrate
                cfr_video
            ], check=True)
            
            encoder, hwaccel, preset_param, preset_val, *extra = get_gpu_encoder() #Get processor settings

            probe = subprocess.run(
                [FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe"),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path],
                capture_output=True, text=True
            )
            input_bitrate = int(probe.stdout.strip()) if probe.stdout.strip().isdigit() else 5_000_000
            target_bitrate = str(input_bitrate)
            maxrate = str(int(input_bitrate * 1.5))
            bufsize = str(int(input_bitrate * 2))

            cmd = [
                FFMPEG_PATH, "-y",
                "-i", cfr_video,
                "-vf", f"subtitles={shlex.quote(ass_path)}:force_style='PlayResX=1920,PlayResY=1080'",
                "-c:v", encoder,
                preset_param, preset_val,
                "-rc", "vbr",
                "-b:v", target_bitrate,
                "-maxrate", maxrate,
                "-bufsize", bufsize,
                *extra,
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "128k",
                "-video_track_timescale", "90000",
                output_path
            ]
            
            if hwaccel: #This is required for GPUs
                cmd.insert(2, "-hwaccel")
                cmd.insert(3, hwaccel)
            
            try:
                subprocess.run(cmd, check=True) #Encode video with subtitles
                print(f"Successfully encoded with {encoder.upper()}")
            except subprocess.CalledProcessError as e:
                print(f"Encoding failed ({e}), trying with fallback settings...")

                # Retry with CPU settings
                subprocess.run([
                    FFMPEG_PATH, "-y",
                    "-i", cfr_video,
                    "-vf", f"subtitles={shlex.quote(ass_path)}:force_style='PlayResX=1920,PlayResY=1080'",
                    "-c:v", "libx264",
                    "-crf", "22",  # Slightly higher CRF for smaller files
                    "-preset", "slower",  # Better compression
                    "-tune", "film",  # Optimize for live action
                    "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k",
                    "-video_track_timescale", "90000",
                    output_path
                ], check=True)
            
            # Cleanup temp CFR file
            os.remove(cfr_video)

            print(f"Successfully created: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

    finally: # Cleanup temporary files
        cleanup = [audio_path, processed_audio]
        if not SAVE_ASS_FILE:
            cleanup.append(ass_path)
        else:
            print(f".ass Subtitles saved: {ass_path}")
        print("Cleaning up temp files...")
        for path in cleanup:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Could not delete {path}: {str(e)}")

def get_gpu_encoder():
    #Detect available GPU encoder with fallback to CPU
    if not USE_GPU:
        return ("libx264", None, "-preset", "slower", "-tune", "film")
    try:
        result = subprocess.run([FFMPEG_PATH, "-hide_banner", "-encoders"], 
                              capture_output=True, text=True)
        if "h264_nvenc" in result.stdout: # Check NVIDIA
            print('Nvidia selected for video encoding')
            return ("h264_nvenc", "cuda", "-preset", "p5")
        
        if "h264_amf" in result.stdout: # Check AMD
            return ("h264_amf", "auto", "-preset", "speed")
            
        if "h264_qsv" in result.stdout: # Check Intel
            return ("h264_qsv", "qsv", "-preset", "fast")
    except Exception:
        pass
    return ("libx264", None, "-preset", "slower", "-tune", "film") # CPU fallback

def format_time(seconds, ass=False):
    """
    Convert seconds to a timestamp string.
    - For .srt: HH:MM:SS,ms
    - For .ass: H:MM:SS.cs (centiseconds)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    frac = seconds - int(seconds)
    
    if ass:
        cs = int(frac * 100)  # centiseconds
        return f"{hours}:{minutes:02}:{sec:02}.{cs:02}"
    else:
        ms = int(frac * 1000) # miliseconds
        return f"{hours:02d}:{minutes:02d}:{sec:02},{ms:03d}"

if __name__ == "__main__": #Run the file
    start_time = time.time() #start time
    translate_video(INPUT_FILENAME) #Do everything
    end_time = time.time() #end time
    total_seconds = end_time - start_time
    print('\n',f"Done! Total runtime: {total_seconds:.2f} seconds ({total_seconds/60:.2f} minutes)")