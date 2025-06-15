import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import librosa
import librosa.effects
from tqdm import tqdm
import tempfile
import shutil
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Clean TensorFlow cache
tfhub_cache_dir = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
if os.path.exists(tfhub_cache_dir):
    print(f"Cleaning TensorFlow cache: {tfhub_cache_dir}")
    shutil.rmtree(tfhub_cache_dir)

# Set cache directory and CPU execution
custom_cache_dir = os.path.join(PROJECT_ROOT, "models", "tfhub_cache")
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['TFHUB_CACHE_DIR'] = custom_cache_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Input and output paths
input_raw_audio = os.path.join(PROJECT_ROOT, "data", "raw")
output_yamnet_features = os.path.join(PROJECT_ROOT, "features", "yamnet_features")
os.makedirs(output_yamnet_features, exist_ok=True)

yamnet_model_dir = os.path.join(PROJECT_ROOT, "models", "yamnet")
os.makedirs(yamnet_model_dir, exist_ok=True)

# Parameter configuration
TARGET_SR = 16000
TARGET_DURATION = 3.0
QUALITY_THRESHOLD = 0.5
POSITIVE_AUG_RATIO = 3
MAX_WORKERS = min(8, mp.cpu_count())
BATCH_SIZE = 32

target_length = int(TARGET_DURATION * TARGET_SR)


def load_yamnet_model():
    """Load YAMNet model"""
    local_model_path = os.path.join(yamnet_model_dir, "yamnet_1")

    if os.path.exists(local_model_path):
        print("Loading YAMNet model from local cache...")
        try:
            return hub.load(local_model_path)
        except Exception as e:
            print(f"Local model loading failed: {e}")

    print("Downloading YAMNet model...")
    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        tf.saved_model.save(model, local_model_path)
        return model
    except Exception as e:
        print(f"Model download failed: {e}")
        return None


def preprocess_audio(file_path):
    """Preprocess audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR)

        # Normalization
        if np.std(audio) > 0:
            audio = (audio - np.mean(audio)) / np.std(audio)

        # Adjust length
        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (pad_length // 2, pad_length - pad_length // 2))

        return audio.astype(np.float32)
    except Exception:
        return None


def apply_augmentation(audio, aug_type):
    """Apply data augmentation"""
    try:
        if aug_type == 'noise':
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_factor, len(audio))
            return audio + noise

        elif aug_type == 'speed':
            temp_sr = 22050
            temp_audio = librosa.resample(audio, orig_sr=16000, target_sr=temp_sr)
            speed_factor = np.random.uniform(0.95, 1.05)
            stretched_audio = librosa.effects.time_stretch(temp_audio, rate=speed_factor)

            # Length adjustment
            if len(stretched_audio) != len(temp_audio):
                if len(stretched_audio) > len(temp_audio):
                    stretched_audio = stretched_audio[:len(temp_audio)]
                else:
                    stretched_audio = np.pad(stretched_audio,
                                             (0, len(temp_audio) - len(stretched_audio)), 'wrap')

            return librosa.resample(stretched_audio, orig_sr=temp_sr, target_sr=16000)

        elif aug_type == 'pitch':
            temp_sr = 22050
            temp_audio = librosa.resample(audio, orig_sr=16000, target_sr=temp_sr)
            pitch_factor = np.random.uniform(-1, 1)
            pitched_audio = librosa.effects.pitch_shift(temp_audio, sr=temp_sr, n_steps=pitch_factor)
            return librosa.resample(pitched_audio, orig_sr=temp_sr, target_sr=16000)

        return audio
    except Exception:
        return audio


def generate_augmented_audio(original_audio, base_name, label):
    """Generate augmented audio versions"""
    audio_versions = [(original_audio, f"{base_name}_original")]

    # Only augment positive samples
    if label == "Positive":
        augmentation_types = ['noise', 'speed', 'pitch']

        for i in range(POSITIVE_AUG_RATIO):
            aug_type = np.random.choice(augmentation_types)
            augmented_audio = apply_augmentation(original_audio, aug_type)

            if augmented_audio is not None:
                version_name = f"{base_name}_aug_{aug_type}_{i:04d}"
                audio_versions.append((augmented_audio.astype(np.float32), version_name))

    return audio_versions


def check_audio_quality(audio):
    """Check audio quality"""
    if audio is None or len(audio) == 0:
        return False

    if np.isnan(audio).any() or np.isinf(audio).any():
        return False

    if np.max(np.abs(audio)) < 0.01:
        return False

    return True


def extract_features(yamnet_model, waveform):
    """Extract YAMNet features"""
    try:
        scores, embeddings, spectrogram = yamnet_model(waveform)

        raw_features = embeddings.numpy() if isinstance(embeddings, tf.Tensor) else embeddings

        # Create optimized features
        frame_energy = np.linalg.norm(raw_features, axis=1, keepdims=True)
        normalized_frames = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-10)

        cfc_features = np.concatenate([
            normalized_frames,
            np.diff(raw_features, axis=0, prepend=raw_features[0:1]),
            frame_energy
        ], axis=1)

        return cfc_features, raw_features
    except Exception:
        return None, None


def check_feature_quality(features):
    """Check feature quality"""
    if np.isnan(features).any() or np.isinf(features).any():
        return False

    feature_variance = np.var(features, axis=0).mean()
    if feature_variance < 0.01:
        return False

    return True


def process_audio_file(args):
    """Process single audio file"""
    audio_file, raw_audio_folder, label, yamnet_model, output_dir = args

    file_path = os.path.join(raw_audio_folder, audio_file)
    base_name = os.path.splitext(audio_file)[0]

    # Load and preprocess
    original_audio = preprocess_audio(file_path)
    if not check_audio_quality(original_audio):
        return {"success": False, "file": audio_file}

    # Generate augmented versions
    audio_versions = generate_augmented_audio(original_audio, base_name, label)

    # Extract features
    successful_count = 0

    for audio_data, version_name in audio_versions:
        cfc_features, raw_features = extract_features(yamnet_model, audio_data)

        if cfc_features is not None and check_feature_quality(raw_features):
            feature_path = os.path.join(output_dir, f"{version_name}.npy")
            try:
                np.save(feature_path, cfc_features)
                successful_count += 1
            except Exception:
                pass

    return {
        "success": True,
        "file": audio_file,
        "count": successful_count,
        "label": label
    }


def process_label_folder(label, yamnet_model):
    """Process label folder"""
    raw_audio_folder = os.path.join(input_raw_audio, label)
    if not os.path.exists(raw_audio_folder):
        print(f"Folder does not exist: {raw_audio_folder}")
        return 0, 0

    # Get audio files
    audio_files = [f for f in os.listdir(raw_audio_folder)
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]

    if not audio_files:
        print(f"No audio files found: {raw_audio_folder}")
        return 0, 0

    # Create output directory
    output_dir = os.path.join(output_yamnet_features, label)
    os.makedirs(output_dir, exist_ok=True)

    # Parallel processing
    process_args = [
        (audio_file, raw_audio_folder, label, yamnet_model, output_dir)
        for audio_file in audio_files
    ]

    successful_features = 0
    processed_files = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_audio_file, process_args),
            total=len(process_args),
            desc=f"Processing {label}"
        ))

    for result in results:
        if result["success"]:
            successful_features += result["count"]
            processed_files += 1

    return successful_features, processed_files


# Main program
print("=" * 50)
print("COVID Cough Detection - Feature Extraction")
print("=" * 50)

# Count original data
positive_folder = os.path.join(input_raw_audio, "Positive")
negative_folder = os.path.join(input_raw_audio, "Negative")

pos_files = len([f for f in os.listdir(positive_folder)
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]) if os.path.exists(positive_folder) else 0
neg_files = len([f for f in os.listdir(negative_folder)
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]) if os.path.exists(negative_folder) else 0

print(f"\nOriginal dataset:")
print(f"  Positive samples: {pos_files}")
print(f"  Negative samples: {neg_files}")
if pos_files > 0:
    print(f"  Negative to positive ratio: {neg_files / pos_files:.1f}:1")

print(f"\nAugmentation strategy:")
print(f"  Positive samples: {POSITIVE_AUG_RATIO}x augmentation")
print(f"  Negative samples: No augmentation")

# Load model
print("\nLoading YAMNet model...")
yamnet_model = load_yamnet_model()
if yamnet_model is None:
    print("Model loading failed!")
    exit(1)

# Process data
print("\nStarting audio file processing...")
stats = {}

for label in ["Positive", "Negative"]:
    feature_count, file_count = process_label_folder(label, yamnet_model)
    stats[label] = {"features": feature_count, "files": file_count}

# Output results
print("\n" + "=" * 50)
print("Processing completed!")
print("=" * 50)

pos_features = stats["Positive"]["features"]
neg_features = stats["Negative"]["features"]
total_features = pos_features + neg_features

print(f"\nFinal dataset:")
print(f"  Positive features: {pos_features}")
print(f"  Negative features: {neg_features}")
print(f"  Total features: {total_features}")

if pos_features > 0 and neg_features > 0:
    print(f"  New negative to positive ratio: {neg_features / pos_features:.1f}:1")

print(f"\nFeatures saved to: {output_yamnet_features}")
print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")