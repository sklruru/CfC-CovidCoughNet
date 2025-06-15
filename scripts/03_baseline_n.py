import os
import numpy as np
import pandas as pd
import librosa
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def get_project_root():
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))


class WaveletScattering:
    def __init__(self, J=6, Q=16, sr=16000):
        self.J = J
        self.Q = Q
        self.sr = sr

    def morlet_wavelet(self, n_samples, center_freq, sigma=1.0):
        t = np.arange(n_samples) - n_samples // 2
        t = t / self.sr

        wavelet = np.exp(1j * 2 * np.pi * center_freq * t) * np.exp(-t ** 2 / (2 * sigma ** 2))
        return wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))

    def transform(self, signal):
        features = [np.mean(np.abs(signal))]

        for j1 in range(1, min(self.J, 5)):
            freq = self.sr / (2 ** j1)
            if freq < 50:
                break

            wavelet = self.morlet_wavelet(len(signal), freq)
            signal_fft = np.fft.fft(signal, n=len(signal))
            wavelet_fft = np.fft.fft(wavelet, n=len(signal))
            conv_result = np.fft.ifft(signal_fft * np.conj(wavelet_fft))

            modulus = np.abs(conv_result)
            features.append(np.mean(modulus))

            if j1 < 3:
                for j2 in range(j1 + 1, min(j1 + 3, self.J)):
                    freq2 = self.sr / (2 ** j2)
                    if freq2 < 50:
                        break

                    wavelet2 = self.morlet_wavelet(len(modulus), freq2)
                    mod_fft = np.fft.fft(modulus, n=len(modulus))
                    wavelet2_fft = np.fft.fft(wavelet2, n=len(modulus))
                    conv_result2 = np.fft.ifft(mod_fft * np.conj(wavelet2_fft))
                    features.append(np.mean(np.abs(conv_result2)))

        return np.array(features)


def apply_data_augmentation(audio, augmentation_type='noise'):
    if augmentation_type == 'noise':
        noise_factor = np.random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise

    elif augmentation_type == 'speed':
        speed_factor = np.random.uniform(0.95, 1.05)
        stretched_audio = librosa.effects.time_stretch(audio, rate=speed_factor)

        if len(stretched_audio) != len(audio):
            if len(stretched_audio) > len(audio):
                stretched_audio = stretched_audio[:len(audio)]
            else:
                stretched_audio = np.pad(stretched_audio, (0, len(audio) - len(stretched_audio)), 'wrap')
        return stretched_audio

    elif augmentation_type == 'pitch':
        pitch_factor = np.random.uniform(-1, 1)
        return librosa.effects.pitch_shift(audio, sr=16000, n_steps=pitch_factor)

    return audio


class CovidCoughClassifier:
    def __init__(self):
        self.wavelet_transformer = WaveletScattering(J=6, Q=8, sr=16000)

    def load_audio_files(self, positive_path, negative_path):
        audio_data, labels = [], []

        # Load positive samples
        pos_files = [f for f in os.listdir(positive_path) if f.endswith('.wav')]
        for file in tqdm(pos_files, desc="Loading positive"):
            audio, _ = librosa.load(os.path.join(positive_path, file), sr=16000)
            audio_data.append(audio)
            labels.append(1)

        # Load negative samples
        neg_files = [f for f in os.listdir(negative_path) if f.endswith('.wav')]
        for file in tqdm(neg_files, desc="Loading negative"):
            audio, _ = librosa.load(os.path.join(negative_path, file), sr=16000)
            audio_data.append(audio)
            labels.append(0)

        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"Loaded {neg_count} negative, {pos_count} positive samples")
        print(f"Ratio: {neg_count / pos_count:.1f}:1")

        return audio_data, labels

    def preprocess_audio(self, audio_data, target_duration=3.0):
        processed_data = []
        target_length = int(target_duration * 16000)

        for audio in tqdm(audio_data, desc="Preprocessing"):
            # Normalize
            if np.std(audio) > 0:
                audio = (audio - np.mean(audio)) / np.std(audio)

            # Adjust length
            if len(audio) > target_length:
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            elif len(audio) < target_length:
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (pad_length // 2, pad_length - pad_length // 2))

            processed_data.append(audio)

        return processed_data

    def data_augmentation(self, audio_data, labels, random_seed=42):
        np.random.seed(random_seed)

        positive_audio = [audio for audio, label in zip(audio_data, labels) if label == 1]
        negative_audio = [audio for audio, label in zip(audio_data, labels) if label == 0]

        augmentation_types = ['noise', 'speed', 'pitch']
        augmented_positive = []

        for base_audio in tqdm(positive_audio, desc="Augmenting"):
            for _ in range(3):
                aug_type = np.random.choice(augmentation_types)
                augmented_audio = apply_data_augmentation(base_audio.copy(), aug_type)
                augmented_positive.append(augmented_audio)

        final_audio_data = positive_audio + augmented_positive + negative_audio
        final_labels = [1] * (len(positive_audio) + len(augmented_positive)) + [0] * len(negative_audio)

        print(
            f"After augmentation: {len(negative_audio)} negative, {len(positive_audio) + len(augmented_positive)} positive")
        return final_audio_data, final_labels

    def extract_features(self, audio_data):
        features = []

        for audio in tqdm(audio_data, desc="Extracting features"):
            # Wavelet scattering features
            scattering_features = self.wavelet_transformer.transform(audio)

            # Traditional audio features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=16000))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=16000))
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=5)
            mfcc_means = np.mean(mfccs, axis=1)

            # Combine all features
            combined_features = np.concatenate([
                scattering_features,
                [spectral_centroid, spectral_bandwidth, zcr],
                mfcc_means
            ])

            features.append(combined_features)

        return np.array(features)

    def train_and_evaluate_single_run(self, features, labels, seed_output_path, seed=42):
        # Initialize model
        scaler = StandardScaler()
        classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50,
            learning_rate=1.0,
            random_state=seed
        )

        # Data preprocessing
        features_scaled = scaler.fit_transform(features)

        # Data split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_scaled, labels, test_size=0.1, random_state=seed, stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=seed, stratify=y_train_val
        )

        # Train model
        classifier.fit(X_train, y_train)

        # Validation evaluation
        y_val_pred = classifier.predict(X_val)
        y_val_pred_prob = classifier.predict_proba(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_pred_prob[:, 1])

        # Test evaluation
        y_pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])

        categories = ["Negative", "Positive"]
        cr = classification_report(y_test, y_pred, target_names=categories, digits=6, output_dict=True)

        print(f"Seed {seed} - Test Acc: {accuracy:.4f}, AUC: {auc_score:.4f}")

        # Save results
        results_df = pd.DataFrame({
            'Sample_ID': [f"Sample_{i}" for i in range(len(y_test))],
            'True_Label': [categories[y] for y in y_test],
            'Predicted_Label': [categories[y] for y in y_pred],
            'Positive_Probability': y_pred_prob[:, 1]
        })
        results_df.to_csv(os.path.join(seed_output_path, 'test_results.csv'), index=False)

        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Test_Accuracy', 'Test_Precision (Positive)', 'Test_Recall (Positive)',
                       'Test_F1-Score (Positive)', 'Test_AUC', 'Val_Accuracy', 'Val_AUC'],
            'Value': [
                accuracy,
                cr['Positive']['precision'],
                cr['Positive']['recall'],
                cr['Positive']['f1-score'],
                auc_score,
                val_accuracy,
                val_auc
            ]
        })
        metrics_df.to_csv(os.path.join(seed_output_path, 'performance_metrics.csv'), index=False)

        return {
            'seed': seed,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': cr['Positive']['precision'],
            'recall': cr['Positive']['recall'],
            'f1_score': cr['Positive']['f1-score'],
            'val_accuracy': val_accuracy,
            'val_auc': val_auc
        }


# Main program execution
# Set paths
project_root = get_project_root()
positive_path = os.path.join(project_root, "data", "raw", "Positive")
negative_path = os.path.join(project_root, "data", "raw", "Negative")
base_results_path = os.path.join(project_root, "results")

# Check data paths
if not os.path.exists(positive_path) or not os.path.exists(negative_path):
    print("Data paths do not exist, please check data location")
    exit()

# Create results directory
os.makedirs(base_results_path, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
results_output_path = os.path.join(base_results_path, f"{timestamp}_benchmark")
os.makedirs(results_output_path, exist_ok=True)

# Initialize classifier
random_seeds = [42, 123, 256, 789, 1024]
classifier = CovidCoughClassifier()

print("Starting data processing...")
audio_data, labels = classifier.load_audio_files(positive_path, negative_path)
processed_audio = classifier.preprocess_audio(audio_data, target_duration=3.0)
augmented_audio, augmented_labels = classifier.data_augmentation(processed_audio, labels, random_seed=42)
features = classifier.extract_features(augmented_audio)

print(f"Feature shape: {features.shape}")
print("Starting multi-seed training...")

# Multi-seed training
all_results = []
for seed in random_seeds:
    seed_output_path = os.path.join(results_output_path, f'seed_{seed}')
    os.makedirs(seed_output_path, exist_ok=True)

    result = classifier.train_and_evaluate_single_run(features, augmented_labels, seed_output_path, seed=seed)
    all_results.append(result)

# Statistics
results_df = pd.DataFrame(all_results)
mean_results = results_df.mean()
std_results = results_df.std()

summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'Val_Accuracy', 'Val_AUC'],
    'Mean': [
        mean_results['accuracy'], mean_results['auc'], mean_results['precision'],
        mean_results['recall'], mean_results['f1_score'], mean_results['val_accuracy'], mean_results['val_auc']
    ],
    'Std Dev': [
        std_results['accuracy'], std_results['auc'], std_results['precision'],
        std_results['recall'], std_results['f1_score'], std_results['val_accuracy'], std_results['val_auc']
    ]
})

# Save statistical results
results_df.to_csv(os.path.join(results_output_path, 'all_runs_results.csv'), index=False)
summary_df.to_csv(os.path.join(results_output_path, 'summary_results.csv'), index=False)

# Print results
print("\nResults from all runs:")
print(results_df.to_string(index=False))
print("\nSummary statistics:")
for _, row in summary_df.iterrows():
    print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std Dev']:.4f}")

print(f"\nTraining completed, results saved to: {results_output_path}")