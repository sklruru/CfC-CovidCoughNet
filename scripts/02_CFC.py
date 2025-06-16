import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from datetime import datetime
from imblearn.over_sampling import SMOTE

# path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from models.tf_cfc import build_cfc_model

features_processed_data = os.path.join(project_root, "features", "yamnet_features")
timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
results_output_path = os.path.join(project_root, "results", timestamp)
os.makedirs(results_output_path, exist_ok=True)

# Experiment parameters
# random_seeds = [42, 123]
epochs = 2
# learning_rate = 0.0004

random_seeds = [42, 123, 256, 789, 1024]
# epochs = 150
validation_ratio = 0.2
test_ratio = 0.1
batch_size = 16
early_stopping_patience = 20
learning_rate = 0.0005

# Model parameters
MODEL_HPARAMS = {
    "backbone_activation": "silu",
    "backbone_layers": 3,
    "backbone_units": 200,
    "backbone_dr": 0.25,
    "weight_decay": 1e-6,
    "minimal": False,
    "no_gate": False,
}

focal_loss_gamma = 2.0
focal_loss_alpha = 0.25
categories = ["Negative", "Positive"]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def categorical_focal_loss(gamma=focal_loss_gamma, alpha=focal_loss_alpha):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1 - y_pred, gamma) * y_true
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


def load_features(base_dir, categories):
    features = []
    labels = []
    for i, category in enumerate(categories):
        cat_dir = os.path.join(base_dir, category)
        files = os.listdir(cat_dir)
        for file in files:
            file_path = os.path.join(cat_dir, file)
            feature = np.load(file_path)
            features.append(feature)
            labels.append(i)
    features_array = np.array(features)
    labels_array = np.array(labels)
    return features_array, labels_array


def create_dataset(features, labels, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_and_evaluate(seed):
    print(f"\nTraining with seed {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    seed_output_path = os.path.join(results_output_path, f'seed_{seed}')
    os.makedirs(seed_output_path, exist_ok=True)
    model_save_path = os.path.join(seed_output_path, 'cfc_model')

    # split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=validation_ratio,
        random_state=seed,
        stratify=y_train_val
    )

    # SMOTE oversampling
    n_samples = X_train.shape[0]
    orig_shape = X_train.shape[1:]
    X_train_reshaped = X_train.reshape(n_samples, -1)
    smote = SMOTE(random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    X_train = X_train_resampled.reshape(-1, *orig_shape)
    y_train = y_train_resampled

    # class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(range(len(categories)), class_weights))

    # convert to one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train)
    y_val_onehot = tf.keras.utils.to_categorical(y_val)
    y_test_onehot = tf.keras.utils.to_categorical(y_test)

    # create datasets
    train_dataset = create_dataset(X_train, y_train_onehot, batch_size)
    val_dataset = create_dataset(X_val, y_val_onehot, batch_size, is_training=False)
    test_dataset = create_dataset(X_test, y_test_onehot, batch_size, is_training=False)

    # build CfC model
    input_shape = X_train[0].shape
    model = build_cfc_model(input_shape, MODEL_HPARAMS, len(categories))

    # compile CfC model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=categorical_focal_loss(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall', patience=early_stopping_patience,
            restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path, monitor='val_recall',
            mode='max', save_best_only=True
        )
    ]

    # train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # evaluate model
    test_metrics = model.evaluate(test_dataset, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # classification report
    cm = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])
    # save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Seed {seed})')
    plt.tight_layout()
    plt.savefig(os.path.join(seed_output_path, 'confusion_matrix.png'))
    plt.close()

    # save training history
    metrics_to_plot = ['accuracy', 'loss', 'recall', 'precision']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f'Training')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation')
        plt.title(f'{metric.capitalize()} (Seed {seed})')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(seed_output_path, f'training_{metric}.png'))
        plt.close()

    # save model
    model.save(model_save_path)

    # save results
    results_df = pd.DataFrame({
        'Sample_ID': [f"Sample_{i}" for i in range(len(y_test))],
        'True_Label': [categories[y] for y in y_test],
        'Predicted_Label': [categories[y] for y in y_pred],
        'Positive_Probability': y_pred_prob[:, 1]
    })
    results_df.to_csv(os.path.join(seed_output_path, 'test_results.csv'), index=False)

    # save metrics
    metrics_dict = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [
            test_metrics[1],
            metrics_dict['Positive']['precision'],
            metrics_dict['Positive']['recall'],
            metrics_dict['Positive']['f1-score'],
            auc_score
        ]
    })
    metrics_df.to_csv(os.path.join(seed_output_path, 'performance_metrics.csv'), index=False)

    return {
        'seed': seed,
        'accuracy': test_metrics[1],
        'auc': test_metrics[2],
        'precision': test_metrics[3],
        'recall': test_metrics[4],
        'f1_score': metrics_dict['Positive']['f1-score'],
        'epochs_trained': len(history.history['loss'])
    }


# load data
X, y = load_features(features_processed_data, categories)
# train models
all_results = []
for seed in random_seeds:
    result = train_and_evaluate(seed)
    all_results.append(result)

# results summary
results_df = pd.DataFrame(all_results)
mean_results = results_df.mean()
std_results = results_df.std()

summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score'],
    'Mean': [mean_results['accuracy'], mean_results['auc'], mean_results['precision'],
             mean_results['recall'], mean_results['f1_score']],
    'Std Dev': [std_results['accuracy'], std_results['auc'], std_results['precision'],
                std_results['recall'], std_results['f1_score']]
})

# save results
results_df.to_csv(os.path.join(results_output_path, 'all_runs_results.csv'), index=False)
summary_df.to_csv(os.path.join(results_output_path, 'summary_results.csv'), index=False)

# visualize results
plt.figure(figsize=(12, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=results_df[metric])
    plt.title(f'{metric.capitalize()}')
    plt.ylabel(metric.capitalize())
plt.tight_layout()
plt.savefig(os.path.join(results_output_path, 'metrics_distribution.png'))
plt.close()

print("\nResults Summary:")
print(results_df.to_string(index=False))
print("\nStatistics:")
for _, row in summary_df.iterrows():
    print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std Dev']:.4f}")
print(f"\nTraining completed. Results saved to: {results_output_path}")