import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    # 1. Force CPU execution
    tf.config.set_visible_devices([], 'GPU')
    print("GPU Disabled. Using CPU strictly.")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Load data
    data_dir = os.path.join(base_dir, 'data', 'processed')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # 3. Load model architecture
    models_dir = os.path.join(base_dir, 'models')
    with open(os.path.join(models_dir, 'model_architecture.json'), 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 4. Train with EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    batch_size = 32
    epochs = 100
    validation_split = 0.2
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    hist_dict = history.history
    
    # Metrics for Epoch 1 and Final Epoch
    epoch_1_metrics = {k: float(v[0]) for k, v in hist_dict.items()}
    final_epoch_metrics = {k: float(v[-1]) for k, v in hist_dict.items()}
    epochs_run = len(hist_dict['loss'])
    
    # 5. Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Save Artifacts
    model.save(os.path.join(models_dir, 'trained_diabetes_model.keras'))
    
    for k in hist_dict:
        hist_dict[k] = [float(x) for x in hist_dict[k]]
    with open(os.path.join(models_dir, 'training_history.json'), 'w') as f:
        json.dump(hist_dict, f)
        
    # Output to stdout to capture for ReadMe
    output = {
        "Hyperparameters": {
            "Batch Size": batch_size,
            "Epochs Configured": epochs,
            "Epochs Run": epochs_run,
            "Early Stopping Triggered": epochs_run < epochs,
            "Validation Split": validation_split
        },
        "Metrics_Epoch_1": epoch_1_metrics,
        "Metrics_Final_Epoch": final_epoch_metrics,
        "Test_Metrics": {
            "Accuracy": float(test_accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "Confusion_Matrix": cm.tolist()
        }
    }
    
    print("\n--- PHASE 4 CAPTURE ---")
    print(json.dumps(output))

if __name__ == '__main__':
    main()
