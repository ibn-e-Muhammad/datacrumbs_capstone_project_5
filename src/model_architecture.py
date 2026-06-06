import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
import tensorflow as tf

def build_model():
    # 1. Hardware Optimization - Force CPU
    tf.config.set_visible_devices([], 'GPU')
    print("GPU Disabled. Using CPU strictly.")
    
    # 2. Build Sequential ANN
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(8,), name='input_layer'),
        tf.keras.layers.Dense(16, activation='relu', name='hidden_1'),
        tf.keras.layers.Dense(8, activation='relu', name='hidden_2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # 3. Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 4. Print Summary
    model.summary()
    
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Exact Trainable Parameter Count: {trainable_count}")
    
    # 5. Save model architecture
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_json = model.to_json()
    with open(os.path.join(base_dir, 'models', 'model_architecture.json'), "w") as json_file:
        json_file.write(model_json)
        
    print("\nModel architecture saved to models/model_architecture.json")

if __name__ == "__main__":
    build_model()
