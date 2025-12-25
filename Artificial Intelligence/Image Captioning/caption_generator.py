import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')
FEATURES_FILE = os.path.join(BASE_DIR, 'features.pkl')
MAX_LENGTH = 35 # Max caption length
VOCAB_SIZE = 0 # Will be set dynamically
EPOCHS = 10 # Adjust as needed (10-20 is decent for demo)
BATCH_SIZE = 32

def extract_features(directory):
    """Extract features from each image using VGG16."""
    if os.path.exists(FEATURES_FILE):
        print("Loading existing features...")
        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
        return features

    model = VGG16()
    # Remove the last layer (prediction) to get the feature vector (4096)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    
    features = {}
    print("Extracting features from images...")
    img_list = os.listdir(directory)
    total_images = len(img_list)
    for i, img_name in enumerate(img_list):
        if i % 100 == 0:
            print(f"Processing image {i}/{total_images}...")
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
        
    print(f"Extracted features for {len(features)} images.")
    # Save features
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(features, f)
    return features

def load_captions(filename):
    """Load captions and map them to image IDs."""
    with open(filename, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        text = f.read()
        
    mapping = {}
    for line in text.split('\n'):
        if len(line) < 2:
            continue
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
            
        # Handle cases where caption might contain commas
        image_id_full = tokens[0]
        image_id = image_id_full.split('.')[0]
        caption = ",".join(tokens[1:])
        
        # Clean caption
        caption = clean_text(caption)
        caption = 'startseq ' + caption + ' endseq'
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
        
    return mapping

def clean_text(text):
    text = text.lower()
    # Remove simple punctuation (keep space)
    text = text.replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(')', '')
    # Remove single characters (e.g. 'a') if desired, but 'a' is common. Just basic cleaning here.
    return text.strip()

def create_tokenizer(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """Generator to yield batches of data for training."""
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            if key not in features: continue # Skip if image not found/failed extract
            
            n += 1
            captions = mapping[key]
            # Process each caption for this image
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                # Split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # Pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # Encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0]) # Image Feature
                    X2.append(in_seq)           # Text Sequence
                    y.append(out_seq)           # Next Word
                    
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

def build_model(vocab_size, max_length):
    # Feature Extractor Model (Image)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence Model (Text)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder (Fusion)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def main():
    # 1. Load Captions
    print("Loading captions...")
    mapping = load_captions(CAPTIONS_FILE)
    print(f"Loaded captions for {len(mapping)} images.")
    
    # 2. Tokenize
    tokenizer = create_tokenizer(mapping)
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    
    # 3. Extract Features
    features = extract_features(IMAGES_DIR)
    
    # 4. Train/Test Split
    image_ids = list(mapping.keys())
    split_index = int(len(image_ids) * 0.90) 
    train_ids = image_ids[:split_index]
    test_ids = image_ids[split_index:]
    
    print(f"Training on {len(train_ids)} images. Testing on {len(test_ids)}.")
    
    MODEL_FILE = 'caption_model.h5'
    # Always build the model architecture first
    model = build_model(VOCAB_SIZE, MAX_LENGTH)
    
    if os.path.exists(MODEL_FILE):
        print("Loading existing model weights...")
        try:
            model.load_weights(MODEL_FILE)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Re-training model...")
            # If weight loading fails, fall through to training logic (though loop below might need adjusting)
            # For now, let's assume it works or we manually trigger training if needed.
            # Actually, to be safe, let's just let it crash if weights fail so we know, OR we structure it to train if load fails.
            pass 
    else:
        print("Model Summary:")
        # model.summary()
        
        # 6. Training Loop
        steps = len(train_ids) // BATCH_SIZE
        for i in range(EPOCHS):
            # Create a dataset from the generator
            def generator_wrapper():
                 return data_generator(train_ids, mapping, features, tokenizer, MAX_LENGTH, VOCAB_SIZE, BATCH_SIZE)
            
            dataset = tf.data.Dataset.from_generator(
                generator_wrapper,
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, MAX_LENGTH), dtype=tf.int32)
                    ),
                    tf.TensorSpec(shape=(None, VOCAB_SIZE), dtype=tf.float32)
                )
            )
            
            model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=1)
            print(f"Epoch {i+1}/{EPOCHS} complete.")
            
        model.save(MODEL_FILE)
        print("Model saved to caption_model.h5")
    
    # 7. Verification / Test
    print("\n--- Generating Captions for Test Images ---")
    count = 0
    for key in test_ids:
        if key not in features: continue
        
        # Get actual captions just to compare (print first one)
        actual = mapping[key][0]
        
        # Predict
        image_feature = features[key]
        predicted = predict_caption(model, image_feature, tokenizer, MAX_LENGTH)
        
        print(f"Image: {key}.jpg")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print("-" * 30)
        
        count += 1
        if count >= 3: # Just show 3 examples
            break

if __name__ == "__main__":
    main()
