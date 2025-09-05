import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize MediaPipe hands detection module
# This detects hand landmarks in images (21 key points per hand)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # For processing static images (not video)
    min_detection_confidence=0.3  # Lower threshold for detecting hands in less ideal images
)

def extract_enhanced_features(hand_landmarks):
    """
    Extract comprehensive feature set from hand landmarks for ASL classification.
    
    This function transforms raw MediaPipe hand landmarks into a rich feature set
    designed specifically for hand sign classification. The features capture
    hand shape, finger positions, and spatial relationships.
    
    Feature Categories:
    1. Normalized coordinates - Make features scale-invariant
    2. Fingertip-palm distances - Capture hand configuration
    3. Joint angles - Represent finger bending
    4. Relative positions - Capture spatial relationships between fingers
    5. Hand shape descriptors - Overall hand geometry
    
    Args:
        hand_landmarks: MediaPipe hand landmarks (21 points)
        
    Returns:
        list: Feature vector containing ~90+ engineered features
    """
    # Extract basic coordinates (x, y, z) for each landmark
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    
    # Find min/max values for normalization
    # This makes features scale-invariant (independent of hand size or distance from camera)
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    z_vals = [lm.z for lm in hand_landmarks.landmark]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    
    # 1. Normalized coordinates (scale-invariant)
    # This makes the model work regardless of hand size or position in the image
    norm_coords = []
    for lm in hand_landmarks.landmark:
        # Small epsilon (1e-6) prevents division by zero
        norm_x = (lm.x - min_x) / (max_x - min_x + 1e-6)
        norm_y = (lm.y - min_y) / (max_y - min_y + 1e-6)
        norm_z = (lm.z - min_z) / (max_z - min_z + 1e-6)
        norm_coords.append([norm_x, norm_y, norm_z])
    
    # 2. Calculate distances between fingertips and palm center (wrist)
    # These features capture the configuration of fingers (extended or bent)
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    fingertips = [4, 8, 12, 16, 20]  # Landmark indices for fingertips: thumb, index, middle, ring, pinky
    distances = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        dist = np.linalg.norm(tip_coords - wrist)  # Euclidean distance
        distances.append(dist)
    
    # 3. Calculate angles at each finger joint
    # These angles show how much each finger is bent
    angles = []
    # Define joint triplets for angle calculation (three connected landmarks forming an angle)
    finger_joints = [
        [1, 2, 3], [2, 3, 4],  # Thumb joints
        [5, 6, 7], [6, 7, 8],  # Index finger joints
        [9, 10, 11], [10, 11, 12],  # Middle finger joints
        [13, 14, 15], [14, 15, 16],  # Ring finger joints
        [17, 18, 19], [18, 19, 20]   # Pinky finger joints
    ]
    
    for joint in finger_joints:
        # Create vectors from middle point to the connected points
        v1 = np.array([
            hand_landmarks.landmark[joint[0]].x - hand_landmarks.landmark[joint[1]].x,
            hand_landmarks.landmark[joint[0]].y - hand_landmarks.landmark[joint[1]].y,
            hand_landmarks.landmark[joint[0]].z - hand_landmarks.landmark[joint[1]].z
        ])
        v2 = np.array([
            hand_landmarks.landmark[joint[2]].x - hand_landmarks.landmark[joint[1]].x,
            hand_landmarks.landmark[joint[2]].y - hand_landmarks.landmark[joint[1]].y,
            hand_landmarks.landmark[joint[2]].z - hand_landmarks.landmark[joint[1]].z
        ])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        # Calculate angle if vectors are non-zero
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            # Clip to handle numerical imprecision
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)  # Angle in radians
            angles.append(angle)
        else:
            angles.append(0)
    
    # 4. Relative position of fingertips to palm center
    # These features capture the spatial relationships between fingers
    # Calculate palm center as average of wrist and base of index and pinky fingers
    palm_center = np.mean([
        [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z],  # wrist
        [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z],  # index MCP (base)
        [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z]  # pinky MCP (base)
    ], axis=0)
    
    relative_positions = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        rel_pos = tip_coords - palm_center  # Vector from palm center to fingertip
        relative_positions.extend(rel_pos)  # Add x, y, z components
    
    # 5. Hand shape descriptors
    # Simple metrics that describe overall hand geometry
    hand_width = max_x - min_x
    hand_height = max_y - min_y
    hand_area = hand_width * hand_height  # Approximate hand area in image space
    
    # Combine all features into a single feature vector
    features = []
    
    # Add normalized coordinates (x,y,z for each of the 21 landmarks = 63 features)
    for coord in norm_coords:
        features.extend(coord)
    
    # Add distances from fingertips to wrist (5 features)
    features.extend(distances)
    
    # Add joint angles (10 features)
    features.extend(angles)
    
    # Add relative positions of fingertips to palm center (15 features)
    features.extend(relative_positions)
    
    # Add hand shape descriptors (1 feature)
    features.append(hand_area)
    
    # Total: ~94 features capturing various aspects of hand configuration
    return features

def process_dataset(dataset_dir, max_samples_per_class=None, force_reprocess=False):
    """
    Process ASL dataset images to extract hand features
    
    This function:
    1. Loads images from class folders
    2. Detects hand landmarks using MediaPipe
    3. Extracts features for each detected hand
    4. Organizes data for model training
    
    The function implements caching to avoid reprocessing the same data.
    
    Args:
        dataset_dir: Path to directory containing class folders (one per ASL sign)
        max_samples_per_class: Optional maximum number of samples per class
        force_reprocess: Force reprocessing even if pickle file exists
    
    Returns:
        data: List of feature vectors
        labels: List of class labels
    """
    # Check if processed data already exists to avoid redundant processing
    pickle_file = 'processed_data.pickle'
    if os.path.exists(pickle_file) and not force_reprocess:
        print(f"Loading processed data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict['data'], data_dict['labels']
    
    print(f"Processing dataset from {dataset_dir}...")
    
    # Verify directory exists
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Directory {dataset_dir} does not exist!")
        return [], []
    
    # Get class folders (each folder represents an ASL sign class)
    class_folders = [d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not class_folders:
        print(f"ERROR: No class folders found in {dataset_dir}")
        return [], []
    
    print(f"Found {len(class_folders)} class folders: {class_folders}")
    
    # Initialize data structures for features and labels
    data = []
    labels = []
    class_counts = {}
    
    # Process each class folder
    for class_folder in class_folders:
        class_path = os.path.join(dataset_dir, class_folder)
        class_counts[class_folder] = 0
        
        # Get all image files in this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"WARNING: No images found in class folder {class_folder}")
            continue
        
        print(f"Processing class {class_folder}: {len(image_files)} images")
        
        # Limit samples if specified (useful for quick testing or balanced datasets)
        if max_samples_per_class and len(image_files) > max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
            print(f"Limited to {max_samples_per_class} images for class {class_folder}")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"WARNING: Could not read image {img_path}")
                continue
            
            # Convert to RGB for MediaPipe (MediaPipe expects RGB, OpenCV uses BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe to detect hand landmarks
            results = hands.process(img_rgb)
            
            # If hand landmarks were detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Ensure we have all landmarks (should be 21 per hand)
                    if len(hand_landmarks.landmark) == 21:
                        # Extract features from landmarks
                        features = extract_enhanced_features(hand_landmarks)
                        data.append(features)
                        labels.append(class_folder)
                        class_counts[class_folder] += 1
            else:
                # Report no hand landmarks only occasionally to avoid too many messages
                if i % 100 == 0:
                    print(f"No hand landmarks detected in {img_path}")
            
            # Show progress periodically
            if (i + 1) % 100 == 0 or i + 1 == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images for class {class_folder}")
    
    # Print summary of processed data
    print("\nSummary of processed data:")
    total_images = 0
    for class_name, count in class_counts.items():
        print(f"  Class {class_name}: {count} images")
        total_images += count
    
    print(f"Total: {total_images} images processed")
    
    # Save processed data to avoid reprocessing later
    if total_images > 0:
        with open(pickle_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"Saved processed data to {pickle_file}")
    
    return data, labels

def train_and_evaluate(data, labels, test_size=0.2, param_tuning=False):
    """
    Train a Random Forest model on hand sign features and evaluate performance
    
    This function:
    1. Splits data into training and test sets
    2. Optionally performs hyperparameter tuning
    3. Trains a Random Forest classifier
    4. Evaluates model performance
    5. Saves the trained model
    
    Args:
        data: List of feature vectors
        labels: List of class labels
        test_size: Proportion of data to use for testing
        param_tuning: Whether to perform hyperparameter tuning
    
    Returns:
        The trained model
    """
    if len(data) == 0:
        print("ERROR: No data available for training")
        return None
    
    # Convert lists to numpy arrays for scikit-learn
    X = np.array(data)
    y = np.array(labels)
    
    # Split into train and test sets 
    # stratify=y ensures balanced class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y 
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model (with optional hyperparameter tuning)
    if param_tuning:
        from sklearn.model_selection import RandomizedSearchCV
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300, 500],  # Number of trees
            'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of trees
            'min_samples_split': [2, 5, 10],  # Min samples required to split node
            'min_samples_leaf': [1, 2, 4],  # Min samples required at leaf node
            'max_features': ['sqrt', 'log2', None]  # Number of features to consider
        }
        
        print("Starting hyperparameter tuning...")
        # Use RandomizedSearchCV for efficient parameter space exploration
        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=3,  # Cross-validation folds
            verbose=1,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        
        # Fit the randomized search
        random_search.fit(X_train, y_train)
        
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Use the tuned model
        model = random_search.best_estimator_
    else:
        # Use default model with minimal tuning
        print("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display feature importance for interpretability
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Sort in descending order
        
        print("\nTop 10 most important features:")
        for i in range(min(10, len(importances))):
            print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    # Save the trained model for future use
    with open('rf_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved to 'rf_model.p'")
    
    return model

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Process ASL dataset and train model")
    parser.add_argument('--dataset', type=str, default="asl_alphabet_train", 
                       help='Path to dataset directory')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum samples per class')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of data')
    parser.add_argument('--tune', action='store_true', 
                       help='Perform hyperparameter tuning')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    print(f"Dataset directory: {args.dataset}")
    print(f"Max samples per class: {args.max_samples}")
    print(f"Test size: {args.test_size}")
    
    # Step 1: Process dataset to extract features
    data, labels = process_dataset(
        dataset_dir=args.dataset,
        max_samples_per_class=args.max_samples,
        force_reprocess=args.force
    )
    
    # Step 2: Train and evaluate model
    if len(data) > 0:
        train_and_evaluate(
            data=data,
            labels=labels,
            test_size=args.test_size,
            param_tuning=args.tune
        )
    else:
        print("No data available for training. Exiting.")