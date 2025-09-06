import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    min_detection_confidence=0.3  
)

def extract_enhanced_features(hand_landmarks):
    
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    z_vals = [lm.z for lm in hand_landmarks.landmark]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    
    norm_coords = []
    for lm in hand_landmarks.landmark:
        norm_x = (lm.x - min_x) / (max_x - min_x + 1e-6)
        norm_y = (lm.y - min_y) / (max_y - min_y + 1e-6)
        norm_z = (lm.z - min_z) / (max_z - min_z + 1e-6)
        norm_coords.append([norm_x, norm_y, norm_z])
    

    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    fingertips = [4, 8, 12, 16, 20]  
    distances = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        dist = np.linalg.norm(tip_coords - wrist)  
        distances.append(dist)
    
    
    angles = []
    
    finger_joints = [
        [1, 2, 3], [2, 3, 4],  
        [5, 6, 7], [6, 7, 8], 
        [9, 10, 11], [10, 11, 12],  
        [13, 14, 15], [14, 15, 16],  
        [17, 18, 19], [18, 19, 20]   
    ]
    
    for joint in finger_joints:
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
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        

        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)

            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)  
            angles.append(angle)
        else:
            angles.append(0)
    

    palm_center = np.mean([
        [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z],  
        [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z], 
        [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z] 
    ], axis=0)
    
    relative_positions = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        rel_pos = tip_coords - palm_center  
        relative_positions.extend(rel_pos)  
    

    hand_width = max_x - min_x
    hand_height = max_y - min_y
    hand_area = hand_width * hand_height  
    
    features = []
    
    
    for coord in norm_coords:
        features.extend(coord)
    

    features.extend(distances)
    
    features.extend(angles)
    

    features.extend(relative_positions)
    
    features.append(hand_area)
    
    return features

def process_dataset(dataset_dir, max_samples_per_class=None, force_reprocess=False):
    
    pickle_file = 'processed_data.pickle'
    if os.path.exists(pickle_file) and not force_reprocess:
        print(f"Loading processed data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict['data'], data_dict['labels']
    
    print(f"Processing dataset from {dataset_dir}...")
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Directory {dataset_dir} does not exist!")
        return [], []
    
    class_folders = [d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not class_folders:
        print(f"ERROR: No class folders found in {dataset_dir}")
        return [], []
    
    print(f"Found {len(class_folders)} class folders: {class_folders}")
    

    data = []
    labels = []
    class_counts = {}
    
    
    for class_folder in class_folders:
        class_path = os.path.join(dataset_dir, class_folder)
        class_counts[class_folder] = 0
        

        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"WARNING: No images found in class folder {class_folder}")
            continue
        
        print(f"Processing class {class_folder}: {len(image_files)} images")
        
  
        if max_samples_per_class and len(image_files) > max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
            print(f"Limited to {max_samples_per_class} images for class {class_folder}")
        
     
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            
   
            img = cv2.imread(img_path)
            if img is None:
                print(f"WARNING: Could not read image {img_path}")
                continue
            

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
     
            results = hands.process(img_rgb)
            

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(hand_landmarks.landmark) == 21:
                        features = extract_enhanced_features(hand_landmarks)
                        data.append(features)
                        labels.append(class_folder)
                        class_counts[class_folder] += 1
            else:
                if i % 100 == 0:
                    print(f"No hand landmarks detected in {img_path}")
            
            if (i + 1) % 100 == 0 or i + 1 == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images for class {class_folder}")
    
    print("\nSummary of processed data:")
    total_images = 0
    for class_name, count in class_counts.items():
        print(f"  Class {class_name}: {count} images")
        total_images += count
    
    print(f"Total: {total_images} images processed")
    
    if total_images > 0:
        with open(pickle_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"Saved processed data to {pickle_file}")
    
    return data, labels

def train_and_evaluate(data, labels, test_size=0.2, param_tuning=False):
   
    if len(data) == 0:
        print("ERROR: No data available for training")
        return None
    
    
    X = np.array(data)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y 
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
  
    if param_tuning:
        from sklearn.model_selection import RandomizedSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],  #
            'max_depth': [None, 10, 20, 30, 40],  
            'min_samples_split': [2, 5, 10],  
            'min_samples_leaf': [1, 2, 4],  
            'max_features': ['sqrt', 'log2', None] 
        }
        
        print("Starting hyperparameter tuning...")

        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_grid,
            n_iter=20, 
            cv=3,  
            verbose=1,
            n_jobs=-1,  
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")
        
        model = random_search.best_estimator_
    else:
        print("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  
        
        print("\nTop 10 most important features:")
        for i in range(min(10, len(importances))):
            print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    with open('rf_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved to 'rf_model.p'")
    
    return model

if __name__ == "__main__":
    import argparse
    
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
    

    data, labels = process_dataset(
        dataset_dir=args.dataset,
        max_samples_per_class=args.max_samples,
        force_reprocess=args.force
    )
    

    if len(data) > 0:
        train_and_evaluate(
            data=data,
            labels=labels,
            test_size=args.test_size,
            param_tuning=args.tune
        )
    else:
        print("No data available for training. Exiting.")
