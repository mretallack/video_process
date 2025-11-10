#!/usr/bin/env python3

import os
import glob
import json
import time
import cv2
import numpy as np
import av
import configparser
import re
import shutil
from datetime import datetime
from ultralytics import YOLO



def load_config():
    config = configparser.ConfigParser()
    config_file = 'video_analyzer.ini'
    
    # Default values
    if not os.path.exists(config_file):
        config['PATHS'] = {
            'upload_dir': '/home/mark/.homeassistant/videos/camera_upload',
            'processed_dir': '/home/mark/.homeassistant/videos/processed'
        }
        config['THRESHOLDS'] = {
            'movement_threshold': '30',
            'min_detections': '8',
            'delete_older_than': '259200',
            'delete_originals': 'true'
        }
        config['DETECTION'] = {
            'supported_classes': 'person,car,truck,bus,motorcycle,bicycle,dog,cat,bird,horse,cow'
        }
        with open(config_file, 'w') as f:
            config.write(f)
        #print(f"Created default config file: {config_file}")
    else:
        config.read(config_file)
    
    supported_classes = config.get('DETECTION', 'supported_classes', fallback='person,car,truck,bus,motorcycle,bicycle,dog,cat,bird,horse,cow')
    
    return {
        'upload_dir': config.get('PATHS', 'upload_dir', fallback='/home/mark/.homeassistant/videos/camera_upload'),
        'processed_dir': config.get('PATHS', 'processed_dir', fallback='/home/mark/.homeassistant/videos/processed'),
        'movement_threshold': config.getfloat('THRESHOLDS', 'movement_threshold', fallback=30),
        'min_detections': config.getint('THRESHOLDS', 'min_detections', fallback=8),
        'delete_older_than': config.getint('THRESHOLDS', 'delete_older_than', fallback=259200),
        'delete_originals': config.getboolean('THRESHOLDS', 'delete_originals', fallback=True),
        'supported_classes': [cls.strip() for cls in supported_classes.split(',')]
    }


def delete_video_file(upload_dir, prefix):
    """Delete all files matching the given prefix pattern."""
    #print(f"Removing {prefix}")
    for f in glob.glob(os.path.join(upload_dir, f"{prefix}*")):
        os.remove(f)

def find_timestamp_files(upload_dir):
    """Find all timestamp JSON files (numeric filename pattern) in the upload directory."""
    timestamp_pattern = re.compile(r'^[0-9]+\.json$')
    json_files = [f for f in os.listdir(upload_dir) if timestamp_pattern.match(f)]
    #print(f"Found {len(json_files)} timestamp JSON files")
    return json_files

def process_timestamp_file(filename, upload_dir, model, config):
    """Process a single timestamp file: analyze video1 and video2 separately and save results."""
    basefilename, _ = os.path.splitext(filename)
    json_path = os.path.join(upload_dir, filename)
    
    # Create time-based directory structure
    timestamp = int(basefilename)
    dt = datetime.fromtimestamp(timestamp)
    time_dir = os.path.join(config['processed_dir'], str(dt.year), dt.strftime('%B'), str(dt.day), str(dt.hour))
    os.makedirs(time_dir, exist_ok=True)
    results_json_file = os.path.join(time_dir, f"{basefilename}_results.json")
    
    print(f"\n=== Processing JSON file: {filename} ===")
    
    # Check if file is older than threshold
    file_age = time.time() - int(basefilename)
    if file_age > config['delete_older_than']:
        #print(f"File {filename} is older than threshold, deleting")
        os.remove(json_path)
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'video1' not in data or 'video2' not in data:
        print(f"Missing video1 or video2 in {filename}")
        return False
    
    video1_path = os.path.join(upload_dir, data['video1'])
    video2_path = os.path.join(upload_dir, data['video2'])
    
    if not (os.path.exists(video1_path) and os.path.exists(video2_path)):
        #print(f"Video files not found for {filename}")
        return False
    
    # Process video1 and video2 separately
    moving_count1, detections1, track_movements1, track_detection_counts1 = process_video(video1_path, model, "VIDEO1")
    moving_count2, detections2, track_movements2, track_detection_counts2 = process_video(video2_path, model, "VIDEO2")
    
    total_moving_count = moving_count1 + moving_count2
    
    #print(f"\n=== SUMMARY for {filename} ===")
    #print(f"Video1: {moving_count1} moving objects detected")
    #print(f"Video2: {moving_count2} moving objects detected")
    #print(f"Total: {total_moving_count} moving objects detected")
    
    if total_moving_count == 0:
        #print(f"No movement detected. Deleting files for {filename}")
        delete_video_file(upload_dir, basefilename)
        return False
    
    # Combine results from both videos
    all_detections = detections1 + detections2
    all_track_movements = {**track_movements1, **track_movements2}
    all_track_detection_counts = {**track_detection_counts1, **track_detection_counts2}
    
    # Move video files to time-based processed directory
    processed_video1 = os.path.join(time_dir, data['video1'])
    processed_video2 = os.path.join(time_dir, data['video2'])
    shutil.move(video1_path, processed_video1)
    shutil.move(video2_path, processed_video2)
    
    # Create and save results
    save_results(data, total_moving_count, all_detections, all_track_movements, all_track_detection_counts, 
                data['video1'], data['video2'], results_json_file)
    
    # Delete original JSON file
    os.remove(json_path)
    print(f"Deleted original JSON file: {filename}")
    
    return True

def save_results(data, moving_count, detections, track_movements, track_detection_counts, 
                video1_filename, video2_filename, results_json_file):
    """Create and save the results JSON file with detection data and metadata."""
    json_results = {
        'moving_count': moving_count,
        'track_movements': track_movements,
        'track_detection_counts': track_detection_counts,
        'detections': detections,
        'video1': video1_filename,
        'video2': video2_filename
    }
    
    if 'image' in data:
        json_results['image'] = data['image']
    
    with open(results_json_file, 'w') as f:
        json.dump(json_results, f, indent=4)

def analyze_videos():
    """Main function to analyze all timestamp files in the upload directory."""
    config = load_config()
    upload_dir = config['upload_dir']
    
    try:
        model = YOLO('yolov8n.pt')
        #print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    json_files = find_timestamp_files(upload_dir)
    
    for filename in json_files:
        try:
            process_timestamp_file(filename, upload_dir, model, config)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def process_video(video_path, model, video_type=""):
    """Process a video file using YOLO object detection and tracking.
    Returns: moving_count, detections, track_movements, track_detection_counts
    """
    print(f"Processing: {video_path} ({video_type})")
    
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        frame_count = 0
        trackers = {}
        next_id = 1
        all_detections = []
        
        for packet in container.demux(video_stream):
            try:
                frames = video_stream.decode(packet)
                for frame in frames:
                    frame_count += 1
                    # Convert PyAV frame to numpy array
                    img = frame.to_ndarray(format='bgr24')
                    
                    try:
                        results = model(img, verbose=False)
                        current_detections = []
                        
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    if conf > 0.5:
                                        class_name = model.names[cls]
                                        config = load_config()
                                        if class_name in config['supported_classes']:
                                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                                            center_x = int((x1 + x2) / 2)
                                            center_y = int((y1 + y2) / 2)
                                            detection = {
                                                'frame': frame_count,
                                                'class': class_name,
                                                'confidence': conf,
                                                'center': (center_x, center_y),
                                                'bbox': (x1, y1, x2, y2)
                                            }
                                            current_detections.append(detection)
                                            all_detections.append(detection.copy())
                        
                        # Simple tracking: assign IDs based on proximity
                        for i, detection in enumerate(current_detections):
                            best_match = None
                            min_distance = float('inf')
                            
                            for track_id, tracker in trackers.items():
                                if tracker['class'] == detection['class']:
                                    distance = np.sqrt((tracker['center'][0] - detection['center'][0])**2 + 
                                                     (tracker['center'][1] - detection['center'][1])**2)
                                    if distance < min_distance and distance < 100:  # threshold
                                        min_distance = distance
                                        best_match = track_id
                            
                            if best_match:
                                trackers[best_match]['center'] = detection['center']
                                trackers[best_match]['confidence'] = detection['confidence']
                                track_id = best_match
                            else:
                                track_id = next_id
                                trackers[track_id] = detection.copy()
                                trackers[track_id]['initial_center'] = detection['center']
                                next_id += 1
                            
                            # Update detection with track_id and remove bbox
                            all_detections[len(all_detections) - len(current_detections) + i]['track_id'] = track_id
                            del all_detections[len(all_detections) - len(current_detections) + i]['bbox']
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                        continue
            except (av.InvalidDataError, av.error.InvalidDataError):
                continue  # Skip corrupted packets
        
        container.close()
        
        # Calculate movement and detection density for each track
        moving_objects = set()
        track_movements = {}
        track_detection_counts = {}
        
        # Count detections per track
        for detection in all_detections:
            if 'track_id' in detection:
                track_id = detection['track_id']
                track_detection_counts[track_id] = track_detection_counts.get(track_id, 0) + 1
        
        for track_id, tracker in trackers.items():
            if 'initial_center' in tracker:
                total_movement = np.sqrt((tracker['center'][0] - tracker['initial_center'][0])**2 + 
                                       (tracker['center'][1] - tracker['initial_center'][1])**2)
                track_movements[track_id] = round(total_movement, 2)
                detection_count = track_detection_counts.get(track_id, 0)
                
                # Load config for thresholds
                config = load_config()
                
                # Require both movement AND sufficient detection density
                if total_movement > config['movement_threshold'] and detection_count >= config['min_detections']:
                    moving_objects.add(track_id)
        
        #print(f"Processed {frame_count} frames from {video_path} ({video_type})")
        #print(f"Moving objects detected: {len(moving_objects)}")
        return len(moving_objects), all_detections, track_movements, track_detection_counts
        
    except Exception as e:
        print(f"Error opening video {video_path}: {e}")
        return 0, []

def cleanup_old_results():
    config = load_config()
    processed_dir = config['processed_dir']
    delete_older_than = config['delete_older_than']
    
    # Walk through time-based directory structure
    for root, dirs, files in os.walk(processed_dir):
        for filename in files:
            if filename.endswith('_results.json'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Extract timestamp from filename
                    basefilename = filename.replace('_results.json', '')
                    file_age = time.time() - int(basefilename)
                    
                    if file_age > delete_older_than:
                        # Delete result file
                        os.remove(filepath)
                        
                        # Delete video1 and video2 files
                        if 'video1' in data:
                            video1_path = os.path.join(root, data['video1'])
                            if os.path.exists(video1_path):
                                os.remove(video1_path)
                        
                        if 'video2' in data:
                            video2_path = os.path.join(root, data['video2'])
                            if os.path.exists(video2_path):
                                os.remove(video2_path)
                        
                        # Delete image if it exists
                        if 'image' in data:
                            image_path = os.path.join(root, data['image'])
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                
                except Exception as e:
                    print(f"Error processing result file {filename}: {e}")

if __name__ == "__main__":
    try:
        analyze_videos()
        cleanup_old_results()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
