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
import logging
from datetime import datetime
from ultralytics import YOLO



def setup_logging(log_level):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

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
        config['PROCESSING'] = {
            'frame_scale': '0.5',
            'max_runtime_minutes': '30',
            'frame_skip': '2'
        }
        config['LOGGING'] = {
            'log_level': 'INFO'
        }
        config['DETECTION'] = {
            'supported_classes': 'person,car,truck,bus,motorcycle,bicycle,dog,cat,bird,horse,cow',
            'min_confidence': '0.6'
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
        'frame_scale': config.getfloat('PROCESSING', 'frame_scale', fallback=0.5),
        'max_runtime_minutes': config.getint('PROCESSING', 'max_runtime_minutes', fallback=30),
        'frame_skip': config.getint('PROCESSING', 'frame_skip', fallback=2),
        'min_confidence': config.getfloat('DETECTION', 'min_confidence', fallback=0.6),
        'log_level': config.get('LOGGING', 'log_level', fallback='INFO'),
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
    start_time = time.time()
    basefilename, _ = os.path.splitext(filename)
    json_path = os.path.join(upload_dir, filename)
    
    # Create time-based directory structure
    timestamp = int(basefilename)
    dt = datetime.fromtimestamp(timestamp)
    time_dir = os.path.join(config['processed_dir'], str(dt.year), dt.strftime('%B'), str(dt.day), str(dt.hour))
    os.makedirs(time_dir, exist_ok=True)
    results_json_file = os.path.join(time_dir, f"{basefilename}_results.json")
    
    logger = setup_logging(config['log_level'])
    logger.info(f"Processing JSON file: {filename}")
    
    # Check if file is older than threshold
    file_age = time.time() - int(basefilename)
    if file_age > config['delete_older_than']:
        #print(f"File {filename} is older than threshold, deleting")
        os.remove(json_path)
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'video1' not in data or 'video2' not in data:
        logger.error(f"Missing video1 or video2 in {filename}")
        return False
    
    video1_path = os.path.join(upload_dir, data['video1'])
    video2_path = os.path.join(upload_dir, data['video2'])
    
    if not (os.path.exists(video1_path) and os.path.exists(video2_path)):
        #print(f"Video files not found for {filename}")
        return False
    
    # Process video1 and video2 separately
    moving_count1, detections1, track_movements1, track_detection_counts1 = process_video(video1_path, model, "VIDEO1", config)
    moving_count2, detections2, track_movements2, track_detection_counts2 = process_video(video2_path, model, "VIDEO2", config)
    
    total_moving_count = moving_count1 + moving_count2
    
    processing_time = time.time() - start_time
    logger.info(f"Processing time: {processing_time:.2f} seconds")

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
    
    # Create human-readable filenames
    human_timestamp = dt.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Rename and move video files
    video1_ext = os.path.splitext(data['video1'])[1]
    video2_ext = os.path.splitext(data['video2'])[1]
    new_video1_name = f"{human_timestamp}_video1{video1_ext}"
    new_video2_name = f"{human_timestamp}_video2{video2_ext}"
    
    processed_video1 = os.path.join(time_dir, new_video1_name)
    processed_video2 = os.path.join(time_dir, new_video2_name)
    shutil.move(video1_path, processed_video1)
    shutil.move(video2_path, processed_video2)
    
    # Move and rename image file if it exists
    new_image_name = None
    if 'image' in data:
        image_path = os.path.join(upload_dir, data['image'])
        if os.path.exists(image_path):
            image_ext = os.path.splitext(data['image'])[1]
            new_image_name = f"{human_timestamp}_image{image_ext}"
            processed_image = os.path.join(time_dir, new_image_name)
            shutil.move(image_path, processed_image)
    
    # Create and save results with new filenames
    save_results(data, total_moving_count, all_detections, all_track_movements, all_track_detection_counts, 
                new_video1_name, new_video2_name, new_image_name, results_json_file)
    
    # Delete original JSON file
    os.remove(json_path)
    logger.info(f"Deleted original JSON file: {filename}")
    
    return True

def save_results(data, moving_count, detections, track_movements, track_detection_counts, 
                video1_filename, video2_filename, image_filename, results_json_file):
    """Create and save the results JSON file with detection data and metadata."""
    json_results = {
        'moving_count': moving_count,
        'track_movements': track_movements,
        'track_detection_counts': track_detection_counts,
        'detections': detections,
        'video1': video1_filename,
        'video2': video2_filename
    }
    
    if image_filename:
        json_results['image'] = image_filename
    
    with open(results_json_file, 'w') as f:
        json.dump(json_results, f, indent=4)

def analyze_videos():
    """Main function to analyze all timestamp files in the upload directory."""
    config = load_config()
    upload_dir = config['upload_dir']
    start_time = time.time()
    max_runtime = config['max_runtime_minutes'] * 60
    
    try:
        model = YOLO('yolov8n.pt')
        #print("YOLO model loaded successfully")
    except Exception as e:
        logger = setup_logging(config['log_level'])
        logger.error(f"Error loading YOLO model: {e}")
        return
    
    json_files = find_timestamp_files(upload_dir)
    
    for filename in json_files:
        # Check if maximum runtime exceeded
        if time.time() - start_time > max_runtime:
            logger.warning(f"Maximum runtime of {config['max_runtime_minutes']} minutes exceeded. Stopping processing.")
            break
            
        try:
            process_timestamp_file(filename, upload_dir, model, config)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

def process_video(video_path, model, video_type="", config=None):
    """Process a video file using YOLO object detection and tracking.
    Returns: moving_count, detections, track_movements, track_detection_counts
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing: {video_path} ({video_type})")
    
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
                    # Skip frames for faster processing
                    if config and frame_count % config['frame_skip'] != 0:
                        continue
                    # Convert PyAV frame to numpy array
                    img = frame.to_ndarray(format='bgr24')
                    
                    # Scale frame for faster processing
                    if config and config['frame_scale'] != 1.0:
                        h, w = img.shape[:2]
                        new_h, new_w = int(h * config['frame_scale']), int(w * config['frame_scale'])
                        img = cv2.resize(img, (new_w, new_h))
                        scale_factor = config['frame_scale']
                        if frame_count == 1:  # Log only for first frame
                            logger.debug(f"Frame resized to: {new_w}x{new_h} (scale: {scale_factor})")
                    else:
                        scale_factor = 1.0
                        if frame_count == 1:  # Log only for first frame
                            h, w = img.shape[:2]
                            logger.debug(f"Frame size: {w}x{h} (no scaling)")
                    
                    try:
                        results = model(img, verbose=False)
                        current_detections = []
                        
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    if conf > config['min_confidence']:
                                        class_name = model.names[cls]
                                        if class_name in config['supported_classes']:
                                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                                            # Scale coordinates back to original size
                                            if scale_factor != 1.0:
                                                x1, y1, x2, y2 = x1/scale_factor, y1/scale_factor, x2/scale_factor, y2/scale_factor
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
                        logger.error(f"Error processing frame {frame_count}: {e}")
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
        logger.error(f"Error opening video {video_path}: {e}")
        return 0, []

def find_result_files(directory):
    """Recursively find all _results.json files and return their full paths"""
    result_files = []
    if not os.path.exists(directory):
        return result_files
    
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively search subdirectories
                result_files.extend(find_result_files(item_path))
            elif item.endswith('_results.json'):
                result_files.append(item_path)
    except PermissionError:
        pass
    
    return result_files

def find_empty_directories(directory):
    """Recursively find all empty directories and return their full paths"""
    empty_dirs = []
    
    if not os.path.exists(directory):
        return empty_dirs
    
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively search subdirectories first
                empty_dirs.extend(find_empty_directories(item_path))
                # Check if current directory is empty after recursive cleanup
                if not os.listdir(item_path):
                    empty_dirs.append(item_path)
    except PermissionError:
        pass
    
    return empty_dirs

def cleanup_old_results():
    config = load_config()
    processed_dir = config['processed_dir']
    delete_older_than = config['delete_older_than']
    logger = setup_logging(config['log_level'])
    
    if not os.path.exists(processed_dir):
        return
    
    dirs_to_check = set()
    result_files = find_result_files(processed_dir)
    
    for filepath in result_files:
        filename = os.path.basename(filepath)
        root = os.path.dirname(filepath)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename (original timestamp format)
            basefilename = filename.replace('_results.json', '')
            file_age = time.time() - int(basefilename)
            
            if file_age > delete_older_than:
                logger.info(f"Deleting old result file: {filename}")
                
                # Delete video1 and video2 files
                if 'video1' in data:
                    video1_path = os.path.join(root, data['video1'])
                    if os.path.exists(video1_path):
                        os.remove(video1_path)
                        logger.debug(f"Deleted video1: {data['video1']}")
                
                if 'video2' in data:
                    video2_path = os.path.join(root, data['video2'])
                    if os.path.exists(video2_path):
                        os.remove(video2_path)
                        logger.debug(f"Deleted video2: {data['video2']}")
                
                # Delete image if it exists
                if 'image' in data:
                    image_path = os.path.join(root, data['image'])
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        logger.debug(f"Deleted image: {data['image']}")
                
                # Delete result file
                os.remove(filepath)
                
                # Mark directory for cleanup check
                dirs_to_check.add(root)
                        
        except Exception as e:
            logger.error(f"Error processing result file {filename}: {e}")
    
    # Remove empty directories
    empty_dirs = find_empty_directories(processed_dir)
    for dir_path in empty_dirs:
        try:
            if os.path.exists(dir_path):
                os.rmdir(dir_path)
                logger.debug(f"Removed empty directory: {dir_path}")
        except OSError:
            pass

if __name__ == "__main__":
    try:
        analyze_videos()
        cleanup_old_results()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
