"""
Phase 2.2: Generate Training Data for Tackle Model from Video Logs

This script extracts context-action pairs from Court_code2.py execution logs
and generates a CSV suitable for training the lightweight tackle model.

Expected input:
  - CONFIRMED_EVENT_LOG from Court_code2.py (JSON format, dumped per video)
  - Scene graph contexts and action results from process_frame_actions calls

Output:
  - CSV with columns: contact_conf, nearby_count, raider_speed, containment_angle, label
  
NOTE: This is a helper template. You'll need to:
  1. Modify Court_code2.py to dump structured logs (JSON per frame)
  2. Run videos through Court_code2.py to generate logs
  3. Run this script to create training CSV
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def extract_tackle_training_data(log_dir, output_csv="tackle_training_data.csv"):
    """
    Extract tackle detection training data from video processing logs.
    
    Args:
        log_dir: Directory containing JSON logs from Court_code2.py
        output_csv: Output CSV path
    """
    
    print(f"Extracting tackle training data from {log_dir}...\n")
    
    training_samples = []
    
    # Scan for JSON log files
    log_files = list(Path(log_dir).glob("*.json"))
    print(f"Found {len(log_files)} log files")
    
    for log_file in log_files:
        print(f"  Processing {log_file.name}...")
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            # logs format: list of frame dicts with:
            #   - context: scene_graph info
            #   - action_results: from process_frame_actions
            #   - confirmed_events: labels
            
            for frame_log in logs:
                if "action_results" not in frame_log:
                    continue
                
                action_results = frame_log["action_results"]
                context = frame_log.get("context", {})
                
                # Extract features
                contact_conf = float(context.get("contact_confidence", 0.0))
                nearby_count = float(context.get("nearby_defender_count", 0.0))
                raider_speed = float(context.get("raider_speed", 0.0))
                containment_angle = float(context.get("containment_angle", 0.0))
                
                # Determine label: 1 if DEFENDER_TACKLE or RAIDER_CAUGHT in actions
                tackle_actions = {"DEFENDER_TACKLE", "RAIDER_CAUGHT", "SUPER_TACKLE_TRIGGER"}
                label = 1 if any(a["type"] in tackle_actions for a in action_results.get("actions", [])) else 0
                
                training_samples.append({
                    "contact_conf": contact_conf,
                    "nearby_count": nearby_count,
                    "raider_speed": raider_speed,
                    "containment_angle": containment_angle,
                    "label": label,
                })
        
        except Exception as e:
            print(f"    Warning: {e}")
    
    # Write CSV
    if training_samples:
        print(f"\n✓ Extracted {len(training_samples)} training samples")
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "contact_conf", "nearby_count", "raider_speed", "containment_angle", "label"
            ])
            writer.writeheader()
            writer.writerows(training_samples)
        
        print(f"✓ Saved to {output_csv}\n")
        
        # Print statistics
        positive = sum(1 for s in training_samples if s["label"] == 1)
        negative = len(training_samples) - positive
        print(f"Dataset statistics:")
        print(f"  Total samples: {len(training_samples)}")
        print(f"  Positive (tackle): {positive} ({100*positive/len(training_samples):.1f}%)")
        print(f"  Negative (no tackle): {negative} ({100*negative/len(training_samples):.1f}%)")
    else:
        print(f"⚠ No training samples extracted. Check log format.\n")


def create_dummy_training_data(output_csv="tackle_training_data.csv", num_samples=500):
    """
    Create dummy training data for testing.
    
    This generates synthetic context-action pairs for quick validation.
    Replace with real data from Court_code2.py logs before actual training.
    """
    
    print(f"Generating {num_samples} dummy training samples...\n")
    
    import random
    random.seed(42)
    
    samples = []
    for _ in range(num_samples):
        # Contact confidence: higher when tackle happens
        contact_conf = random.uniform(0.0, 0.8)
        
        # Nearby defenders: more defenders when tackle happens
        nearby_count = random.randint(0, 6)
        
        # Raider speed: slower when caught
        raider_speed = random.uniform(0.1, 0.5)
        
        # Containment angle: higher when tackle happens
        containment_angle = random.uniform(0.0, 1.0)
        
        # Generate label based on features (simple heuristic for demo)
        # Tackle more likely if: high contact, 2+ defenders, low speed, good containment
        tackle_score = (
            0.3 * contact_conf + 
            0.3 * (nearby_count / 6.0) + 
            0.2 * (1.0 - raider_speed / 0.5) + 
            0.2 * containment_angle
        )
        label = 1 if tackle_score > 0.55 else 0
        
        samples.append({
            "contact_conf": round(contact_conf, 3),
            "nearby_count": nearby_count,
            "raider_speed": round(raider_speed, 3),
            "containment_angle": round(containment_angle, 3),
            "label": label,
        })
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "contact_conf", "nearby_count", "raider_speed", "containment_angle", "label"
        ])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"✓ Created {output_csv}")
    
    # Statistics
    positive = sum(1 for s in samples if s["label"] == 1)
    negative = len(samples) - positive
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Positive (tackle): {positive} ({100*positive/len(samples):.1f}%)")
    print(f"  Negative (no tackle): {negative} ({100*negative/len(samples):.1f}%)")
    print(f"\n⚠ NOTE: This is dummy data. Replace with real logs from Court_code2.py!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate tackle training data")
    parser.add_argument("--log-dir", type=str, help="Directory with JSON logs from Court_code2.py")
    parser.add_argument("--output", type=str, default="tackle_training_data.csv", help="Output CSV path")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data for testing")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of dummy samples")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PHASE 2.2: TRAINING DATA GENERATION")
    print("="*70 + "\n")
    
    if args.dummy:
        create_dummy_training_data(args.output, args.num_samples)
    elif args.log_dir:
        extract_tackle_training_data(args.log_dir, args.output)
    else:
        print("Usage:")
        print("  Real data:   python generate_tackle_data.py --log-dir path/to/logs")
        print("  Dummy data:  python generate_tackle_data.py --dummy\n")
