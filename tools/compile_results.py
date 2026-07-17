import re
import os
from collections import defaultdict

def parse_metrics_nested_weather(root_dir="./outputs/all_weather"):
    """Parse mAP, NDS, AP per-class from nested weather/level folders."""
    results = defaultdict(dict)
    
    # Iterate weather dirs (fog, rain, snow, dust)
    for weather_dir in os.listdir(root_dir):
        weather_path = os.path.join(root_dir, weather_dir)
        if not os.path.isdir(weather_path):
            continue
            
        results[weather_dir] = {}
        print(f"Processing {weather_dir}...")
        
        # Iterate levels inside each weather (light_fog, heavy_fog, etc.)
        for level_path in os.listdir(weather_path):
            #print(level_path)
            #level_path = os.path.join(weather_path, level_dir)
            #if not os.path.isdir(level_path):
            #    continue
                

            
                print(level_path)
                metrics = parse_single_file(weather_path+'/'+level_path)
                results[weather_dir][level_path] = metrics
                print(f"  {level_path}: mAP={metrics.get('overall', {}).get('mAP', 'N/A')}")
    
    return results

def parse_single_file(file_path):
    """Parse single metrics file (same as before)."""
    metrics = defaultdict(dict)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Overall metrics
    map_match = re.search(r'mAP:\s*([\d.]+)', content)
    nds_match = re.search(r'NDS:\s*([\d.]+)', content)
    
    if map_match: metrics['overall']['mAP'] = float(map_match.group(1))
    if nds_match: metrics['overall']['NDS'] = float(nds_match.group(1))
    
    # Per-class AP
    class_matches = re.findall(r'(\w+)@R\d+:\s*([\d.]+)', content)
    for class_name, ap in class_matches:
        metrics[class_name]['AP'] = float(ap)
    
    return metrics

# Usage
all_results = parse_metrics_nested_weather("./outputs/all_weather")

# Summary table
print("\nWeather Summary:")
print("Weather\t\tLevel\tmAP\tNDS")
for weather, levels in all_results.items():
    for level, metrics in levels.items():
        mAP = metrics.get('overall', {}).get('mAP', 'N/A')
        NDS = metrics.get('overall', {}).get('NDS', 'N/A')
        print(f"{weather:<10}\t{level:<8}\t{mAP}\t{NDS}")

# Export to CSV
import pandas as pd
rows = []
for weather, levels in all_results.items():
    for level, metrics in levels.items():
        row = {'weather': weather, 'level': level}
        row.update(metrics.get('overall', {}))
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('weather_metrics_summary.csv', index=False)
print("\nSaved to weather_metrics_summary.csv")
