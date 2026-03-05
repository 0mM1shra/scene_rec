import os
import zipfile
import shutil
import glob

def setup_data():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data', 'nuscenes')
    
    # search for zip files in current dir, parent dir, or script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = ['*.zip', '*.tar', '*.tgz']
    possible_zips = []
    for d in ['.', '..', script_dir]:
        for p in patterns:
            possible_zips.extend(glob.glob(os.path.join(d, p)))
    
    if not possible_zips:
        print("No zip/tar/tgz files found in current or parent directory.")
        return

    print(f"Found archives: {possible_zips}")
    os.makedirs(data_dir, exist_ok=True)
    
    for archive in possible_zips:
        print(f"Extracting {archive} to {data_dir}...")
        try:
            shutil.unpack_archive(archive, data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract {archive}: {e}")

    # specific check for mini
    if os.path.exists(os.path.join(data_dir, 'v1.0-mini')):
        print("Success: Found v1.0-mini dataset.")
    elif os.path.exists(os.path.join(data_dir, 'v1.0-trainval')):
        print("Success: Found v1.0-trainval dataset.")
    else:
        print("Warning: Extracted data but didn't find standard v1.0 folder structure. You might need to move files manually.")

if __name__ == "__main__":
    setup_data()
