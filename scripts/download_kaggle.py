import subprocess
import glob
import os
import zipfile
import shutil
import argparse

def get_dataset(target: str, input_dir: str = "../data/raw"):
    """
    Adapted from github.com/tttza/kaggle_dataset_downloader
    Optimized for cross-platform (Windows/Linux) and the current project structure.
    """
    # Parse target to extract directory name
    if "/" in target:
        # It's a dataset, e.g. vatsalmavro/spotify-dataset
        dirname = target.split("/")[-1]
        download_command = ['kaggle', 'datasets', 'download', '-d', target]
    else:
        # It's a competition, e.g. titanic
        dirname = target
        download_command = ['kaggle', 'competitions', 'download', '-c', target]

    # Prepare directories
    download_dir = os.path.join(".", "download", dirname)
    final_dir = os.path.join(input_dir, dirname)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # 1. Download
    print(f'Downloading: {target}')
    cmd = download_command + ['-p', download_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Command failed:", " ".join(cmd))
        if len(r.stderr) == 0:
            raise Exception(r.stdout)
        else:
            raise Exception(r.stderr)

    # 2. Unzip and move files
    print('Extracting and moving files...')
    zip_files = glob.glob(os.path.join(download_dir, '*.zip'))
    for zf in zip_files:
        with zipfile.ZipFile(zf, 'r') as zip_ref:
            zip_ref.extractall(final_dir)
            
    # Move non-zip files natively
    files = glob.glob(os.path.join(download_dir, '*'))
    for f in files:
        if not f.endswith('.zip'):
             dest = os.path.join(final_dir, os.path.basename(f))
             shutil.move(f, dest)
             
    # Cleanup temporal download folder
    shutil.rmtree(download_dir, ignore_errors=True)
    
    print(f'Download Completed: {target}')
    print(f'Files saved to: {final_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset based on tttza/kaggle_dataset_downloader")
    parser.add_argument("target", help="Dataset identifier (e.g. 'vatsalmavro/spotify-dataset' or 'titanic')")
    parser.add_argument("--dir", default="../data/raw", help="Output directory")
    args = parser.parse_args()
    
    get_dataset(args.target, args.dir)
