import os
import argparse
import subprocess
import re

def parse_kaggle_link(link):
    """
    Parses a Kaggle link to extract the dataset or competition identifier.
    Returns a tuple (type, identifier) where type is 'dataset' or 'competition'.
    """
    # Remove trailing slash
    link = link.rstrip('/')
    
    # Check if it's a dataset
    dataset_match = re.search(r'kaggle\.com/datasets/([^/]+/[^/]+)', link)
    if dataset_match:
        return 'dataset', dataset_match.group(1)
    
    # Check if it's a generic link that implies a dataset
    # e.g., https://www.kaggle.com/zsinghrahulk/global-airport-database
    generic_match = re.search(r'kaggle\.com/([^/]+/[^/]+)$', link)
    if generic_match and 'competitions' not in link:
        return 'dataset', generic_match.group(1)
        
    # Check if it's a competition
    comp_match = re.search(r'kaggle\.com/c/([^/]+)', link)
    if not comp_match:
        comp_match = re.search(r'kaggle\.com/competitions/([^/]+)', link)
        
    if comp_match:
        return 'competition', comp_match.group(1)
        
    return None, None

def download_data(link, output_dir):
    """
    Downloads data from Kaggle using the Kaggle CLI.
    """
    req_type, identifier = parse_kaggle_link(link)
    
    if not req_type:
        print(f"Error: Could not parse Kaggle link: {link}")
        print("Expected format: https://www.kaggle.com/datasets/user/dataset-name OR https://www.kaggle.com/c/competition-name")
        return False
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {req_type}: {identifier} into {output_dir}")
    print("Ensure you have your kaggle.json configured (usually in ~/.kaggle/).")
    
    try:
        if req_type == 'dataset':
            # Download dataset and unzip
            cmd = ["kaggle", "datasets", "download", "-d", identifier, "-p", output_dir, "--unzip"]
        else: # competition
            # Download competition data
            cmd = ["kaggle", "competitions", "download", "-c", identifier, "-p", output_dir]
            
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        
        # Unzip if it's a competition (kaggle CLI doesn't have --unzip for comps reliably)
        if req_type == 'competition':
            zip_path = os.path.join(output_dir, f"{identifier}.zip")
            if os.path.exists(zip_path):
                print(f"Unzipping {zip_path}...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                os.remove(zip_path) # Clean up
                
        print(f"Successfully downloaded to {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing kaggle CLI: {e}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: The 'kaggle' CLI tool was not found.")
        print("Please install it running: pip install kaggle")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset or competition using its URL.")
    parser.add_argument("link", help="The full Kaggle URL (e.g. https://www.kaggle.com/datasets/uciml/iris)")
    parser.add_argument("--out", "-o", default="../data/raw", help="Output directory path (default: ../data/raw)")
    
    args = parser.parse_args()
    
    # Handle relative paths from scripts directory
    output_dir = args.out
    if not os.path.isabs(output_dir):
        # Resolve to real path relative to where script is executed, 
        # but standardizing if run from project root or scripts folder
        current_dir = os.path.basename(os.getcwd())
        if current_dir == "scripts":
            output_dir = os.path.abspath(os.path.join(os.getcwd(), args.out))
        else:
            # Assume run from project root, so ../data/raw might need adjustment to data/raw
            # We'll just use what the user passed
            output_dir = os.path.abspath(args.out)
            
    download_data(args.link, output_dir)
