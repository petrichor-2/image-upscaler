"""
Setup script for the project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install packages"""
    print("Installing packages...")
    
    # Look for requirements.txt
    requirements_paths = [
        "requirements.txt",
        "../requirements.txt",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    ]
    
    requirements_file = None
    for path in requirements_paths:
        if os.path.exists(path):
            requirements_file = path
            break
    
    if not requirements_file:
        print("Could not find requirements.txt")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed")
    except subprocess.CalledProcessError:
        print("Failed to install requirements")
        return False
    return True

def download_and_process_data():
    """Download and process data"""
    print("\nDATA SETUP")
    
    data_dir = input("Enter directory for data: ").strip()
    if not data_dir:
        print("Using './data'")
        data_dir = "./data"
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Data will be stored in: {data_dir}")
    
    # Download data
    print("\nDownloading data...")
    from batch_download_zips import download_data
    
    try:
        num_batches = int(input("How many batches? (1-12): "))
        if not (1 <= num_batches <= 12):
            num_batches = 2
    except ValueError:
        num_batches = 2
    
    try:
        force_redownload = False
        if os.path.exists(data_dir) and os.listdir(data_dir):
            force_input = input("Redownload existing data? (y/n): ").strip().lower()
            force_redownload = (force_input == 'y')
        
        download_data(num_batches, data_dir, force_redownload)
        print("Data download completed")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Process data
    print("\nProcessing data...")
    from process_data import generate_downsampled_pairs
    
    try:
        generate_downsampled_pairs(data_dir)
        print("Data processed")
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
    
    return data_dir

def test_setup(data_dir):
    """Test setup"""
    print("\nTESTING SETUP")
    
    # Check if we have images
    hr_folder = os.path.join(data_dir, "HR_256")
    lr_folder = os.path.join(data_dir, "LR_64")
    
    if not os.path.exists(hr_folder) or not os.path.exists(lr_folder):
        print("HR_256 or LR_64 folders not found")
        return False
    
    hr_files = [f for f in os.listdir(hr_folder) if f.lower().endswith('.png')]
    lr_files = [f for f in os.listdir(lr_folder) if f.lower().endswith('.png')]
    
    if not hr_files or not lr_files:
        print("No PNG files found")
        return False
    
    print(f"Found {len(hr_files)} HR images and {len(lr_files)} LR images")
    
    try:
        from process_data import get_data_loaders
        data_loaders = get_data_loaders(data_dir, batch_size=2, get_test=False, augment_train=True)
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Val: {len(val_loader.dataset)} samples")
        
        # Test loading a batch
        if len(train_loader.dataset) > 0:
            lr_batch, hr_batch = next(iter(train_loader))
            print(f"Batch shapes - LR: {lr_batch.shape}, HR: {hr_batch.shape}")
        else:
            print("No samples in train dataset")
            return False
        
        return True
    except Exception as e:
        print(f"Error testing setup: {e}")
        return False

def main():
    """Main setup function"""
    print("SETUP SCRIPT")
    
    print("This will:")
    print("1. Install packages")
    print("2. Download data") 
    print("3. Test everything")
    
    # Install requirements
    if not install_requirements():
        print("Fix installation issues and try again")
        return
    
    # Download and process data
    data_dir = download_and_process_data()
    if not data_dir:
        print("Fix data issues and try again")
        return
    
    # Test setup
    if not test_setup(data_dir):
        print("Fix setup issues and try again")
        return
    
    # Instructions
    print("\nSETUP COMPLETE!")
    print(f"Data location: {data_dir}")
    print("\nNext:")
    print("python train_latent_diffusion.py")
    print("python inference.py")

if __name__ == "__main__":
    main()
