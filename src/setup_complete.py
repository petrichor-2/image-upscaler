"""
Setup script for the latent diffusion super-resolution project
Helps you get started with the complete pipeline
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Look for requirements.txt in current directory or parent directory
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
        print("❌ Could not find requirements.txt file")
        print("Please make sure you're running this script from the correct directory")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    return True

def download_and_process_data():
    """Guide user through data download and processing"""
    print("\n" + "="*60)
    print("DATA SETUP")
    print("="*60)
    
    data_dir = input("Enter directory where you want to store data: ").strip()
    if not data_dir:
        print("No directory specified, using './data'")
        data_dir = "./data"
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Data will be stored in: {data_dir}")
    
    # Download data
    print("\nDownloading chest X-ray data...")
    from batch_download_zips import download_data
    
    try:
        num_batches = int(input("How many batches to download? (1-12, start with 2): "))
        if not (1 <= num_batches <= 12):
            print("Invalid number, using 2 batches")
            num_batches = 2
    except ValueError:
        print("Invalid input, using 2 batches")
        num_batches = 2
    
    try:
        # Ask if user wants to force redownload if data exists
        force_redownload = False
        if os.path.exists(data_dir) and os.listdir(data_dir):
            force_input = input("Force redownload existing data? (y/n): ").strip().lower()
            force_redownload = (force_input == 'y')
        
        download_data(num_batches, data_dir, force_redownload)
        print("✓ Data download process completed")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None
    
    # Process data
    print("\nProcessing data into HR/LR pairs...")
    from process_data import generate_downsampled_pairs
    
    try:
        generate_downsampled_pairs(data_dir)
        print("✓ Data processed successfully")
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None
    
    return data_dir

def test_setup(data_dir):
    """Test that everything is working"""
    print("\n" + "="*60)
    print("TESTING SETUP")
    print("="*60)
    
    # First check if we have any images
    hr_folder = os.path.join(data_dir, "HR_256")
    lr_folder = os.path.join(data_dir, "LR_64")
    
    if not os.path.exists(hr_folder) or not os.path.exists(lr_folder):
        print("❌ HR_256 or LR_64 folders not found")
        return False
    
    hr_files = [f for f in os.listdir(hr_folder) if f.lower().endswith('.png')]
    lr_files = [f for f in os.listdir(lr_folder) if f.lower().endswith('.png')]
    
    if not hr_files or not lr_files:
        print("❌ No PNG files found in HR/LR folders")
        return False
    
    print(f"✓ Found {len(hr_files)} HR images and {len(lr_files)} LR images")
    
    try:
        from process_data import get_data_loaders
        data_loaders = get_data_loaders(data_dir, batch_size=2, get_test=False)
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"✓ Train dataset: {len(train_loader.dataset)} samples")
        print(f"✓ Val dataset: {len(val_loader.dataset)} samples")
        
        # Test loading a batch
        if len(train_loader.dataset) > 0:
            lr_batch, hr_batch = next(iter(train_loader))
            print(f"✓ Batch shapes - LR: {lr_batch.shape}, HR: {hr_batch.shape}")
        else:
            print("❌ No samples in train dataset")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing setup: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 LATENT DIFFUSION SUPER-RESOLUTION SETUP")
    print("="*60)
    
    print("This script will help you:")
    print("1. Install required packages")
    print("2. Download and process chest X-ray data")
    print("3. Test the complete pipeline")
    print("4. Show you how to start training")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Please fix installation issues and try again")
        return
    
    # Step 2: Download and process data
    data_dir = download_and_process_data()
    if not data_dir:
        print("Please fix data issues and try again")
        return
    
    # Step 3: Test setup
    if not test_setup(data_dir):
        print("Please fix setup issues and try again")
        return
    
    # Step 4: Instructions
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print(f"📁 Data location: {data_dir}")
    print("\nNext steps:")
    print("1. Visualize your data:")
    print("   python quick_visualize.py")
    print()
    print("2. Start training:")
    print("   python train_latent_diffusion.py")
    print()
    print("3. After training, run inference:")
    print("   python inference.py")
    print()
    print("Tips:")
    print("- Start with a small number of epochs (10-20) to test")
    print("- Monitor GPU memory usage")
    print("- The model will save checkpoints every 10 epochs")
    print("- Training curves will be saved as 'training_curves.png'")
    print("- Run all commands from the 'src' directory")
    print("- Make sure you have activated your virtual environment")

if __name__ == "__main__":
    main()
