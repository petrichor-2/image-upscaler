# ADAPTED FROM CODE PROVIDED WITH DATASET

# #!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import urllib.request
import tarfile
import os

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

def download_data(num_batches, dst_folder, force_redownload=False):
    """Download the first num_batches zip files and extract them."""
    
    # Check if we already have images in the destination folder
    if not force_redownload and os.path.exists(dst_folder):
        existing_images = [f for f in os.listdir(dst_folder) if f.lower().endswith('.png')]
        existing_archives = [f for f in os.listdir(dst_folder) if f.lower().endswith('.tar.gz')]
        
        if existing_images:
            print(f"Found {len(existing_images)} existing PNG images in {dst_folder}")
            user_input = input("Data already exists. Download more? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Skipping download. Using existing data.")
                return
        
        if existing_archives:
            print(f"Found {len(existing_archives)} existing tar.gz files in {dst_folder}")
            extract_input = input("Extract existing tar.gz files? (y/n): ").strip().lower()
            if extract_input == 'y':
                print("Extracting existing archives...")
                corrupted_files = []
                
                for archive in existing_archives:
                    archive_path = os.path.join(dst_folder, archive)
                    try:
                        print(f"Extracting {archive}...")
                        with tarfile.open(archive_path, 'r:gz') as tar:
                            tar.extractall(path=dst_folder)
                        
                        # Move images from extracted folder
                        extracted_folder = os.path.join(dst_folder, 'images')
                        if os.path.exists(extracted_folder):
                            for filename in os.listdir(extracted_folder):
                                src_file = os.path.join(extracted_folder, filename)
                                dst_file = os.path.join(dst_folder, filename)
                                if os.path.isfile(src_file):
                                    os.rename(src_file, dst_file)
                            os.rmdir(extracted_folder)
                        
                        # Remove the archive after extraction
                        os.remove(archive_path)
                        print(f"✓ Extracted and removed {archive}")
                        
                    except Exception as e:
                        print(f"❌ Error extracting {archive}: {e}")
                        corrupted_files.append(archive)
                        
                        # Ask if user wants to delete corrupted file and re-download
                        delete_input = input(f"Delete corrupted {archive} and re-download? (y/n): ").strip().lower()
                        if delete_input == 'y':
                            try:
                                os.remove(archive_path)
                                print(f"✓ Deleted corrupted {archive}")
                            except:
                                print(f"❌ Could not delete {archive}")
                
                # Count final images
                final_images = [f for f in os.listdir(dst_folder) if f.lower().endswith('.png')]
                print(f"✓ Extraction complete. Total images: {len(final_images)}")
                
                # If we have corrupted files, offer to re-download
                if corrupted_files:
                    print(f"\nFound {len(corrupted_files)} corrupted files.")
                    redownload_input = input("Re-download corrupted files? (y/n): ").strip().lower()
                    if redownload_input == 'y':
                        print("Continuing with download process...")
                        # Don't return here, continue to download section
                    else:
                        return
                else:
                    return
    
    downloaded_count = 0
    for idx, link in enumerate(links):
        if idx >= num_batches:
            break
        
        batch_name = f'images_{idx+1:02d}'
        fn = os.path.join(dst_folder, f'{batch_name}.tar.gz')
        
        # Check if this batch was already downloaded by looking for a marker file
        marker_file = os.path.join(dst_folder, f'.{batch_name}_downloaded')
        if not force_redownload and os.path.exists(marker_file):
            print(f'Batch {idx+1} already downloaded, skipping...')
            continue
        
        print(f'Downloading batch {idx+1}/{num_batches}: {batch_name}...')
        print(f'URL: {link}')
        print(f'Size: This may take several minutes depending on your connection...')
        
        try:
            # Add a progress callback for urllib
            def progress_callback(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    print(f'\rDownloading: {percent:.1f}%', end='', flush=True)
            
            urllib.request.urlretrieve(link, fn, reporthook=progress_callback)
            print()  # New line after progress
            print(f'✓ Download completed. Extracting...')
            
            with tarfile.open(fn, 'r:gz') as tar:
                tar.extractall(path=dst_folder)
            print(f'✓ Extraction completed.')
            
            os.remove(fn)  # remove the zip
            
            # Move all the images inside the extracted folder to the destination folder
            extracted_folder = os.path.join(dst_folder, 'images')
            if os.path.exists(extracted_folder):
                for filename in os.listdir(extracted_folder):
                    src_file = os.path.join(extracted_folder, filename)
                    dst_file = os.path.join(dst_folder, filename)
                    if os.path.isfile(src_file):
                        os.rename(src_file, dst_file)
                # Remove the extracted folder
                os.rmdir(extracted_folder)
            
            # Create marker file to indicate this batch was downloaded
            with open(marker_file, 'w') as f:
                f.write(f'Downloaded on {os.path.getctime(dst_folder)}\n')
            
            downloaded_count += 1
            print(f'✓ Batch {idx+1} downloaded and extracted successfully')
            
        except Exception as e:
            print(f'❌ Error downloading batch {idx+1}: {e}')
            # Clean up partial downloads
            if os.path.exists(fn):
                os.remove(fn)
            continue
    
    # Summary
    total_images = len([f for f in os.listdir(dst_folder) if f.lower().endswith('.png')])
    print(f"\nDownload Summary:")
    print(f"- New batches downloaded: {downloaded_count}")
    print(f"- Total images in directory: {total_images}")
    print("Download complete. Please check the checksums")
