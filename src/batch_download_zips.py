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

def download_data(num_batches, dst_folder):
    """Download the first num_batches zip files and extract them."""
    for idx, link in enumerate(links):
        if idx >= num_batches:
            break
        fn = os.path.join(dst_folder, 'images_%02d.tar.gz' % (idx+1))
        print('downloading'+fn+'...')
        urllib.request.urlretrieve(link, fn)  # download the zip file
        with tarfile.open(fn, 'r:gz') as tar:
            tar.extractall(path=dst_folder)     # extract
        os.remove(fn)  # remove the zip
        # Move all the images inside the extracted folder to the destination folder
        extracted_folder = os.path.join(dst_folder, 'images')
        for filename in os.listdir(extracted_folder):
            src_file = os.path.join(extracted_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            if os.path.isfile(src_file):
                os.rename(src_file, dst_file)
        # Remove the extracted folder
        os.rmdir(extracted_folder)

    print("Download complete. Please check the checksums")
