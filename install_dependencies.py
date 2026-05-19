import os
import sys
import glob
import tarfile
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_package(package, base_dest_dir, extra_args):
    # Create an isolated sub-directory for this package to avoid filename collisions/concurrent write corruption
    safe_pkg_name = package.replace(">=", "_gt_").replace("<=", "_lt_").replace("==", "_eq_").replace(" ", "_")
    pkg_dest_dir = os.path.join(base_dest_dir, f"pkg_{safe_pkg_name}")
    os.makedirs(pkg_dest_dir, exist_ok=True)
    
    print(f"[INFO] Downloading {package}...")
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", pkg_dest_dir,
        package
    ] + extra_args
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return False, f"Failed to download {package}: {result.stderr}"
    return True, f"Successfully downloaded {package}"

def main():
    requirements_file = "requirements.txt"
    extra_args = []
    is_conda = False
    
    # Parse command line arguments
    args = sys.argv[1:]
    if "--conda" in args:
        is_conda = True
        args.remove("--conda")
    extra_args = args
    
    temp_dir = "temp_packages_download"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Determine what packages to download
    packages_to_download = []
    if is_conda:
        packages_to_download = ["megadetector>=5.0.0", "ultralytics-yolov5==0.1.1"]
    else:
        if os.path.exists(requirements_file):
            with open(requirements_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        packages_to_download.append(line)
        else:
            print(f"[ERROR] Requirements file {requirements_file} not found.")
            sys.exit(1)
            
        # Ensure ultralytics-yolov5 is explicitly downloaded so we can patch it
        if not any("ultralytics-yolov5" in pkg for pkg in packages_to_download):
            packages_to_download.append("ultralytics-yolov5==0.1.1")
            
    print(f"[INFO] Starting parallel download of {len(packages_to_download)} packages...")
    
    # 2. Download packages concurrently (using max_workers=5)
    failed = False
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_package, pkg, temp_dir, extra_args): pkg for pkg in packages_to_download}
        for future in as_completed(futures):
            pkg = futures[future]
            success, message = future.result()
            print(message)
            if not success:
                failed = True
                
    if failed:
        print("[ERROR] One or more package downloads failed.")
        sys.exit(1)
        
    # 3. Consolidate all downloaded files from sub-directories into the main temp directory
    print("[INFO] Consolidating downloaded packages...")
    for root, dirs, files in os.walk(temp_dir):
        # Skip the root temp directory itself
        if root == temp_dir:
            continue
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(temp_dir, file)
            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
            else:
                # If already exists, just remove the duplicate in the subdirectory
                os.remove(src_path)
                
    # Clean up empty subdirectories
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        if root != temp_dir:
            try:
                os.rmdir(root)
            except OSError:
                pass
                
    # 4. Find and patch ultralytics-yolov5
    archives = glob.glob(os.path.join(temp_dir, "ultralytics-yolov5-*.tar.gz"))
    if not archives:
        wheels = glob.glob(os.path.join(temp_dir, "ultralytics_yolov5-*.whl"))
        if wheels:
            print("[INFO] ultralytics-yolov5 already exists as a wheel. Skipping patch.")
        else:
            print("[ERROR] Could not find ultralytics-yolov5 package in downloads.")
            sys.exit(1)
    else:
        archive_path = archives[0]
        extract_dir = os.path.join(temp_dir, "extracted")
        print(f"[INFO] Extracting {archive_path} for patching...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            
        setup_files = glob.glob(os.path.join(extract_dir, "*", "setup.py"))
        if not setup_files:
            print("[ERROR] Could not find setup.py in extracted ultralytics-yolov5.")
            sys.exit(1)
        setup_path = setup_files[0]
        pkg_dir = os.path.dirname(setup_path)
        
        print(f"[INFO] Patching setup.py in {setup_path}...")
        with open(setup_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        target = "README = request.urlopen('https://raw.githubusercontent.com/ultralytics/yolov5/master/README.md').read().decode('utf-8')"
        replacement = "README = 'ultralytics-yolov5 description'"
        
        if target in content:
            content = content.replace(target, replacement)
        else:
            import re
            content, count = re.subn(
                r"README\s*=\s*request\.urlopen\([^)]+\)\.read\(\)\.decode\([^)]+\)",
                "README = 'ultralytics-yolov5 description'",
                content
            )
            
        with open(setup_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Build a wheel from the patched folder
        print("[INFO] Building wheel from patched ultralytics-yolov5...")
        wheel_cmd = [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps",
            "--wheel-dir", temp_dir,
            pkg_dir
        ] + extra_args
        
        result = subprocess.run(wheel_cmd)
        if result.returncode != 0:
            print("[ERROR] Failed to build wheel for patched ultralytics-yolov5.")
            sys.exit(1)
            
        # Clean up the source archive and extracted folder
        os.remove(archive_path)
        shutil.rmtree(extract_dir)
        
    # 5. Install all packages offline
    print("[INFO] Installing packages offline from local cache...")
    if is_conda:
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-index",
            "--find-links", temp_dir,
            "megadetector>=5.0.0"
        ] + extra_args
    else:
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-index",
            "--find-links", temp_dir,
            "-r", requirements_file
        ] + extra_args
        
    result = subprocess.run(install_cmd)
    
    # Clean up download directory
    shutil.rmtree(temp_dir)
    
    if result.returncode != 0:
        print("[ERROR] Package installation failed.")
        sys.exit(1)
        
    print("[OK] All packages installed successfully.")

if __name__ == "__main__":
    main()
