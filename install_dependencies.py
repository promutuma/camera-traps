import os
import sys
import glob
import tarfile
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def clean_line(line):
    # Strip common terminal progress bar blocks/lines to make the text clean
    bar_chars = ["━", "╸", "╺", "█", "░", "▊", "▋", "▌", "▍", "▎", "▏", "▕", "■", "□"]
    for char in bar_chars:
        line = line.replace(char, "")
    line = " ".join(line.split())
    return line

def download_package(package, base_dest_dir, extra_args, progress_dict):
    safe_pkg_name = package.replace(">=", "_gt_").replace("<=", "_lt_").replace("==", "_eq_").replace(" ", "_")
    pkg_dest_dir = os.path.join(base_dest_dir, f"pkg_{safe_pkg_name}")
    os.makedirs(pkg_dest_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", pkg_dest_dir,
        package
    ] + extra_args
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Thread to read stdout in real-time
    def read_stdout(stream):
        buffer = b""
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            buffer += chunk
            while b"\r" in buffer or b"\n" in buffer:
                r_idx = buffer.find(b"\r")
                n_idx = buffer.find(b"\n")
                if r_idx == -1:
                    idx = n_idx
                elif n_idx == -1:
                    idx = r_idx
                else:
                    idx = min(r_idx, n_idx)
                line = buffer[:idx].decode("utf-8", errors="ignore").strip()
                buffer = buffer[idx+1:]
                if line:
                    cleaned = clean_line(line)
                    if cleaned:
                        progress_dict[package] = cleaned
                        
    # Thread to read stderr in real-time
    def read_stderr(stream):
        for line in stream:
            line = line.decode("utf-8", errors="ignore").strip()
            if line:
                if "error" in line.lower() or "warning" in line.lower():
                    # Truncate warning/error lines so they fit nicely
                    progress_dict[package] = f"Status: {line[:60]}..."
                    
    t1 = threading.Thread(target=read_stdout, args=(proc.stdout,))
    t2 = threading.Thread(target=read_stderr, args=(proc.stderr,))
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    
    ret_code = proc.wait()
    t1.join()
    t2.join()
    
    if ret_code != 0:
        progress_dict[package] = "Failed"
        return False, f"Failed to download {package}"
    
    progress_dict[package] = "Downloaded and ready"
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
    
    # 1. Determine packages
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
            
        if not any("ultralytics-yolov5" in pkg for pkg in packages_to_download):
            packages_to_download.append("ultralytics-yolov5==0.1.1")
            
    print(f"[INFO] Initializing parallel download of {len(packages_to_download)} packages...")
    
    # 2. Concurrently download packages
    progress_dict = {pkg: "Pending..." for pkg in packages_to_download}
    
    executor = ThreadPoolExecutor(max_workers=5)
    futures = {
        executor.submit(download_package, pkg, temp_dir, extra_args, progress_dict): pkg 
        for pkg in packages_to_download
    }
    
    num_lines = len(packages_to_download)
    # Print initial spacing lines for moving cursor
    is_tty = sys.stdout.isatty()
    if is_tty:
        for _ in range(num_lines):
            print()
            
    last_output = ""
    last_print_time = 0.0
    
    while not all(f.done() for f in futures):
        # Build progress status
        status_lines = []
        for pkg in packages_to_download:
            status = progress_dict.get(pkg, "Pending...")
            status_lines.append(f"  - {pkg}: {status}")
            
        current_output = "\n".join(status_lines)
        if current_output != last_output:
            if is_tty:
                sys.stdout.write(f"\033[{num_lines}A")
                for line in status_lines:
                    sys.stdout.write(f"\r\033[K{line}\n")
                sys.stdout.flush()
                last_output = current_output
            else:
                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    print("\n--- Download Progress ---")
                    for line in status_lines:
                        print(line)
                    print("-------------------------")
                    last_print_time = current_time
                    last_output = current_output
                    
        time.sleep(0.5)
        
    # Final print status
    status_lines = []
    for pkg in packages_to_download:
        status = progress_dict.get(pkg, "Downloaded and ready")
        status_lines.append(f"  - {pkg}: {status}")
    if is_tty:
        sys.stdout.write(f"\033[{num_lines}A")
        for line in status_lines:
            sys.stdout.write(f"\r\033[K{line}\n")
        sys.stdout.flush()
    else:
        print("\n--- Final Download Status ---")
        for line in status_lines:
            print(line)
        print("-----------------------------")
        
    # Check execution results
    failed = False
    for f in futures:
        success, message = f.result()
        if not success:
            print(f"[ERROR] {message}")
            failed = True
            
    if failed:
        sys.exit(1)
        
    # 3. Consolidate downloads
    print("[INFO] Consolidating downloaded packages...")
    for root, dirs, files in os.walk(temp_dir):
        if root == temp_dir:
            continue
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(temp_dir, file)
            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
            else:
                os.remove(src_path)
                
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
            
        os.remove(archive_path)
        shutil.rmtree(extract_dir)
        
    # 5. Offline install
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
    shutil.rmtree(temp_dir)
    
    if result.returncode != 0:
        print("[ERROR] Package installation failed.")
        sys.exit(1)
        
    print("[OK] All packages installed successfully.")

if __name__ == "__main__":
    main()
