import os
import sys
import glob
import tarfile
import shutil
import subprocess
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor

# Regex to match contiguous segments of progress bar characters
bar_pattern = re.compile(r"([━╸╺█░▊▋▌▍▎▏▕■□]+)")

# Regex to match package names in pip's download logs
pattern = re.compile(r"(?:Downloading|Using cached)\s+([a-zA-Z0-9_\-\.]+?)(?:-py[30]|-cp[30]|\.metadata|\.whl|\.tar\.gz|\.zip|\s*\()")
collecting_pattern = re.compile(r"Collecting\s+([a-zA-Z0-9_\-\.]+)")

def clean_line(line):
    return " ".join(line.split())

def extract_pkg_name(line):
    m = pattern.search(line)
    if m:
        return m.group(1).rstrip("-")
    m2 = collecting_pattern.search(line)
    if m2:
        pkg = m2.group(1)
        # Split on common dependency specifiers
        pkg = re.split(r"[><= \(\[#]", pkg)[0]
        return pkg
    return None

def format_status(pkg, status_obj, is_tty, cols):
    pkg_str = f"  - {pkg}: "
    phase = "Downloading"
    bar = ""
    meta = ""
    
    if isinstance(status_obj, str):
        if status_obj == "Pending...":
            phase = "Pending"
        elif status_obj == "Downloaded and ready":
            phase = "Downloaded"
            meta = "✓ Done"
        elif status_obj == "Failed":
            phase = "Failed"
            meta = "✗ Error occurred"
        else:
            phase = "Resolving"
            meta = status_obj
    else:
        phase = status_obj.get("phase", "Downloading")
        bar = status_obj.get("bar", "")
        meta = status_obj.get("meta", "")

    # Allocate space based on terminal width cols (minus 1 to prevent edge wrapping)
    budget = cols - 1 - len(pkg_str)
    phase_text = f"[{phase}]"
    meta_text = meta
    bar_text = bar
    
    current_len = len(phase_text) + (1 if bar_text else 0) + len(bar_text) + (1 if meta_text else 0) + len(meta_text)
    if current_len > budget:
        if bar_text:
            allowed_bar_len = budget - len(phase_text) - len(meta_text) - 2
            if allowed_bar_len >= 5:
                bar_text = bar_text[:allowed_bar_len]
            else:
                bar_text = ""
                
        current_len = len(phase_text) + (1 if bar_text else 0) + len(bar_text) + (1 if meta_text else 0) + len(meta_text)
        if current_len > budget:
            allowed_meta_len = budget - len(phase_text) - (2 if bar_text else 1) - len(bar_text)
            if allowed_meta_len > 3:
                meta_text = meta_text[:allowed_meta_len - 3] + "..."
            else:
                meta_text = ""

    green = "\033[32m"
    yellow = "\033[33m"
    cyan = "\033[36m"
    red = "\033[31m"
    reset = "\033[0m"
    
    if is_tty:
        if phase == "Pending":
            phase_colored = f"{yellow}[Pending]{reset}"
        elif phase == "Downloaded":
            phase_colored = f"{green}[Downloaded]{reset}"
        elif phase == "Failed":
            phase_colored = f"{red}[Failed]{reset}"
        elif phase == "Using Cache":
            phase_colored = f"{green}[Using Cache]{reset}"
        elif phase == "Resolving":
            phase_colored = f"{cyan}[Resolving]{reset}"
        else:
            phase_colored = f"{cyan}[{phase}]{reset}"
            
        bar_colored = f"{green}{bar_text}{reset}" if bar_text else ""
        
        if "✓" in meta_text:
            meta_text = meta_text.replace("✓", f"{green}✓{reset}")
        if "✗" in meta_text:
            meta_text = meta_text.replace("✗", f"{red}✗{reset}")
            
        parts = [phase_colored]
        if bar_colored:
            parts.append(bar_colored)
        if meta_text:
            parts.append(meta_text)
            
        return pkg_str + " ".join(parts)
    else:
        parts = [phase_text]
        if bar_text:
            parts.append(bar_text)
        if meta_text:
            parts.append(meta_text)
        return pkg_str + " ".join(parts)

def download_package(package, base_dest_dir, extra_args, progress_dict, tracked_items, list_lock):
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
    
    active_item = None
    
    def read_stdout(stream):
        nonlocal active_item
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
                    pkg = extract_pkg_name(cleaned)
                    if pkg:
                        with list_lock:
                            if pkg not in progress_dict:
                                tracked_items.append(pkg)
                                progress_dict[pkg] = "Pending..."
                        if active_item and active_item != pkg:
                            if isinstance(progress_dict.get(active_item), dict):
                                progress_dict[active_item] = "Downloaded and ready"
                        active_item = pkg
                        
                    if active_item:
                        match = bar_pattern.search(cleaned)
                        if match:
                            bar = match.group(1)
                            meta = cleaned[match.end():].strip()
                            progress_dict[active_item] = {
                                "phase": "Downloading",
                                "bar": bar,
                                "meta": meta
                            }
                        else:
                            if "cached" in cleaned.lower():
                                progress_dict[active_item] = {
                                    "phase": "Using Cache",
                                    "bar": "",
                                    "meta": cleaned
                                }
                            elif "downloading" in cleaned.lower():
                                progress_dict[active_item] = {
                                    "phase": "Downloading",
                                    "bar": "",
                                    "meta": cleaned
                                }
                            elif "saved" in cleaned.lower() or "successfully downloaded" in cleaned.lower():
                                progress_dict[active_item] = "Downloaded and ready"
                            else:
                                current_status = progress_dict.get(active_item)
                                if isinstance(current_status, str) or current_status.get("phase") not in ("Downloading", "Using Cache"):
                                    progress_dict[active_item] = {
                                        "phase": "Resolving",
                                        "bar": "",
                                        "meta": cleaned
                                    }
                        
    def read_stderr(stream):
        nonlocal active_item
        for line in stream:
            line = line.decode("utf-8", errors="ignore").strip()
            if line:
                if "error" in line.lower() or "warning" in line.lower():
                    target = active_item if active_item else package
                    progress_dict[target] = {
                        "phase": "Warning/Error",
                        "bar": "",
                        "meta": line[:60]
                    }
                    
    t1 = threading.Thread(target=read_stdout, args=(proc.stdout,))
    t2 = threading.Thread(target=read_stderr, args=(proc.stderr,))
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    
    ret_code = proc.wait()
    t1.join()
    t2.join()
    
    if active_item:
        if ret_code == 0:
            progress_dict[active_item] = "Downloaded and ready"
        else:
            progress_dict[active_item] = "Failed"
            
    if ret_code != 0:
        return False, f"Failed to download {package}"
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
            
    print(f"[INFO] Initializing parallel download of dependencies...")
    
    # 2. Concurrently download packages
    tracked_items = []
    progress_dict = {}
    list_lock = threading.Lock()
    
    executor = ThreadPoolExecutor(max_workers=5)
    futures = {
        executor.submit(download_package, pkg, temp_dir, extra_args, progress_dict, tracked_items, list_lock): pkg 
        for pkg in packages_to_download
    }
    
    last_printed_lines = 0
    is_tty = sys.stdout.isatty()
    last_print_time = 0.0
    last_output = ""
    
    while not all(f.done() for f in futures):
        # Fetch current terminal dimensions
        try:
            cols = shutil.get_terminal_size().columns
        except Exception:
            cols = 80

        # Build progress status
        status_lines = []
        with list_lock:
            items_to_print = list(tracked_items)
            
        for pkg in items_to_print:
            status = progress_dict.get(pkg, "Pending...")
            formatted = format_status(pkg, status, is_tty, cols)
            status_lines.append(formatted)
            
        current_output = "\n".join(status_lines)
        if current_output != last_output:
            if is_tty:
                if last_printed_lines > 0:
                    sys.stdout.write(f"\033[{last_printed_lines}A")
                for line in status_lines:
                    sys.stdout.write(f"\r\033[K{line}\n")
                sys.stdout.flush()
                last_printed_lines = len(status_lines)
                last_output = current_output
            else:
                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    print("\n--- Download Progress ---")
                    for pkg in items_to_print:
                        status_obj = progress_dict.get(pkg, "Pending...")
                        formatted_plain = format_status(pkg, status_obj, is_tty=False, cols=80)
                        print(formatted_plain)
                    print("-------------------------")
                    last_print_time = current_time
                    last_output = current_output
                    
        time.sleep(0.5)
        
    # Final clean up states
    for pkg in tracked_items:
        if isinstance(progress_dict.get(pkg), dict):
            progress_dict[pkg] = "Downloaded and ready"

    # Final print status
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80

    status_lines = []
    for pkg in tracked_items:
        status = progress_dict.get(pkg, "Downloaded and ready")
        formatted = format_status(pkg, status, is_tty, cols)
        status_lines.append(formatted)
        
    if is_tty:
        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A")
        for line in status_lines:
            sys.stdout.write(f"\r\033[K{line}\n")
        sys.stdout.flush()
    else:
        print("\n--- Final Download Status ---")
        for pkg in tracked_items:
            status_obj = progress_dict.get(pkg, "Downloaded and ready")
            formatted_plain = format_status(pkg, status_obj, is_tty=False, cols=80)
            print(formatted_plain)
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
