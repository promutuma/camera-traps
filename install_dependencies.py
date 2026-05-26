import os
import sys
import glob
import tarfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import shutil
import subprocess
import threading
import time
import re
import queue

# ── ANSI colours ──────────────────────────────────────────────────────────────
_G = "\033[32m"; _Y = "\033[33m"; _C = "\033[36m"; _R = "\033[31m"
_B = "\033[1m";  _D = "\033[2m";  _Z = "\033[0m"

# ── Pip output regexes ────────────────────────────────────────────────────────
_RE_NAME     = re.compile(r"(?:Downloading|Using cached)\s+([A-Za-z0-9_\-\.]+?)(?:-py[30]|-cp[30]|\.metadata|\.whl|\.tar\.gz|\.zip|\s*\()")
_RE_COLLECT  = re.compile(r"Collecting\s+([A-Za-z0-9_\-\.]+)")
_RE_BAR      = re.compile(r"([━╸╺█░▊▋▌▍▎▏▕■□]+)")
_RE_PROGRESS = re.compile(r"([\d\.]+)\s*/\s*([\d\.]+)\s*(kB|MB|GB)")
_RE_SPEED    = re.compile(r"([\d\.]+)\s*(kB|MB|GB)/s")
_RE_ETA      = re.compile(r"\beta\s+([\d:]+)", re.I)
_RE_FILESIZE = re.compile(r"\(([\d\.]+)\s*(kB|MB|GB)\)")
_RE_ANSI     = re.compile(r"\033\[[0-9;]*m")

MAX_RETRIES  = 3
RETRY_DELAYS = [5, 15, 30]   # seconds between consecutive attempts


def _to_mb(value: float, unit: str) -> float:
    if unit == "kB": return value / 1024
    if unit == "GB": return value * 1024
    return value


def detect_tty() -> bool:
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)
            m = ctypes.c_ulong()
            if k32.GetConsoleMode(h, ctypes.byref(m)):
                k32.SetConsoleMode(h, m.value | 0x0004)
        except Exception:
            pass
    if sys.stdout.isatty():
        return True
    term = os.environ.get("TERM", "").lower()
    if term in ("xterm", "xterm-256color", "screen", "screen-256color",
                "tmux", "tmux-256color", "rxvt", "cygwin"):
        return True
    return bool(os.environ.get("COLORTERM") or os.environ.get("BASH_VERSION") or os.environ.get("SHELL"))


# ── Package state ─────────────────────────────────────────────────────────────

class PkgState:
    __slots__ = ("phase", "bar", "done_mb", "total_mb", "speed", "eta", "retries", "note", "is_sub")

    def __init__(self, is_sub: bool = False):
        self.phase    = "Queued"
        self.bar      = ""
        self.done_mb  = 0.0
        self.total_mb = 0.0
        self.speed    = ""
        self.eta      = ""
        self.retries  = 0
        self.note     = ""
        self.is_sub   = is_sub

_PHASE_COLOR = {
    "Queued":      _Y,
    "Resolving":   _C,
    "Downloading": _C,
    "Using Cache": _G,
    "Done":        _G,
    "Retrying":    _Y,
    "Failed":      _R,
}


def _badge(phase: str, is_tty: bool) -> str:
    text = f"[{phase}]"
    if not is_tty:
        return text
    return f"{_PHASE_COLOR.get(phase, _C)}{text}{_Z}"


def _visible_len(s: str) -> int:
    return len(_RE_ANSI.sub("", s))


def _render_line(name: str, st: PkgState, is_tty: bool, cols: int) -> str:
    prefix = f"  {name}: "
    parts  = [_badge(st.phase, is_tty)]

    if st.bar:
        parts.append(f"{_G}{st.bar}{_Z}" if is_tty else st.bar)

    metas = []
    if st.done_mb and st.total_mb:
        metas.append(f"{st.done_mb:.1f}/{st.total_mb:.1f} MB")
    elif st.total_mb:
        metas.append(f"{st.total_mb:.1f} MB")
    if st.speed:
        metas.append(st.speed)
    if st.eta and st.phase == "Downloading":
        metas.append(f"eta {st.eta}")
    if st.retries:
        tag = f"attempt {st.retries + 1}/{MAX_RETRIES + 1}"
        metas.append(f"{_Y}{tag}{_Z}" if is_tty else tag)
    if st.note and st.phase in ("Failed", "Retrying"):
        metas.append(st.note[:40])

    if metas:
        parts.append(" ".join(metas))

    line    = prefix + " ".join(parts)
    visible = _visible_len(line)
    if visible > cols - 1:
        trim = visible - (cols - 4)
        line = line[:max(len(prefix) + 8, len(line) - trim)] + "..."
    return line


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, is_tty: bool):
        self.is_tty      = is_tty
        self._prev       = 0
        self._last_text  = ""
        self._last_flush = 0.0

    def draw(self, lines: list) -> None:
        text = "\n".join(lines)
        now  = time.monotonic()
        if self.is_tty:
            if text == self._last_text:
                return
            if self._prev:
                sys.stdout.write(f"\033[{self._prev}A")
            for line in lines:
                sys.stdout.write(f"\r\033[K{line}\n")
            sys.stdout.flush()
            self._prev      = len(lines)
            self._last_text = text
        else:
            if now - self._last_flush < 5.0 or text == self._last_text:
                return
            print()
            for line in lines:
                print(line)
            self._last_flush = now
            self._last_text  = text

    def clear(self) -> None:
        if self.is_tty and self._prev:
            sys.stdout.write(f"\033[{self._prev}A")
            for _ in range(self._prev):
                sys.stdout.write("\r\033[K\n")
            sys.stdout.write(f"\033[{self._prev}A")
            sys.stdout.flush()
            self._prev      = 0
            self._last_text = ""

    def println(self, msg: str) -> None:
        self.clear()
        print(msg)


# ── Pip line parser ───────────────────────────────────────────────────────────

def _apply_pip_line(line: str, st: PkgState) -> None:
    lower = line.lower()

    if "using cached" in lower or ("cached" in lower and "download" not in lower):
        st.phase = "Using Cache"
        st.bar   = ""
        return

    m_sz = _RE_FILESIZE.search(line)
    if m_sz and not st.total_mb:
        st.total_mb = _to_mb(float(m_sz.group(1)), m_sz.group(2))

    m_bar = _RE_BAR.search(line)
    if m_bar:
        st.phase = "Downloading"
        st.bar   = m_bar.group(1)
        m_prog   = _RE_PROGRESS.search(line)
        if m_prog:
            st.done_mb  = _to_mb(float(m_prog.group(1)), m_prog.group(3))
            st.total_mb = _to_mb(float(m_prog.group(2)), m_prog.group(3))
        m_spd = _RE_SPEED.search(line)
        if m_spd:
            st.speed = f"{m_spd.group(1)} {m_spd.group(2)}/s"
        m_eta = _RE_ETA.search(line)
        if m_eta:
            st.eta = m_eta.group(1)
        return

    if "download" in lower or "resuming" in lower:
        if st.phase not in ("Downloading", "Using Cache", "Done"):
            st.phase = "Downloading"
    elif "collecting" in lower or "looking in" in lower or "requirement already" in lower:
        if st.phase not in ("Downloading", "Using Cache", "Done"):
            st.phase = "Resolving"
    elif "saved" in lower or "successfully downloaded" in lower:
        st.phase = "Done"


def _extract_name(line: str):
    m = _RE_NAME.search(line)
    if m:
        return m.group(1).rstrip("-")
    m2 = _RE_COLLECT.search(line)
    if m2:
        return re.split(r"[><= \(\[#]", m2.group(1))[0]
    return None


# ── Download worker ───────────────────────────────────────────────────────────

def _worker(work_q, done_evt, temp_dir, extra_args,
            state_map, state_lock, results, results_lock):
    while not done_evt.is_set():
        try:
            pkg, attempt = work_q.get(timeout=0.5)
        except queue.Empty:
            continue

        safe   = re.sub(r"[>=<!. ]", "_", pkg)
        pkgdir = os.path.join(temp_dir, f"dl_{safe}")
        os.makedirs(pkgdir, exist_ok=True)

        with state_lock:
            if pkg not in state_map:
                state_map[pkg] = PkgState()
            st         = state_map[pkg]
            st.phase   = "Resolving"
            st.retries = attempt
            st.note    = ""

        cmd = [sys.executable, "-m", "pip", "download",
               "--progress-bar", "on", "--dest", pkgdir, pkg] + extra_args

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            with state_lock:
                state_map[pkg].phase = "Failed"
                state_map[pkg].note  = str(exc)[:50]
            with results_lock:
                results.append((False, pkg, str(exc)))
            work_q.task_done()
            continue

        stderr_lines = []

        def _read_err(stream):
            for raw in stream:
                stderr_lines.append(raw.decode("utf-8", errors="ignore").strip())

        t_err = threading.Thread(target=_read_err, args=(proc.stderr,), daemon=True)
        t_err.start()

        buf        = b""
        active_sub = None

        while True:
            chunk = proc.stdout.read(64)
            if not chunk:
                break
            buf += chunk
            while True:
                ri = buf.find(b"\r")
                ni = buf.find(b"\n")
                if ri == -1 and ni == -1:
                    break
                idx      = min(x for x in (ri, ni) if x != -1)
                raw_line = buf[:idx].decode("utf-8", errors="ignore").strip()
                buf      = buf[idx + 1:]
                if not raw_line:
                    continue

                cleaned  = " ".join(raw_line.split())
                sub_name = _extract_name(cleaned)

                if sub_name:
                    norm = sub_name.lower().replace("-", "_")
                    with state_lock:
                        matched = None
                        for k in list(state_map.keys()):
                            kn = k.lower().replace("-", "_")
                            if kn == norm or norm.startswith(kn + "_") or kn.startswith(norm + "_"):
                                matched = k
                                break
                        if matched is None:
                            state_map[sub_name] = PkgState(is_sub=True)
                            matched = sub_name
                        if active_sub and active_sub != matched:
                            prev = state_map.get(active_sub)
                            if prev and prev.phase not in ("Done", "Using Cache", "Failed"):
                                prev.phase = "Done"
                                prev.bar   = ""
                        active_sub = matched

                if active_sub:
                    with state_lock:
                        sub_st = state_map.get(active_sub)
                        if sub_st:
                            _apply_pip_line(cleaned, sub_st)

        proc.wait()
        t_err.join()
        ok = (proc.returncode == 0)

        with state_lock:
            if active_sub and active_sub in state_map:
                sub = state_map[active_sub]
                if sub.phase not in ("Done", "Using Cache"):
                    sub.phase = "Done" if ok else "Failed"
                    sub.bar   = ""
            top = state_map.get(pkg)
            if top:
                top.phase = "Done" if ok else "Failed"
                top.bar   = ""

        if not ok and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            with state_lock:
                st2       = state_map[pkg]
                st2.phase = "Retrying"
                st2.note  = f"in {delay}s — attempt {attempt + 2}/{MAX_RETRIES + 1}"
            time.sleep(delay)
            work_q.put((pkg, attempt + 1))
        else:
            with results_lock:
                if ok:
                    results.append((True, pkg, ""))
                else:
                    err = "\n".join(stderr_lines[-5:]) if stderr_lines else "unknown error"
                    results.append((False, pkg, err))

        work_q.task_done()


# ── Dashboard builder ─────────────────────────────────────────────────────────

def _build_lines(state_map, state_lock, is_tty, cols):
    with state_lock:
        snapshot = list(state_map.items())

    counts  = dict(Queued=0, Resolving=0, Downloading=0, Done=0, Cached=0, Retrying=0, Failed=0)
    active  = []
    done    = []
    failed  = []

    for name, st in snapshot:
        phase = st.phase
        if phase == "Queued":
            counts["Queued"] += 1
        elif phase == "Resolving":
            counts["Resolving"] += 1
            active.append((name, st))
        elif phase == "Downloading":
            counts["Downloading"] += 1
            active.append((name, st))
        elif phase == "Using Cache":
            counts["Cached"] += 1
            done.append((name, st))
        elif phase == "Done":
            counts["Done"] += 1
            done.append((name, st))
        elif phase == "Retrying":
            counts["Retrying"] += 1
            active.append((name, st))
        elif phase == "Failed":
            counts["Failed"] += 1
            failed.append((name, st))

    total      = len(snapshot)
    total_done = counts["Done"] + counts["Cached"]
    n_active   = counts["Downloading"] + counts["Resolving"] + counts["Retrying"]

    def c(s, color): return f"{color}{s}{_Z}" if is_tty else str(s)

    parts = [
        f"[INFO] Packages: {c(total_done, _G)}/{total} done",
        f"{c(n_active, _C)} active",
        f"{c(counts['Queued'], _Y)} queued",
    ]
    if counts["Failed"]:
        parts.append(f"{c(counts['Failed'], _R)} failed")

    lines = [" | ".join(parts)]

    # Active downloads — show all of them
    for name, st in active:
        lines.append(_render_line(name, st, is_tty, cols))

    # Last 6 completions (most recent at bottom)
    for name, st in done[-6:]:
        lines.append(_render_line(name, st, is_tty, cols))

    # All failures
    for name, st in failed:
        lines.append(_render_line(name, st, is_tty, cols))

    return lines


# ── Parallel download orchestrator ────────────────────────────────────────────

def run_parallel_download(packages, temp_dir, extra_args, is_tty):
    cpu      = os.cpu_count() or 4
    n_workers = min(max(4, cpu // 2), 8, len(packages))

    work_q       = queue.Queue()
    done_evt     = threading.Event()
    state_map    = {}
    state_lock   = threading.Lock()
    results      = []
    results_lock = threading.Lock()

    with state_lock:
        for pkg in packages:
            state_map[pkg] = PkgState()

    for pkg in packages:
        work_q.put((pkg, 0))

    workers = [
        threading.Thread(
            target=_worker,
            args=(work_q, done_evt, temp_dir, extra_args,
                  state_map, state_lock, results, results_lock),
            daemon=True,
        )
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    dash      = Dashboard(is_tty)
    stop_dash = threading.Event()

    def _dashboard_loop():
        while not stop_dash.is_set():
            try:
                cols = shutil.get_terminal_size().columns
            except Exception:
                cols = 100
            dash.draw(_build_lines(state_map, state_lock, is_tty, cols))
            time.sleep(0.12)

    dash_thread = threading.Thread(target=_dashboard_loop, daemon=True)
    dash_thread.start()

    print(f"[INFO] Parallel download started — {n_workers} workers, up to {MAX_RETRIES} retries per package.")

    work_q.join()      # blocks until every task_done() has been called (retries included)
    done_evt.set()     # signal workers to exit their wait loop
    stop_dash.set()    # stop the dashboard thread
    dash_thread.join(timeout=1.0)

    for w in workers:
        w.join()

    # Final dashboard pass
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 100
    dash.clear()

    succeeded = [r[1] for r in results if r[0]]
    failed    = [(r[1], r[2]) for r in results if not r[0]]

    print(f"[OK] Downloads complete: {len(succeeded)} succeeded, {len(failed)} failed.")
    for pkg, err in failed:
        label = f"{_R}[FAILED]{_Z}" if is_tty else "[FAILED]"
        print(f"  {label} {pkg}")
        for line in err.splitlines()[-3:]:
            print(f"    {line}")

    return failed


# ── Install phase with live output ────────────────────────────────────────────

def run_install(install_cmd, is_tty):
    proc = subprocess.Popen(
        install_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    current_pkg = None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="ignore").rstrip()
        if not line:
            continue
        clean = " ".join(line.split())

        m = _RE_COLLECT.search(clean)
        if m:
            current_pkg = re.split(r"[><= \(\[#]", m.group(1))[0]
            print(f"  {_C}Installing{_Z} {current_pkg}..." if is_tty else f"  Installing {current_pkg}...")
            continue

        if "already satisfied" in clean.lower() and current_pkg:
            print(f"  {_D}{current_pkg}: already satisfied{_Z}" if is_tty else f"  {current_pkg}: already satisfied")
            current_pkg = None
            continue

        if "successfully installed" in clean.lower():
            pkgs = clean.replace("Successfully installed", "").strip()
            label = f"{_G}Successfully installed{_Z}" if is_tty else "Successfully installed"
            print(f"  {label}: {pkgs}")
            continue

        if "error" in clean.lower():
            print(f"  {_R}{clean}{_Z}" if is_tty else f"  {clean}")
            continue

    proc.wait()
    return proc.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    requirements_file = "requirements.txt"
    is_conda          = False
    args              = sys.argv[1:]

    if "--conda" in args:
        is_conda = True
        args.remove("--conda")

    i = 0
    while i < len(args):
        if args[i].startswith("--requirements="):
            requirements_file = args[i].split("=", 1)[1]
            args.pop(i)
        elif args[i] == "--requirements" and i + 1 < len(args):
            requirements_file = args[i + 1]
            args = args[:i] + args[i + 2:]
        else:
            i += 1

    extra_args = args
    is_tty     = detect_tty()

    temp_dir = "temp_packages_download"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # ── Optional: pre-build ultralytics-yolov5 (needed only for megadetector <5.0) ──
    print("[INFO] Attempting to pre-build patched ultralytics-yolov5 (optional)...")
    yolo_url     = "https://files.pythonhosted.org/packages/96/30/75569405437893eaa6ca177f2b8493d6d54318a62b45cf693463e51ef572/ultralytics-yolov5-0.1.1.tar.gz"
    yolo_archive = os.path.join(temp_dir, "ultralytics-yolov5-0.1.1.tar.gz")
    try:
        import urllib.request
        urllib.request.urlretrieve(yolo_url, yolo_archive)

        yolo_extract = os.path.join(temp_dir, "extracted_yolo")
        with tarfile.open(yolo_archive, "r:gz") as tar:
            tar.extractall(path=yolo_extract)

        setup_files = glob.glob(os.path.join(yolo_extract, "*", "setup.py"))
        if setup_files:
            setup_path = setup_files[0]
            pkg_dir    = os.path.dirname(setup_path)
            with open(setup_path, "r", encoding="utf-8") as f:
                content = f.read()

            target = "README = request.urlopen('https://raw.githubusercontent.com/ultralytics/yolov5/master/README.md').read().decode('utf-8')"
            if target in content:
                content = content.replace(target, "README = 'ultralytics-yolov5 description'")
            else:
                content = re.sub(
                    r"README\s*=\s*request\.urlopen\([^)]+\)\.read\(\)\.decode\([^)]+\)",
                    "README = 'ultralytics-yolov5 description'",
                    content,
                )
            with open(setup_path, "w", encoding="utf-8") as f:
                f.write(content)

            print("[INFO] Pre-building patched ultralytics-yolov5 wheel...")
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps",
                 "--wheel-dir", temp_dir, pkg_dir] + extra_args,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        for path in (yolo_archive, yolo_extract):
            if os.path.exists(path):
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        print("[OK] ultralytics-yolov5 pre-build complete.")

    except Exception as exc:
        print(f"[WARNING] Could not pre-build ultralytics-yolov5 ({exc}). Continuing.")

    # ── 1. Determine packages to download ────────────────────────────────────
    if is_conda:
        packages = ["megadetector"]
    else:
        if not os.path.exists(requirements_file):
            print(f"[ERROR] Requirements file not found: {requirements_file}")
            sys.exit(1)
        packages = []
        with open(requirements_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "ultralytics-yolov5" not in line:
                    packages.append(line)

    extra_args += ["--find-links", temp_dir]

    # ── 2. Parallel download ──────────────────────────────────────────────────
    failed = run_parallel_download(packages, temp_dir, extra_args, is_tty)
    if failed:
        sys.exit(1)

    # ── 3. Consolidate into flat temp_dir ────────────────────────────────────
    print("[INFO] Consolidating downloaded files...")
    for root, dirs, files in os.walk(temp_dir):
        if root == temp_dir:
            continue
        for fname in files:
            src  = os.path.join(root, fname)
            dest = os.path.join(temp_dir, fname)
            if not os.path.exists(dest):
                shutil.move(src, dest)
            else:
                os.remove(src)
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        if root != temp_dir:
            try: os.rmdir(root)
            except OSError: pass

    wheels = glob.glob(os.path.join(temp_dir, "ultralytics_yolov5-*.whl"))
    if wheels:
        print(f"[INFO] ultralytics-yolov5 wheel: {os.path.basename(wheels[0])}")
    else:
        print("[INFO] No ultralytics-yolov5 wheel — not required for megadetector >=5.0.")

    # ── 4. Install ────────────────────────────────────────────────────────────
    print("[INFO] Installing packages...")
    if is_conda:
        install_cmd = [sys.executable, "-m", "pip", "install",
                       "--find-links", temp_dir, "megadetector"] + extra_args
    else:
        install_cmd = [sys.executable, "-m", "pip", "install",
                       "--find-links", temp_dir, "-r", requirements_file] + extra_args

    ret = run_install(install_cmd, is_tty)
    shutil.rmtree(temp_dir)

    if ret != 0:
        print("[ERROR] Package installation failed.")
        sys.exit(1)

    print("[OK] All packages installed successfully.")


if __name__ == "__main__":
    main()
