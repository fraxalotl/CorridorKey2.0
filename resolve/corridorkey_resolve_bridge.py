#!/usr/bin/env python3
"""Background worker for the CorridorKey Fusion node (CorridorKey.fuse).

The Fuse exports a shot's frames into a cache folder as raw half-float RGBA
files, writes a job.json describing the run, and launches this script. The
script runs the CorridorKey runtime on those frames and writes one result
file per frame (foreground RGB + alpha) that the Fuse loads back.

Shot cache layout (all under <cache>/<shot>/):

  job.json             settings written by the Fuse
  input/000000.ckraw   exported plate, half-float RGBA, bottom-up rows
  hint/000000.ckraw    exported alpha hint (same format), optional
  result/000000.ckraw  straight foreground RGB + alpha (same format)
  result/meta.txt      key=value summary of the finished run
  status.txt           key=value progress, read by the Fuse
  log.txt              stdout/stderr of the last run
  cancel               created by the Fuse to stop a run

Results are written to result_tmp/ and swapped in when the run completes, so
the previous results stay visible while a new run is in progress.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

RAW_EXT = ".ckraw"
HALF_MAX = 65504.0


class Cancelled(Exception):
    pass


class Shot:
    def __init__(self, job_path: Path):
        self.job_path = job_path
        self.dir = job_path.parent
        self.job = json.loads(job_path.read_text(encoding="utf-8"))
        self.width = int(self.job["width"])
        self.height = int(self.job["height"])
        self.num_frames = int(self.job["num_frames"])
        self.input_dir = self.dir / "input"
        self.hint_dir = self.dir / "hint"
        self.result_dir = self.dir / "result"
        self.tmp_dir = self.dir / "result_tmp"
        self.work_dir = self.dir / "work"
        self.status_path = self.dir / "status.txt"
        self.cancel_path = self.dir / "cancel"
        self.started = time.time()

    # -- status ---------------------------------------------------------
    def status(self, state: str, progress: float, message: str) -> None:
        lines = {
            "state": state,
            "progress": f"{max(0.0, min(1.0, progress)):.4f}",
            "message": message.replace("\n", " ").strip(),
            "pid": str(os.getpid()),
            "job": str(self.job.get("request", "")),
            "updated": f"{time.time():.0f}",
            "elapsed": f"{time.time() - self.started:.0f}",
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
        os.replace(tmp, self.status_path)

    def check_cancel(self) -> None:
        if self.cancel_path.exists():
            raise Cancelled()

    # -- raw frame io ---------------------------------------------------
    def read_raw(self, path: Path, width: int | None = None,
                 height: int | None = None) -> np.ndarray:
        """Read a .ckraw file as top-down float32 RGBA (H, W, 4)."""
        w = width or self.width
        h = height or self.height
        data = np.fromfile(path, dtype=np.float16)
        if data.size != w * h * 4:
            raise ValueError(f"{path}: expected {w}x{h} RGBA half, got {data.size} values")
        return data.reshape(h, w, 4)[::-1].astype(np.float32)

    def write_raw(self, path: Path, rgba: np.ndarray) -> None:
        """Write top-down float RGBA (H, W, 4) as a bottom-up half .ckraw file."""
        arr = np.clip(rgba[::-1], -HALF_MAX, HALF_MAX).astype(np.float16)
        tmp = path.with_suffix(".part")
        np.ascontiguousarray(arr).tofile(tmp)
        os.replace(tmp, path)

    def frame_name(self, index: int) -> str:
        return f"{index:06d}{RAW_EXT}"


# -- colour helpers (numpy, matching corridorkey.color_utils) -----------

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def detect_screen_colour(plate_rgb: np.ndarray, alpha: np.ndarray) -> str:
    """Return "green" or "blue" from the mean background colour of a frame."""
    bg = alpha < 0.05
    if bg.sum() < 64:
        bg = np.ones_like(alpha, dtype=bool)
    mean = plate_rgb[bg].mean(axis=0)
    return "blue" if mean[2] > mean[1] else "green"


# -- hint export ----------------------------------------------------------

def hint_matte(shot: Shot, rgba: np.ndarray) -> np.ndarray:
    if shot.job.get("hint_channel", "alpha") == "luma":
        rgb = rgba[..., :3]
        if shot.job.get("input_linear", False):
            rgb = linear_to_srgb(rgb)
        matte = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    else:
        matte = rgba[..., 3]
    matte = np.clip(matte, 0.0, 1.0)
    if shot.job.get("hint_invert", False):
        matte = 1.0 - matte
    return matte


def export_hints(shot: Shot) -> tuple[str | None, str | None]:
    """Convert exported hint frames to 16-bit PNGs the runtime can read.

    Returns (first_frame_hint_path, hint_dir_path); at most one is set.
    """
    import cv2

    mode = shot.job.get("hint_mode", "off")
    if mode == "off":
        return None, None
    out_dir = shot.work_dir / "hint_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 1 if mode == "first" else shot.num_frames
    hw = shot.job.get("hint_width") or shot.width
    hh = shot.job.get("hint_height") or shot.height
    for i in range(count):
        shot.check_cancel()
        src = shot.hint_dir / shot.frame_name(i)
        if not src.exists():
            raise FileNotFoundError(f"missing exported hint frame {src}")
        matte = hint_matte(shot, shot.read_raw(src, int(hw), int(hh)))
        if matte.shape != (shot.height, shot.width):
            matte = cv2.resize(matte, (shot.width, shot.height),
                               interpolation=cv2.INTER_LINEAR)
        png = (np.clip(matte, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        cv2.imwrite(str(out_dir / f"hint_{i:06d}.png"), png)
        shot.status("running", 0.05 * (i + 1) / count, "Preparing alpha hint")
    if mode == "first":
        return str(out_dir / "hint_000000.png"), None
    return None, str(out_dir)


# -- result sink ----------------------------------------------------------

class ResultSink:
    """Collects the runtime's alpha and fg outputs and writes .ckraw results.

    Replaces corridorkey.runtime.ExrSequenceWriter for the duration of a run:
    the runtime hands each output stream its own writer, and frames are paired
    by index as both streams arrive.
    """

    def __init__(self, shot: Shot, progress_from: float, progress_to: float):
        self.shot = shot
        self.pending: dict[str, dict[int, np.ndarray]] = {"alpha": {}, "fg": {}}
        self.counts = {"alpha": 0, "fg": 0}
        self.written = 0
        self.lock = threading.Lock()
        self.p0, self.p1 = progress_from, progress_to
        self.screen_colour: str | None = None

    def writer(self, dir_path, fps, size, *, channels=3, dtype="half",
               maxsize=2, frame_name="frame"):
        return _SinkWriter(self, frame_name)

    def push(self, name: str, frames: np.ndarray) -> None:
        self.shot.check_cancel()
        with self.lock:
            for frame in frames:
                self.pending[name][self.counts[name]] = frame
                self.counts[name] += 1
            ready = sorted(set(self.pending["alpha"]) & set(self.pending["fg"]))
            pairs = [(i, self.pending["alpha"].pop(i), self.pending["fg"].pop(i))
                     for i in ready]
        for index, alpha, fg_bgr in pairs:
            self._write(index, alpha, fg_bgr)

    def _write(self, index: int, alpha: np.ndarray, fg_bgr: np.ndarray) -> None:
        shot = self.shot
        rgb = fg_bgr[..., ::-1]
        if not shot.job.get("input_linear", False):
            rgb = linear_to_srgb(rgb)
        a = np.clip(alpha[..., 0], 0.0, 1.0)
        if self.screen_colour is None:
            plate = shot.read_raw(shot.input_dir / shot.frame_name(index))[..., :3]
            self.screen_colour = detect_screen_colour(plate, a)
        rgba = np.concatenate([rgb, a[..., None]], axis=-1)
        shot.write_raw(shot.tmp_dir / shot.frame_name(index), rgba)
        self.written += 1
        frac = self.written / max(1, shot.num_frames)
        shot.status("running", self.p0 + (self.p1 - self.p0) * frac,
                    f"Keying frame {self.written}/{shot.num_frames}")


class _SinkWriter:
    def __init__(self, sink: ResultSink, name: str):
        self.sink = sink
        self.name = name

    def write_many(self, frames: np.ndarray) -> None:
        self.sink.push(self.name, np.asarray(frames, dtype=np.float32))

    def write(self, frame: np.ndarray) -> None:
        self.write_many(frame[None])

    def release(self) -> None:
        pass


# -- runs -------------------------------------------------------------------

def choose_memory_profile(shot: Shot, model_hw: tuple[int, int]) -> tuple[int, int, bool, str]:
    """Pick (hann_chunk, hann_stride, low_vram, note) for this shot.

    Temporal Awareness caps the window size. In Auto mode the largest window
    up to that cap that fits free GPU memory is used, preferring a larger
    low-VRAM window over a smaller normal one (low-VRAM mode is exact).
    """
    import torch
    from corridorkey.runtime import estimate_peak_vram_gb

    cap = max(1, min(80, int(shot.job.get("temporal", 24))))
    mode = shot.job.get("memory_mode", "auto")
    T = shot.num_frames
    megapixels = model_hw[0] * model_hw[1] / 1e6
    if mode == "low":
        return cap, max(1, cap // 2), True, "low VRAM"
    if mode == "high":
        return cap, max(1, cap // 2), False, "high VRAM"
    free, _total = torch.cuda.mem_get_info()
    budget = (free + torch.cuda.memory_reserved()) / 2**30
    chunk = cap
    while True:
        for low_vram in (False, True):
            est = estimate_peak_vram_gb(megapixels, min(chunk, T), low_vram)
            if est * 1.1 + 0.3 <= budget:
                note = f"auto: {budget:.1f} GB free, est {est:.1f} GB"
                return chunk, max(1, chunk // 2), low_vram, note
        if chunk == 1:
            return 1, 1, True, f"auto: {budget:.1f} GB free, may not fit"
        chunk = max(1, chunk // 2)


def write_gpu_info(cache_root: Path) -> None:
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        text = (f"name={name}\nfree_gb={free / 2**30:.2f}\n"
                f"total_gb={total / 2**30:.2f}\nupdated={time.time():.0f}\n")
        (cache_root / "gpu.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass


def patch_runtime_io(shot: Shot, sink: ResultSink) -> None:
    """Teach the runtime to read .ckraw frames and write into the sink."""
    from corridorkey import runtime, video_io

    original_frame_paths = video_io.frame_paths
    original_read_rgb = video_io.read_image_rgb

    def frame_paths(frame_dir):
        if Path(frame_dir).resolve() == shot.input_dir.resolve():
            return [str(shot.input_dir / shot.frame_name(i)) for i in range(shot.num_frames)]
        return original_frame_paths(frame_dir)

    def read_image_rgb(path):
        path = str(path)
        if path.endswith(RAW_EXT):
            return np.ascontiguousarray(shot.read_raw(Path(path))[..., :3])
        return original_read_rgb(path)

    video_io.frame_paths = frame_paths
    video_io.read_image_rgb = read_image_rgb
    runtime.frame_paths = frame_paths
    runtime.ExrSequenceWriter = sink.writer


def run_corridorkey(shot: Shot, initial_hint: str | None, hint_dir: str | None) -> ResultSink:
    root = Path(shot.job["ck_root"]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)

    shot.status("running", 0.06, "Loading PyTorch")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CorridorKey needs a CUDA GPU, but PyTorch cannot see one. "
                           "Check the NVIDIA driver and that CUDA-enabled PyTorch is installed.")
    from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model, run_inference
    from corridorkey.video_io import WriterConfig, ceil_to_multiple, scaled_hw

    write_gpu_info(shot.dir.parent)
    model_scale = float(shot.job.get("model_scale", 1.0))
    h_m, w_m = scaled_hw(shot.height, shot.width, model_scale)
    model_hw = (ceil_to_multiple(h_m, 128), ceil_to_multiple(w_m, 128))

    shot.status("running", 0.08, "Loading CorridorKey model")
    model = load_model(RuntimePaths.from_package(root), device="cuda")
    shot.check_cancel()

    chunk, stride, low_vram, note = choose_memory_profile(shot, model_hw)
    print(f"  memory profile: hann_chunk={chunk} stride={stride} low_vram={low_vram} ({note})")
    sink = ResultSink(shot, 0.15, 0.97)
    patch_runtime_io(shot, sink)
    cfg = InferenceConfig(
        hann_chunk=chunk,
        hann_stride=stride,
        auto_vram=False,
        low_vram=low_vram,
        carry_hint=bool(shot.job.get("carry_hint", True)),
        hint_quality=float(shot.job.get("hint_quality", 0.95)),
        carry_hint_quality=float(shot.job.get("carry_hint_quality", 0.95)),
        model_scale=model_scale,
        outputs={"alpha", "fg"},
        temp_dir=shot.work_dir / "runtime",
        frame_dir_linear=bool(shot.job.get("input_linear", False)),
        despill_strength=0.0,
        output_format="exr",
        exr_dtype="float",
    )
    shot.status("running", 0.12, f"Keying ({note})")
    run_inference(
        model, str(shot.input_dir), str(shot.work_dir / "runtime_out"), cfg,
        WriterConfig(), initial_hint_path=initial_hint, hint_video_path=hint_dir)
    write_gpu_info(shot.dir.parent)
    return sink


def run_test_mode(shot: Shot) -> ResultSink:
    """Simple colour-difference key used to test the plugin without a GPU.

    It exercises the same file exchange as a real run, so a working test-mode
    key means the Fuse, the cache folder and the Python environment are set
    up correctly.
    """
    sink = ResultSink(shot, 0.1, 0.97)
    hint_mode = shot.job.get("hint_mode", "off")
    for i in range(shot.num_frames):
        shot.check_cancel()
        rgba = shot.read_raw(shot.input_dir / shot.frame_name(i))
        rgb = rgba[..., :3]
        lin = rgb if shot.job.get("input_linear", False) else srgb_to_linear(rgb)
        r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
        screen = np.maximum(g, b)
        diff = screen - np.maximum(r, np.minimum(g, b))
        alpha = np.clip(1.0 - diff * 4.0, 0.0, 1.0)
        if hint_mode == "every" or (hint_mode == "first" and i == 0):
            hw = int(shot.job.get("hint_width") or shot.width)
            hh = int(shot.job.get("hint_height") or shot.height)
            if (hw, hh) == (shot.width, shot.height):
                hint = hint_matte(shot, shot.read_raw(shot.hint_dir / shot.frame_name(i), hw, hh))
                alpha = alpha * hint
        fg_bgr = lin[..., ::-1]
        sink.push("alpha", alpha[None, ..., None])
        sink.push("fg", fg_bgr[None])
        time.sleep(0.02)
    return sink


def write_meta(shot: Shot, sink: ResultSink) -> None:
    meta = {
        "signature": shot.job.get("signature", ""),
        "request": shot.job.get("request", ""),
        "first_frame": shot.job.get("first_frame", 0),
        "num_frames": shot.num_frames,
        "width": shot.width,
        "height": shot.height,
        "screen_colour": sink.screen_colour or "green",
        "test_mode": int(bool(shot.job.get("test_mode", False))),
        "elapsed": f"{time.time() - shot.started:.0f}",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (shot.tmp_dir / "meta.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in meta.items()), encoding="utf-8")


def swap_results(shot: Shot) -> None:
    old = shot.dir / "result_old"
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if shot.result_dir.exists():
        os.replace(shot.result_dir, old)
    os.replace(shot.tmp_dir, shot.result_dir)
    shutil.rmtree(old, ignore_errors=True)


def cleanup(shot: Shot, success: bool) -> None:
    shutil.rmtree(shot.work_dir, ignore_errors=True)
    if not success:
        shutil.rmtree(shot.tmp_dir, ignore_errors=True)
    if not shot.job.get("keep_input", False):
        shutil.rmtree(shot.input_dir, ignore_errors=True)
        shutil.rmtree(shot.hint_dir, ignore_errors=True)
    try:
        shot.cancel_path.unlink()
    except FileNotFoundError:
        pass


def watch_cancel(shot: Shot, done: threading.Event) -> None:
    """Hard-stop the process if the Fuse asks to cancel mid-inference."""
    while not done.wait(0.5):
        if shot.cancel_path.exists():
            try:
                shot.status("cancelled", 0.0, "Cancelled")
                cleanup(shot, success=False)
            finally:
                os._exit(3)


def poke_fusion(shot: Shot) -> None:
    """Ask Resolve to re-render the node so new results show up right away.

    Uses Resolve's scripting API, which needs Preferences > System > General >
    External scripting using: Local. If that is off, the Fuse picks up the
    results the next time the frame is rendered or Reload Results is pressed.
    """
    tool_name = shot.job.get("tool_name")
    if not tool_name:
        return
    candidates = []
    env_lib = os.environ.get("RESOLVE_SCRIPT_LIB")
    if env_lib:
        candidates.append(env_lib)
    if sys.platform.startswith("win"):
        candidates.append(r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll")
    elif sys.platform == "darwin":
        candidates.append("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so")
        candidates.append("/Applications/DaVinci Resolve Studio.app/Contents/Libraries/Fusion/fusionscript.so")
    else:
        candidates.append("/opt/resolve/libs/Fusion/fusionscript.so")
    lib = next((c for c in candidates if os.path.exists(c)), None)
    if lib is None:
        return
    try:
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.ExtensionFileLoader("fusionscript", lib)
        spec = importlib.util.spec_from_loader("fusionscript", loader)
        bmd = importlib.util.module_from_spec(spec)
        loader.exec_module(bmd)
        resolve = bmd.scriptapp("Resolve")
        fusion = resolve.Fusion() if resolve else bmd.scriptapp("Fusion")
        comp = fusion.GetCurrentComp() if fusion else None
        tool = comp.FindTool(tool_name) if comp else None
        if tool is None:
            return
        if tool.GetInput("ShotID") != shot.job.get("shot_id"):
            return
        tool.SetInput("ResultStamp", float(time.time() % 1e6))
    except Exception as exc:
        print(f"  (could not refresh Resolve automatically: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, help="Path to the shot's job.json")
    args = ap.parse_args()
    shot = Shot(Path(args.job))

    log = open(shot.dir / "log.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
    print(f"CorridorKey Resolve bridge, job {shot.job.get('request')} started "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(json.dumps(shot.job, indent=2))

    done = threading.Event()
    threading.Thread(target=watch_cancel, args=(shot, done), daemon=True).start()
    success = False
    try:
        shot.status("running", 0.0, "Starting")
        shutil.rmtree(shot.tmp_dir, ignore_errors=True)
        shot.tmp_dir.mkdir(parents=True)
        shot.work_dir.mkdir(parents=True, exist_ok=True)
        if shot.job.get("test_mode", False):
            sink = run_test_mode(shot)
        else:
            initial_hint, hint_dir = export_hints(shot)
            sink = run_corridorkey(shot, initial_hint, hint_dir)
        if sink.written != shot.num_frames:
            raise RuntimeError(f"only {sink.written} of {shot.num_frames} frames were keyed")
        write_meta(shot, sink)
        swap_results(shot)
        success = True
        shot.status("done", 1.0, f"Keyed {shot.num_frames} frames in "
                    f"{time.time() - shot.started:.0f}s")
        print("done")
    except Cancelled:
        shot.status("cancelled", 0.0, "Cancelled")
        print("cancelled")
    except Exception as exc:
        traceback.print_exc()
        shot.status("error", 0.0, f"{type(exc).__name__}: {exc}")
    finally:
        done.set()
        cleanup(shot, success)
    if success:
        poke_fusion(shot)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
