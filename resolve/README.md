# CorridorKey for DaVinci Resolve (Fusion node)

A Fusion node that runs the CorridorKey 2.0 keyer from inside DaVinci Resolve's
Fusion page (and Fusion Studio). It has an optional **Alpha Hint** input, keys
the whole shot in the background on your NVIDIA GPU, and shows the cached
result per frame with live despill and output options.

## Requirements

- Windows with an NVIDIA GPU (same requirements as the main README's Setup).
- CorridorKey set up as described in the main README: the `venv` with
  CUDA-enabled PyTorch, `pip install -r requirements.txt`, and the weights
  downloaded with `python scripts/download_weights.py`.
- DaVinci Resolve 20 (free or Studio). Resolve 18–19 and Fusion Studio
  should also work but have not been tested.

## Install

1. Finish the main README's Setup.
2. Double-click `resolve\Install_Resolve_Plugin.bat`, or run:

   ```
   venv\Scripts\python resolve\install_resolve_plugin.py
   ```

   This copies `CorridorKey.fuse` into
   `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses`, with
   the path to this folder and to the venv's Python filled in, then checks
   that PyTorch can see your GPU and that the weights are present.
3. Restart Resolve.

Keep the CorridorKey folder where it is. The node runs
`resolve\corridorkey_resolve_bridge.py` from it. If you move the folder, run
the installer again (or set the paths under **Help & Support** on the node).

## Use

1. On the Fusion page, add **Add Tool > Fuses > Keying > CorridorKey** (or
   press Shift+Space and type `CorridorKey`).
2. Connect your green screen plate to the yellow **Input**.
3. Optional: connect a matte, roto or garbage matte to the green
   **Alpha Hint** input (white = subject).
4. View the node and press **Process Shot**.

Process Shot exports every frame of the comp's render range to the cache
folder, then keys the shot in a background process while Resolve stays
usable. **Status** shows progress. When the run finishes the node re-renders
by itself and shows the key. If it does not (see Troubleshooting), press
**Reload Results** or move the playhead.

After that, every frame is read from the cache ("keyed (cached, no
re-render)"), so scrubbing and rendering are fast. Output, matte levels and
despill are applied live and never need re-processing.

**Process Shot again** when you change the plate, the hint, the frame range,
or any of the settings that affect the key: Input is Linear, Internal
Resolution, GPU Memory Limit, Temporal Awareness and the Alpha Hint settings.
**Processed Shot** tells you when the settings no longer match the cached
run. It cannot detect changes to the upstream image itself.

### Controls

| Control | What it does |
| --- | --- |
| Status | Progress of the current run, or whether this frame is keyed. |
| Estimated VRAM | Estimated peak GPU memory / free / total, from the last run's GPU reading. "(busy)" while a run is in progress. |
| Input is Linear? | Tick for scene-linear plates (e.g. linear EXR or ACEScg working space). Leave off for display-referred (Rec.709/sRGB) footage. Output stays in the same colour space as the input. |
| Linear Guess | A hint from the current frame's pixel values. It only helps you decide; it does not change anything. |
| Internal Resolution | Full / Half / Quarter. Lower is faster and uses less memory; output is always full size. |
| GPU Memory Limit | **Auto** picks the largest temporal window that fits free GPU memory (up to Temporal Awareness). **Low VRAM** uses the staged low-memory path. **High VRAM** forces the fast path. |
| Cache Folder | Where exported frames and results go. Default `%LOCALAPPDATA%\CorridorKey\ResolveCache`. Each node gets its own subfolder. |
| Temporal Awareness | Frames the model looks at together (the Hann window size, 1–80). Larger is steadier over time but uses more VRAM. |
| Process Shot / Cancel Pass | Start or stop a run. |
| Hint Mode | Off, First Frame Only, or Every Frame. Ignored if nothing is connected to Alpha Hint. |
| Hint Channel / Invert Hint | Use the hint's alpha or its luminance, optionally inverted. |
| Hint Quality | How much the model trusts the hint: 0.95 for a clean matte, 0.75 for a rough one, 0.5 or lower for a loose garbage matte. |
| Carry Hint | Feeds the model's own matte forward between temporal windows (recommended, especially with First Frame Only). |
| Output | Premultiplied Key, Straight Foreground, Matte, Original + Matte (your plate with the predicted alpha) or Raw Foreground. |
| Matte Black / White Point | Crush or lift the matte. |
| Despill | Colour (Auto detects green or blue from the shot), algorithm and amount. **Freeze Auto Colour** sets the colour to what Auto detected. CorridorKey's foreground is already largely spill-free, so the default amount is 0. |
| Help & Support | Log for the last run, the CorridorKey folder and Python the node uses, and a test mode. |

**Pipeline Test Mode** runs a simple colour-difference key instead of the
neural network. It needs no GPU and takes seconds, so it is a quick way to
check that the node, the cache folder and Python are wired up.

## Rendering on the Deliver page

Resolve renders the node from the cache like any other frame, so process the
shot before you render. Frames outside the processed range, or from a shot
that was never processed, pass the input through unkeyed.

## Disk space

During a run the exported plate takes 8 bytes per pixel per frame (about
16 MB per 1080p frame, 66 MB per UHD frame). It is deleted when the run
finishes. The results that stay in the cache take the same space. Delete a
node's folder from the cache when you no longer need it. **Open Cache
Folder** takes you there.

## Troubleshooting

- **Nothing happens after Process Shot.** The export runs when the node
  renders, so view the node (or something downstream of it).
- **"Python did not start"**: the Python path under Help & Support is wrong,
  or the venv is missing. Run the installer again. `launch.txt` in the
  node's cache folder shows the exact command that was run.
- **error: ...** in Status: press **Open Log** for the full Python output.
  Out-of-memory errors: set GPU Memory Limit to Low VRAM, lower Temporal
  Awareness, or use Half Internal Resolution.
- **The node doesn't refresh when a run finishes.** The bridge refreshes it
  through Resolve's scripting API. Enable *Preferences > System > General >
  External scripting using: Local*, or press **Reload Results**.
- **The node isn't in the Add Tool menu.** Check that `CorridorKey.fuse` is
  in the Fuses folder above, and restart Resolve. Errors from loading the
  Fuse appear in *Workspace > Console*.

## How it works

`CorridorKey.fuse` is a Lua Fuse. Fusion images are copied to and from the
cache as raw half-float RGBA (`.ckraw`, Fusion's own memory layout, bottom-up
rows) through LuaJIT FFI, so the copy costs about 0.15 s per 1080p frame.
`corridorkey_resolve_bridge.py` reads those frames through the normal
CorridorKey runtime (`corridorkey.runtime.run_inference`), writes straight
foreground + alpha per frame, and reports progress in `status.txt`.
Despill, matte levels and the output mode run in a small DCTL kernel on the
GPU inside the Fuse.
