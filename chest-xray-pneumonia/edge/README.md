# TrustEdge-CXR edge node

Runs the INT8 pneumonia screener on a Raspberry Pi 4/5 (64-bit OS), a Jetson,
or any small Linux box. Everything happens on the device; only the result
(and, optionally, the heatmap) is sent to the hospital server.

## Install

```bash
python3 -m venv ~/trustedge && source ~/trustedge/bin/activate
pip install -r requirements-edge.txt
```

## Run

```bash
# one-off
python edge_infer.py chest_xray.jpeg --save-overlay overlays/

# IoT mode: watch the scanner's export folder and report to the hospital server
python edge_infer.py --watch /srv/xray_out --server http://<server-ip>:8000 --send-image
```

Each image prints a JSON line with `decision` (`NORMAL`, `PNEUMONIA` or
`REFER`), the calibrated `prob_pneumonia`, the MC-dropout `uncertainty` and
the on-device `latency_ms`. `REFER` means the model is unsure and a
radiologist should read the film.

If the server sets `TRUSTEDGE_EDGE_TOKEN`, pass the same value with
`--token` (or the same environment variable on the device).

Educational project — not a medical device.
