#!/usr/bin/env python3
"""Small local UI for correcting YOLO draft labels."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image


CLASSES = ["passion_fruit"]


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chinola Label Review</title>
<style>
:root {
  --bg: #101417;
  --panel: #171d20;
  --panel2: #20282c;
  --text: #eef3ef;
  --muted: #9fad9f;
  --line: #334047;
  --accent: #c7f06b;
  --warn: #ffbd5a;
  --bad: #ff6b6b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
main {
  display: grid;
  grid-template-columns: minmax(220px, 300px) minmax(0, 1fr) minmax(260px, 340px);
  min-height: 100vh;
}
aside, section {
  border-right: 1px solid var(--line);
  background: var(--panel);
}
h1, h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}
button, select {
  border: 1px solid var(--line);
  background: var(--panel2);
  color: var(--text);
  border-radius: 6px;
  min-height: 34px;
  padding: 7px 10px;
  font: inherit;
}
button:hover { border-color: var(--accent); }
button.active { background: var(--accent); color: #182000; border-color: var(--accent); }
button.danger { border-color: #724047; color: #ffd9dc; }
.top {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--line);
}
.list {
  height: calc(100vh - 59px);
  overflow: auto;
}
.item {
  padding: 9px 12px;
  border-bottom: 1px solid #253137;
  cursor: pointer;
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
.item strong {
  color: var(--text);
  display: block;
  font-size: 12px;
}
.item.active {
  background: #26331d;
  color: var(--text);
}
.item .tag {
  display: inline-block;
  margin-top: 5px;
  color: #12170e;
  background: var(--accent);
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 11px;
}
.item.unsaved strong::after {
  content: " *";
  color: var(--warn);
}
.stage {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  background: #0b0f11;
  min-width: 0;
}
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 12px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.tools {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.canvasWrap {
  min-height: 0;
  overflow: auto;
  display: grid;
  place-items: center;
  padding: 18px;
}
canvas {
  max-width: 100%;
  max-height: calc(100vh - 132px);
  background: #050607;
  border: 1px solid var(--line);
}
.status {
  padding: 10px 12px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
}
.right {
  padding: 12px;
  border-right: 0;
}
.boxList {
  display: grid;
  gap: 7px;
  margin-top: 12px;
}
.boxRow {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel2);
  font-size: 12px;
}
.boxRow.active { border-color: var(--accent); }
.help {
  margin-top: 16px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
@media (max-width: 960px) {
  main { grid-template-columns: 1fr; }
  .list { height: 180px; }
  aside, section { border-right: 0; border-bottom: 1px solid var(--line); }
}
</style>
</head>
<body>
<main>
  <aside>
    <div class="top"><h1>Chinola Review</h1><span id="count"></span></div>
    <div id="imageList" class="list"></div>
  </aside>
  <section class="stage">
    <div class="toolbar">
      <h2 id="title">Loading</h2>
      <div class="tools">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <button id="saveBtn" class="active">Save</button>
      </div>
    </div>
    <div class="canvasWrap">
      <canvas id="canvas"></canvas>
    </div>
    <div id="status" class="status">Draw boxes around visible fruit. Save reviewed labels before moving on.</div>
  </section>
  <aside class="right">
    <h2>Box Tools</h2>
    <div class="tools" style="margin-top:12px">
      <button id="passionBtn" class="active">passion fruit</button>
      <button id="deleteBtn" class="danger">Delete selected</button>
      <button id="clearBtn" class="danger">Clear</button>
    </div>
    <div id="boxList" class="boxList"></div>
    <div class="help">
      Drag on image to add a box. Click a box to select it. Use class buttons to change selected box class. Save writes reviewed YOLO labels into the reviewed output folder.
    </div>
  </aside>
</main>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageList = document.getElementById('imageList');
const boxList = document.getElementById('boxList');
const statusEl = document.getElementById('status');
const titleEl = document.getElementById('title');
const countEl = document.getElementById('count');
let images = [];
let index = 0;
let img = new Image();
let boxes = [];
let selected = -1;
let currentClass = 0;
let drawing = null;
let scale = 1;
const classNames = ['passion_fruit'];
const colors = ['#5cc8ff'];

function setStatus(text) { statusEl.textContent = text; }
function normToPx(b) {
  return {
    class_id: b.class_id,
    x: (b.x_center - b.width / 2) * img.naturalWidth,
    y: (b.y_center - b.height / 2) * img.naturalHeight,
    w: b.width * img.naturalWidth,
    h: b.height * img.naturalHeight,
  };
}
function pxToNorm(b) {
  const x = Math.max(0, Math.min(img.naturalWidth, b.x));
  const y = Math.max(0, Math.min(img.naturalHeight, b.y));
  const w = Math.max(1, Math.min(img.naturalWidth - x, b.w));
  const h = Math.max(1, Math.min(img.naturalHeight - y, b.h));
  return {
    class_id: b.class_id,
    x_center: (x + w / 2) / img.naturalWidth,
    y_center: (y + h / 2) / img.naturalHeight,
    width: w / img.naturalWidth,
    height: h / img.naturalHeight,
  };
}
function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const canvasX = (evt.clientX - rect.left) * (canvas.width / rect.width);
  const canvasY = (evt.clientY - rect.top) * (canvas.height / rect.height);
  return {
    x: canvasX / scale,
    y: canvasY / scale,
  };
}
function draw() {
  if (!img.naturalWidth) return;
  const maxW = Math.min(window.innerWidth - 420, img.naturalWidth);
  const maxH = window.innerHeight - 132;
  scale = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  [...boxes, drawing].filter(Boolean).forEach((b, i) => {
    const color = colors[b.class_id] || '#fff';
    const isSel = i === selected;
    ctx.lineWidth = isSel ? 4 : 2;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(b.x * scale, b.y * scale, b.w * scale, b.h * scale);
    ctx.fillRect(b.x * scale, Math.max(0, b.y * scale - 20), 120, 20);
    ctx.fillStyle = '#101417';
    ctx.font = '12px sans-serif';
    ctx.fillText(classNames[b.class_id] || 'fruit', b.x * scale + 5, Math.max(14, b.y * scale - 6));
  });
  renderBoxes();
}
function renderBoxes() {
  boxList.innerHTML = '';
  boxes.forEach((b, i) => {
    const row = document.createElement('div');
    row.className = 'boxRow' + (i === selected ? ' active' : '');
    row.innerHTML = `<span>${i + 1}. ${classNames[b.class_id]} ${Math.round(b.w)}x${Math.round(b.h)}</span><button>select</button>`;
    row.onclick = () => { selected = i; draw(); };
    boxList.appendChild(row);
  });
}
function renderList() {
  imageList.innerHTML = '';
  const reviewedCount = images.filter(item => item.reviewed).length;
  countEl.textContent = `${reviewedCount}/${images.length} saved`;
  images.forEach((item, i) => {
    const el = document.createElement('div');
    el.className = 'item' + (i === index ? ' active' : '');
    el.innerHTML = `<strong>${item.image}</strong>${item.split} · ${item.label_count} boxes ${item.reviewed ? '<span class="tag">reviewed</span>' : ''}`;
    el.onclick = () => loadImage(i);
    imageList.appendChild(el);
  });
}
async function loadImage(i) {
  if (window.dirty) {
    await save();
  }
  index = Math.max(0, Math.min(images.length - 1, i));
  const item = images[index];
  titleEl.textContent = item.image;
  renderList();
  const labels = await fetch(`/api/labels?split=${encodeURIComponent(item.split)}&image=${encodeURIComponent(item.image)}`).then(r => r.json());
  boxes = labels.boxes.map(normToPx);
  selected = boxes.length ? 0 : -1;
  img = new Image();
  img.onload = draw;
  img.src = `/api/image?split=${encodeURIComponent(item.split)}&image=${encodeURIComponent(item.image)}&t=${Date.now()}`;
  setStatus(`${index + 1}/${images.length}: loaded ${boxes.length} boxes`);
  window.dirty = false;
}
async function save() {
  const item = images[index];
  const payload = { split: item.split, image: item.image, boxes: boxes.map(pxToNorm) };
  const res = await fetch('/api/labels', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  if (!res.ok) {
    setStatus('Save failed');
    return;
  }
  const data = await res.json();
  images[index].reviewed = true;
  images[index].label_count = boxes.length;
  window.dirty = false;
  renderList();
  setStatus(`Saved ${boxes.length} boxes to ${data.label_path}`);
  await refreshImages(false);
}
function hitTest(p) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return i;
  }
  return -1;
}
canvas.addEventListener('mousedown', e => {
  const p = canvasPoint(e);
  const hit = hitTest(p);
  if (hit >= 0) {
    selected = hit;
    draw();
    return;
  }
  drawing = { class_id: currentClass, x: p.x, y: p.y, w: 1, h: 1, startX: p.x, startY: p.y };
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = canvasPoint(e);
  drawing.x = Math.min(drawing.startX, p.x);
  drawing.y = Math.min(drawing.startY, p.y);
  drawing.w = Math.abs(p.x - drawing.startX);
  drawing.h = Math.abs(p.y - drawing.startY);
  draw();
});
window.addEventListener('mouseup', () => {
  if (!drawing) return;
  if (drawing.w > 8 && drawing.h > 8) {
    boxes.push({ class_id: drawing.class_id, x: drawing.x, y: drawing.y, w: drawing.w, h: drawing.h });
    selected = boxes.length - 1;
    window.dirty = true;
  }
  drawing = null;
  draw();
});
document.getElementById('saveBtn').onclick = save;
document.getElementById('prevBtn').onclick = () => loadImage(index - 1);
document.getElementById('nextBtn').onclick = () => loadImage(index + 1);
document.getElementById('deleteBtn').onclick = () => {
  if (selected >= 0) boxes.splice(selected, 1);
  selected = Math.min(selected, boxes.length - 1);
  window.dirty = true;
  draw();
};
document.getElementById('clearBtn').onclick = () => { boxes = []; selected = -1; window.dirty = true; draw(); };
function setClass(id) {
  currentClass = id;
  document.getElementById('passionBtn').classList.toggle('active', id === 0);
  if (selected >= 0) {
    boxes[selected].class_id = id;
    window.dirty = true;
    draw();
  }
}
document.getElementById('passionBtn').onclick = () => setClass(0);
window.addEventListener('resize', draw);
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') loadImage(index + 1);
  if (e.key === 'ArrowLeft') loadImage(index - 1);
  if (e.key === 's') save();
  if (e.key === '1' || e.key === '2') setClass(0);
  if (e.key === 'Backspace' || e.key === 'Delete') {
    if (selected >= 0) boxes.splice(selected, 1);
    selected = Math.min(selected, boxes.length - 1);
    window.dirty = true;
    draw();
  }
});
window.addEventListener('beforeunload', e => {
  if (window.dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});
async function refreshImages(keepIndex = true) {
  const previous = images[index]?.image;
  const data = await fetch(`/api/images?t=${Date.now()}`).then(r => r.json());
  images = data.images;
  if (keepIndex && previous) {
    const next = images.findIndex(item => item.image === previous);
    if (next >= 0) index = next;
  }
  renderList();
  return data;
}
refreshImages(false).then(() => {
  if (images.length) loadImage(0);
  else setStatus('No images found');
});
</script>
</body>
</html>
"""


class App:
    def __init__(self, draft: Path, reviewed: Path) -> None:
        self.draft = draft
        self.reviewed = reviewed
        self.reviewed.mkdir(parents=True, exist_ok=True)

    def image_path(self, split: str, image: str) -> Path:
        return self.draft / "images" / split / image

    def draft_label_path(self, split: str, image: str) -> Path:
        return self.draft / "labels" / split / f"{Path(image).stem}.txt"

    def reviewed_label_path(self, split: str, image: str) -> Path:
        return self.reviewed / "labels" / split / f"{Path(image).stem}.txt"

    def reviewed_image_path(self, split: str, image: str) -> Path:
        return self.reviewed / "images" / split / image

    def images(self) -> list[dict]:
        rows = []
        for split in ("train", "val"):
            for image in sorted((self.draft / "images" / split).glob("*")):
                if not image.is_file():
                    continue
                reviewed = self.reviewed_label_path(split, image.name).exists()
                label_path = self.reviewed_label_path(split, image.name) if reviewed else self.draft_label_path(split, image.name)
                rows.append(
                    {
                        "split": split,
                        "image": image.name,
                        "reviewed": reviewed,
                        "label_count": len(read_labels(label_path)),
                    }
                )
        return rows


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, x, y, w, h = parts
        boxes.append(
            {
                "class_id": int(class_id),
                "x_center": float(x),
                "y_center": float(y),
                "width": float(w),
                "height": float(h),
            }
        )
    return boxes


def write_labels(path: Path, boxes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        class_id = int(box["class_id"])
        x = float(box["x_center"])
        y = float(box["y_center"])
        w = float(box["width"])
        h = float(box["height"])
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    path.write_text(("\n".join(lines) + "\n") if lines else "")


def write_data_yaml(app: App) -> None:
    data = (
        f"path: {app.reviewed.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: passion_fruit\n"
    )
    (app.reviewed / "data.yaml").write_text(data)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, text: str, content_type: str = "text/html") -> None:
            body = text.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_text(HTML)
                return
            if parsed.path == "/api/images":
                self.send_json({"images": app.images(), "classes": CLASSES})
                return
            if parsed.path == "/api/labels":
                q = parse_qs(parsed.query)
                split = q.get("split", [""])[0]
                image = unquote(q.get("image", [""])[0])
                reviewed = app.reviewed_label_path(split, image)
                draft = app.draft_label_path(split, image)
                self.send_json({"boxes": read_labels(reviewed if reviewed.exists() else draft), "classes": CLASSES})
                return
            if parsed.path == "/api/image":
                q = parse_qs(parsed.query)
                split = q.get("split", [""])[0]
                image = unquote(q.get("image", [""])[0])
                path = app.image_path(split, image)
                if not path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/labels":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode())
            split = data["split"]
            image = data["image"]
            boxes = data.get("boxes", [])
            source = app.image_path(split, image)
            target = app.reviewed_image_path(split, image)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
            write_labels(app.reviewed_label_path(split, image), boxes)
            write_data_yaml(app)
            self.send_json(
                {
                    "ok": True,
                    "reviewed": image,
                    "boxes": len(boxes),
                    "label_path": str(app.reviewed_label_path(split, image)),
                    "image_path": str(target),
                }
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=Path("data/processed/chinola-yolo-passion-only"))
    parser.add_argument("--reviewed", type=Path, default=Path("data/processed/chinola-yolo-passion-reviewed"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    app = App(args.draft, args.reviewed)
    write_data_yaml(app)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Annotation UI: http://{args.host}:{args.port}")
    print(f"Draft: {args.draft}")
    print(f"Reviewed output: {args.reviewed}")
    server.serve_forever()


if __name__ == "__main__":
    main()
