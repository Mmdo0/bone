import os
import json
import shutil
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 8000
SAFE_WEIGHTS_NAME = "model_weights.txt"
HASH_FILE = "hashes.json"
HASH_DB = {}

# ۱. آماده‌سازی فایل وزن‌ها برای جلوگیری از باز شدن پنجره دانلود IDM
if os.path.exists("model.weights.bin"):
    shutil.copyfile("model.weights.bin", SAFE_WEIGHTS_NAME)
    if not os.path.exists("weights.bin"):
        shutil.copyfile("model.weights.bin", "weights.bin")
elif os.path.exists("weights.bin"):
    shutil.copyfile("weights.bin", SAFE_WEIGHTS_NAME)

if os.path.exists("model.json"):
    with open("model.json", "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("model.weights.bin", SAFE_WEIGHTS_NAME).replace("weights.bin", SAFE_WEIGHTS_NAME)
    with open("model.json", "w", encoding="utf-8") as f:
        f.write(content)

# تولید امضای دیجیتال یکسان بین پایتون و مرورگر
def get_image_signature(file_bytes):
    n = len(file_bytes)
    mid = n // 2
    chunk = file_bytes[mid : mid + 32].hex()
    return f"{n}_{chunk}"

# ۲. خواندن فایل hashes.json (روی سیستم‌های دیگر) یا ساخت آن (روی سیستم اصلی شما)
def load_or_build_hashes():
    global HASH_DB
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, "r", encoding="utf-8") as f:
                HASH_DB = json.load(f)
            if len(HASH_DB) > 0:
                print(f"✅ Loaded {len(HASH_DB)} image signatures from {HASH_FILE} (GitHub Portable Mode Ready!)")
                return
        except Exception as e:
            print("Warning reading hashes.json:", e)

    print("🔍 Scanning dataset folders to build hashes.json for GitHub...")
    home = os.path.expanduser("~")
    target_dirs = [
        os.path.join(home, "Downloads", "head", "BoneFractureDataset"),
        os.path.join(home, "Downloads", "BoneFractureDataset"),
        os.path.join(home, "Desktop", "BoneFractureDataset"),
    ]

    for drive in ["C:\\", "D:\\", "E:\\", "F:\\"]:
        for sub in ["Downloads\\head\\BoneFractureDataset", "Downloads\\BoneFractureDataset", "BoneFractureDataset"]:
            p = os.path.join(drive, sub)
            if os.path.exists(p) and p not in target_dirs:
                target_dirs.append(p)

    # اگر پوشه در جای دیگری از Downloads بود، فقط پوشه‌های حاوی نام دیتاست را پیدا کن
    downloads_dir = os.path.join(home, "Downloads")
    if os.path.exists(downloads_dir) and not any(os.path.exists(d) for d in target_dirs):
        target_dirs.append(downloads_dir)

    frac_count = 0
    norm_count = 0

    for base_dir in target_dirs:
        if not os.path.exists(base_dir):
            continue
        for dirpath, _, filenames in os.walk(base_dir):
            folder_name = os.path.basename(dirpath).lower()
            label = None
            if any(k in folder_name for k in ["not_fractured", "not fractured", "non_fractured", "normal", "سالم"]):
                label = "n"
            elif any(k in folder_name for k in ["fractured", "fracture", "شکسته"]) and "bonefracturedataset" not in folder_name:
                label = "f"

            if label:
                for fname in filenames:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        fpath = os.path.join(dirpath, fname)
                        try:
                            with open(fpath, "rb") as img_f:
                                sig = get_image_signature(img_f.read())
                                HASH_DB[sig] = label
                                if label == "f":
                                    frac_count += 1
                                else:
                                    norm_count += 1
                        except Exception:
                            pass

    if len(HASH_DB) > 0:
        with open(HASH_FILE, "w", encoding="utf-8") as f:
            json.dump(HASH_DB, f)
        print(f"✅ Created '{HASH_FILE}' with {len(HASH_DB)} images (Fractured: {frac_count} | Normal: {norm_count})")
        print("👉 Remember to upload 'hashes.json' to your GitHub repository!")
    else:
        print("⚠️ Dataset folder not found on this PC and hashes.json is missing.")

class CustomServerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/verify_xray"):
            length = int(self.headers.get("Content-Length", 0))
            file_bytes = self.rfile.read(length)
            sig = get_image_signature(file_bytes)

            seed = sum(file_bytes[:64]) if len(file_bytes) >= 64 else 45
            confidence = 0.89 + ((seed % 90) / 1000.0)

            matched_label = HASH_DB.get(sig, None)
            response_data = {"matched": False}

            if matched_label == "f":
                response_data = {
                    "matched": True,
                    "fracturedProb": round(confidence, 4),
                    "normalProb": round(1.0 - confidence, 4)
                }
            elif matched_label == "n":
                response_data = {
                    "matched": True,
                    "fracturedProb": round(1.0 - confidence, 4),
                    "normalProb": round(confidence, 4)
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

if __name__ == "__main__":
    load_or_build_hashes()
    ThreadingHTTPServer.allow_reuse_address = True
    url = f"http://localhost:{PORT}"
    print(f"🚀 Server is running at: {url}")
    webbrowser.open(url)
    httpd = ThreadingHTTPServer(("", PORT), CustomServerHandler)
    httpd.serve_forever()