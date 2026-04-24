from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
import os
import math
from ultralytics import YOLO
from datetime import datetime

# ----------------------------
# App / Config
# ----------------------------
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = '1a2b3c4d5e'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Drawing config (GREEN ONLY)
# ----------------------------
BOX_COLOR = (0, 255, 0)        # green
BOX_THICKNESS = 2
LABEL_BG_COLOR = (0, 255, 0)   # green background
LABEL_TEXT_COLOR = (0, 0, 0)   # black text
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 1

# ----------------------------
# Model (load once)
# ----------------------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Custom waste classification labels
CUSTOM_CLASS_NAMES = [
    'Can', 'Cardboard', 'Egg-Shell', 'Food-Waste', 'Glass', 'Glass-Bottle',
    'Leaves', 'Metal', 'Mug', 'Paper', 'Paper-Cup', 'Plastic',
    'Plastic Wrapper', 'Plastic-Bottle', 'Plastic-Cup',
    'Styrofoam', 'Tetra-Pack'
]

# Map IDs to names (0 → Can, 1 → Cardboard, ...)
MODEL_CLASS_NAMES = {i: name for i, name in enumerate(CUSTOM_CLASS_NAMES)}

# Focus on all waste classes
FOCUS_CLASSES = set(CUSTOM_CLASS_NAMES)

# Precautionary messages (can be customized per class)
precautionary_map = {
    'Can': 'Dispose in recycling bin.',
    'Cardboard': 'Flatten and recycle.',
    'Egg-Shell': 'Can be composted.',
    'Food-Waste': 'Dispose in compost bin.',
    'Glass': 'Handle carefully, recycle.',
    'Glass-Bottle': 'Recycle at bottle banks.',
    'Leaves': 'Use for composting or mulch.',
    'Metal': 'Recycle in metal scrap bin.',
    'Mug': 'Reuse if possible, otherwise recycle.',
    'Paper': 'Recycle in paper bin.',
    'Paper-Cup': 'Check local recycling rules.',
    'Plastic': 'Recycle in plastic bin.',
    'Plastic Wrapper': 'Dispose in appropriate plastic waste.',
    'Plastic-Bottle': 'Recycle after cleaning.',
    'Plastic-Cup': 'Recycle or avoid single-use.',
    'Styrofoam': 'Non-recyclable, dispose properly.',
    'Tetra-Pack': 'Recycle if facilities exist.'
}

# ----------------------------
# Helpers
# ----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_label_with_bg(img, text, org):
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    x, y = org
    tl = (x, y - th - baseline - 4)
    br = (x + tw + 6, y + 4)
    h, w = img.shape[:2]
    tl = (max(0, tl[0]), max(0, tl[1]))
    br = (min(w - 1, br[0]), min(h - 1, br[1]))
    cv2.rectangle(img, tl, br, LABEL_BG_COLOR, thickness=cv2.FILLED)
    cv2.putText(img, text, (x + 3, y - 3), LABEL_FONT, LABEL_SCALE, LABEL_TEXT_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

def annotate_and_collect(img):
    results = model(img, stream=False, conf=0.05, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.floor(float(box.conf[0]) * 100 + 0.5) / 100.0
            cls_id = int(box.cls[0])

            # Map class ID safely to custom names
            class_name = MODEL_CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}")

            if class_name not in FOCUS_CLASSES:
                continue

            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'precautionary': precautionary_map.get(class_name, '')
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            draw_label_with_bg(img, f"{class_name} {conf}", (max(0, x1), max(20, y1)))

    return img, detections

def process_image_bgr(img_bgr):
    if img_bgr is None:
        return None, []
    annotated, detections = annotate_and_collect(img_bgr)
    return annotated, detections

def video_detection(img_path):
    img = cv2.imread(img_path)
    return process_image_bgr(img)

# ----------------------------
# Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == "admin" and password == "admin":
            session['user'] = username
            return render_template('index.html')
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home1')
def home1():
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        img, detections = video_detection(save_path)
        if img is None:
            flash('Could not read the uploaded image.')
            return redirect(request.url)

        result_img_name = 'result.jpg'
        result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_img_name)
        cv2.imwrite(result_img_path, img)

        return render_template('home.html', res=1, filename=result_img_name, detections=detections)

    flash('File type not allowed')
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ----------------------------
# Webcam page + snapshot upload
# ----------------------------
@app.route('/webcam_page')
def webcam_page():
    return render_template('webcam.html')

@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    filename = secure_filename(f"webcam_{ts}.png")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    img, detections = process_image_bgr(img)
    if img is None:
        return jsonify({'error': 'failed to read image'}), 400

    result_name = f"webcam_result_{ts}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
    cv2.imwrite(result_path, img)

    result_url = url_for('display_image', filename=result_name)
    return jsonify({'result_url': result_url, 'detections': detections})

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True,use_reloader=False)
