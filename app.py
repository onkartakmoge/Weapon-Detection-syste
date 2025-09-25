import os, sqlite3, cv2, base64, datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from flask_socketio import SocketIO, emit
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

# ---------------- CONFIG ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key_here"  # change to something secure

DB_PATH = "database.db"

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Flask-Mail (configure Gmail SMTP)
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="onkarshivaji12@gmail.com",     # <-- CHANGE THIS
    MAIL_PASSWORD="sulldyxbawtlaxkx"         # <-- CHANGE THIS (Google App Password)
)
mail = Mail(app)

# YOLOv8 model
model = YOLO("yolov8n.pt")  # small fast model; replace with your fine-tuned weights if needed

# ---------------- USER MODEL ----------------
class User(UserMixin):
    def __init__(self, id, email, password):
        self.id = id
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2])
    return None

# ---------------- INIT DB ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                password TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                label TEXT,
                confidence REAL,
                user_email TEXT)""")
    conn.commit()
    conn.close()

init_db()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (email,password) VALUES (?,?)",(email,password))
            conn.commit()
            conn.close()
            flash("Registration successful, please login.", "success")
            return redirect(url_for("login"))
        except:
            flash("Email already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?",(email,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[2], password):
            user = User(row[0], row[1], row[2])
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", user=current_user)

# ---------------- SOCKET.IO HANDLER ----------------
@socketio.on("frame")
def handle_frame(data):
    try:
        # Decode frame
        img_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame, conf=0.35)[0]
        boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf, "label": label
            })

            # Save alert to DB
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO alerts (timestamp,label,confidence,user_email) VALUES (?,?,?,?)",
                      (datetime.datetime.now().isoformat(), label, conf, current_user.email))
            conn.commit()
            conn.close()

            # Send email notification
            msg = Message(f"🚨 Weapon Detected: {label}",
                          sender="your_email@gmail.com",
                          recipients=[current_user.email])
            msg.body = f"A {label} was detected with {conf*100:.1f}% confidence at {datetime.datetime.now()}."
            try:
                mail.send(msg)
            except Exception as e:
                print("Email error:", e)

        emit("detections", {"boxes": boxes})

    except Exception as e:
        print("Frame handling error:", e)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
