"""SpendSense AI - Flask web app around the purchase-amount model.
Run from the project root:  python webapp/app.py   ->  http://127.0.0.1:5000
(needs models/model.joblib: run  python src/train_model.py  once)"""
import os, re, json, csv, secrets, sqlite3, functools
from datetime import datetime
import joblib, pandas as pd
from flask import (Flask, render_template, request, redirect, url_for, session, flash,
                   g, abort, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MODEL_PATH = os.path.join(ROOT, "models", "model.joblib")
OUTPUTS = os.path.join(ROOT, "outputs")
INSTANCE = os.path.join(BASE, "instance")
DB_PATH = os.path.join(INSTANCE, "app.db")
os.makedirs(INSTANCE, exist_ok=True)

app = Flask(__name__)
key_file = os.path.join(INSTANCE, "secret.key")
if not os.path.exists(key_file):
    open(key_file, "w").write(secrets.token_hex(32))
app.secret_key = os.environ.get("SECRET_KEY") or open(key_file).read()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

AGE_MAP = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------- database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL, inputs TEXT NOT NULL, amount REAL NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id));""")
    c.commit(); c.close()
init_db()

# ---------- model ----------
_bundle = None
def get_bundle():
    global _bundle
    if _bundle is None and os.path.exists(MODEL_PATH):
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

def predict_amount(f, bundle):
    row = {"Gender": int(f["gender"] == "M"), "Age": AGE_MAP[f["age"]], "Occupation": f["occupation"],
           "Stay_In_Current_City_Years": f["stay"], "Marital_Status": f["marital"],
           "Product_Category_1": f["cat1"], "Product_Category_2": f["cat2"], "Product_Category_3": f["cat3"],
           "User_Freq": bundle["median_user_freq"], "Product_Freq": bundle["median_product_freq"],
           "City_Category_A": int(f["city"] == "A"), "City_Category_B": int(f["city"] == "B"),
           "City_Category_C": int(f["city"] == "C")}
    X = pd.DataFrame([row])[bundle["columns"]]
    return float(bundle["model"].predict(X)[0])

# ---------- auth + csrf ----------
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]
app.jinja_env.globals["csrf_token"] = csrf_token

@app.before_request
def check_csrf():
    if request.method == "POST":
        sent = request.form.get("_csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            abort(400, "Invalid or missing form token. Go back and try again.")

@app.before_request
def load_user():
    g.user = None
    if session.get("uid"):
        g.user = db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()

def login_required(view):
    @functools.wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            flash("Please sign in to continue.", "info")
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped

def safe_next(n):
    return n if n and n.startswith("/") and not n.startswith("//") else url_for("dashboard")

@app.template_filter("inr")
def inr(v): return f"₹ {v:,.0f}"

@app.template_filter("when")
def when(s): return datetime.fromisoformat(s).strftime("%d %b %Y, %H:%M")

# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user: return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        err = None
        if not (2 <= len(name) <= 60): err = "Please enter your name."
        elif not EMAIL_RE.match(email): err = "Enter a valid email address."
        elif len(pw) < 8: err = "Password must be at least 8 characters."
        elif db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            err = "An account with this email already exists."
        if err:
            flash(err, "error")
            return render_template("register.html", name=name, email=email)
        cur = db().execute("INSERT INTO users(name,email,pw_hash,created_at) VALUES(?,?,?,?)",
                           (name, email, generate_password_hash(pw), datetime.now().isoformat()))
        db().commit()
        session.clear(); session["uid"] = cur.lastrowid
        flash(f"Welcome, {name}!", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", name="", email="")

@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user: return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if u and check_password_hash(u["pw_hash"], request.form.get("password", "")):
            session.clear(); session["uid"] = u["id"]
            return redirect(safe_next(request.args.get("next")))
        flash("Incorrect email or password.", "error")
        return render_template("login.html", email=email)
    return render_template("login.html", email="")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    rows = db().execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (g.user["id"],)).fetchall()
    stats = {"count": len(rows),
             "avg": sum(r["amount"] for r in rows) / len(rows) if rows else 0,
             "max": max((r["amount"] for r in rows), default=0)}
    recent = [{"r": r, "i": json.loads(r["inputs"])} for r in rows[:4]]
    return render_template("dashboard.html", stats=stats, recent=recent, active="dashboard")

def parse_form(form):
    def num(k, lo, hi):
        v = int(form.get(k, ""))
        if not lo <= v <= hi: raise ValueError(k)
        return v
    f = {"gender": form.get("gender"), "age": form.get("age"), "city": form.get("city"),
         "occupation": num("occupation", 0, 20), "marital": num("marital", 0, 1),
         "stay": num("stay", 0, 4), "cat1": num("cat1", 1, 18),
         "cat2": num("cat2", 0, 18), "cat3": num("cat3", 0, 18)}
    if f["gender"] not in ("M", "F") or f["age"] not in AGE_MAP or f["city"] not in ("A", "B", "C"):
        raise ValueError("choice")
    return f

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    bundle = get_bundle()
    if request.method == "POST":
        if bundle is None:
            flash("Model not found. Run  python src/train_model.py  first.", "error")
            return redirect(url_for("predict"))
        try:
            f = parse_form(request.form)
        except (ValueError, TypeError):
            flash("Please fill in every field with valid values.", "error")
            return redirect(url_for("predict"))
        amount = predict_amount(f, bundle)
        cur = db().execute("INSERT INTO predictions(user_id,created_at,inputs,amount) VALUES(?,?,?,?)",
                           (g.user["id"], datetime.now().isoformat(), json.dumps(f), amount))
        db().commit()
        return redirect(url_for("result", pid=cur.lastrowid))
    return render_template("predict.html", ages=list(AGE_MAP), model_ready=bundle is not None, active="predict")

@app.route("/result/<int:pid>")
@login_required
def result(pid):
    r = db().execute("SELECT * FROM predictions WHERE id=? AND user_id=?", (pid, g.user["id"])).fetchone()
    if not r: abort(404)
    b = get_bundle(); mae = b["metrics"]["mae"] if b else 0
    return render_template("result.html", r=r, i=json.loads(r["inputs"]), mae=mae,
                           low=max(r["amount"] - mae, 0), high=r["amount"] + mae, active="predict")

@app.route("/history")
@login_required
def history():
    rows = db().execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (g.user["id"],)).fetchall()
    return render_template("history.html", items=[{"r": r, "i": json.loads(r["inputs"])} for r in rows], active="history")

@app.route("/history/<int:pid>/delete", methods=["POST"])
@login_required
def delete_prediction(pid):
    db().execute("DELETE FROM predictions WHERE id=? AND user_id=?", (pid, g.user["id"])); db().commit()
    flash("Prediction deleted.", "info")
    return redirect(url_for("history"))

@app.route("/models")
@login_required
def models_page():
    table = []
    p = os.path.join(OUTPUTS, "model_comparison.csv")
    if os.path.exists(p):
        table = list(csv.DictReader(open(p, newline="")))
    imgs = [f for f in ["01_rmse_comparison.png", "02_r2_comparison.png", "03_actual_vs_predicted.png",
                        "04_feature_importance.png"] if os.path.exists(os.path.join(OUTPUTS, f))]
    return render_template("models.html", table=table, imgs=imgs, active="models")

@app.route("/outputs/<path:name>")
@login_required
def output_file(name):
    if not name.lower().endswith(".png"): abort(404)
    return send_from_directory(OUTPUTS, name)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
