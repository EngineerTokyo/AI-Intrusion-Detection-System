from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import random
import threading
import sqlite3
import os
import time

from werkzeug.utils import secure_filename

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)

# =========================
# FOLDERS
# =========================

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# =========================
# DATABASE SETUP
# =========================

conn = sqlite3.connect("cybersecurity.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    threat TEXT,
    severity TEXT,
    status TEXT
)
""")

conn.commit()

# =========================
# GLOBAL STORAGE
# =========================

live_traffic = []
captured_packets = []
last_upload_result = None

packet_window_tracker = {}
port_scan_tracker = {}
suspicious_ip_stats = {}

data_lock = threading.Lock()

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

X = data.iloc[:, :-2].copy()
y = data.iloc[:, -2].copy()

# =========================
# ENCODE FEATURES
# =========================

feature_encoders = {}

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

# =========================
# MULTI-CLASS LABEL ENCODING
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# TRAIN TEST SPLIT FOR MAIN MODEL
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =========================
# MAIN MODEL: RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# =========================
# BINARY DATA FOR MODEL COMPARISON
# =========================

y_binary = y.apply(lambda label: "normal" if label == "normal" else "attack")

binary_encoder = LabelEncoder()
y_binary_encoded = binary_encoder.fit_transform(y_binary)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X,
    y_binary_encoded,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL COMPARISON
# =========================

comparison_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

model_results = []

best_model_name = ""
best_model_accuracy = 0

for model_name, clf in comparison_models.items():

    clf.fit(X_train_bin, y_train_bin)

    pred = clf.predict(X_test_bin)

    acc = accuracy_score(y_test_bin, pred)

    model_results.append({
        "name": model_name,
        "accuracy": round(acc * 100, 2)
    })

    if acc > best_model_accuracy:
        best_model_accuracy = acc
        best_model_name = model_name

# =========================
# MODEL COMPARISON GRAPH
# =========================

model_names = [item["name"] for item in model_results]
model_accuracies = [item["accuracy"] for item in model_results]

plt.figure(figsize=(8, 5))
plt.bar(model_names, model_accuracies)
plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("static/model_comparison.png")
plt.close()

# =========================
# CONFUSION MATRIX AND CLASSIFICATION REPORT
# =========================

rf_binary_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_binary_model.fit(X_train_bin, y_train_bin)

binary_predictions = rf_binary_model.predict(X_test_bin)

cm = confusion_matrix(y_test_bin, binary_predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Normal vs Attack")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], binary_encoder.classes_)
plt.yticks([0, 1], binary_encoder.classes_)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

report_dict = classification_report(
    y_test_bin,
    binary_predictions,
    target_names=binary_encoder.classes_,
    output_dict=True
)

classification_metrics = []

for label in binary_encoder.classes_:

    classification_metrics.append({
        "class": label,
        "precision": round(report_dict[label]["precision"] * 100, 2),
        "recall": round(report_dict[label]["recall"] * 100, 2),
        "f1_score": round(report_dict[label]["f1-score"] * 100, 2),
        "support": int(report_dict[label]["support"])
    })

overall_accuracy_binary = round(report_dict["accuracy"] * 100, 2)

# =========================
# DATASET CHARTS
# =========================

attack_counts = y.value_counts().head(5)

plt.figure(figsize=(8, 5))
attack_counts.plot(kind="bar")
plt.title("Top Attack Categories")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/attack_chart.png")
plt.close()

normal_count = len(data[data[41] == "normal"])
attack_count = len(data[data[41] != "normal"])

labels = ["Normal", "Attack"]
sizes = [normal_count, attack_count]

plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%"
)
plt.title("Normal vs Attack Traffic")
plt.savefig("static/pie_chart.png")
plt.close()

# =========================
# HELPER FUNCTIONS
# =========================

def save_detection_to_db(threat, severity, status):
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute(
        """
        INSERT INTO detections
        (time, threat, severity, status)
        VALUES (?, ?, ?, ?)
        """,
        (
            current_time,
            threat,
            severity,
            status
        )
    )

    conn.commit()


def load_history_from_db(limit=20):
    cursor.execute(
        """
        SELECT time, threat, severity, status
        FROM detections
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    history = []

    for row in rows:
        history.append({
            "time": row[0],
            "threat": row[1],
            "severity": row[2],
            "status": row[3]
        })

    return history


def get_severity_and_status(threat):
    if threat == "normal":
        return "LOW", "SAFE"

    return "HIGH", "ALERT"


def encode_uploaded_features(uploaded_features):
    uploaded_features = uploaded_features.copy()
    uploaded_features.columns = range(uploaded_features.shape[1])

    for col in uploaded_features.columns:

        uploaded_features[col] = uploaded_features[col].astype(str)

        if col in feature_encoders:

            le = feature_encoders[col]
            known_classes = set(le.classes_)

            uploaded_features[col] = uploaded_features[col].apply(
                lambda value: value if value in known_classes else le.classes_[0]
            )

            uploaded_features[col] = le.transform(uploaded_features[col])

        else:
            uploaded_features[col] = 0

    return uploaded_features

# =========================
# REAL PACKET ANALYSIS
# =========================

def process_packet(packet):
    try:
        if not packet.haslayer(IP):
            return

        now_time = time.time()
        now_text = datetime.now().strftime("%H:%M:%S")

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst

        protocol_name = "OTHER"
        dst_port = "N/A"

        if packet.haslayer(TCP):
            protocol_name = "TCP"
            dst_port = packet[TCP].dport

        elif packet.haslayer(UDP):
            protocol_name = "UDP"
            dst_port = packet[UDP].dport

        else:
            proto_num = packet[IP].proto

            if proto_num == 1:
                protocol_name = "ICMP"
            else:
                protocol_name = str(proto_num)

        packet_size = len(packet)

        with data_lock:

            recent_packets = packet_window_tracker.get(src_ip, [])
            recent_packets = [t for t in recent_packets if now_time - t <= 10]
            recent_packets.append(now_time)
            packet_window_tracker[src_ip] = recent_packets
            packet_rate = len(recent_packets)

            recent_ports = port_scan_tracker.get(src_ip, [])
            recent_ports = [item for item in recent_ports if now_time - item[0] <= 15]

            if dst_port != "N/A":
                recent_ports.append((now_time, dst_port))

            port_scan_tracker[src_ip] = recent_ports

            unique_ports = len(set(p for _, p in recent_ports)) if recent_ports else 0

            if packet_rate >= 50:
                threat_status = "DDoS ATTEMPT"
                severity = "CRITICAL"

            elif unique_ports >= 12:
                threat_status = "PORT SCAN"
                severity = "HIGH"

            elif packet_rate >= 25:
                threat_status = "TRAFFIC BURST"
                severity = "HIGH"

            elif packet_size > 1000:
                threat_status = "LARGE PACKET"
                severity = "MEDIUM"

            else:
                threat_status = "SAFE"
                severity = "LOW"

            captured_packets.append({
                "time": now_text,
                "src": src_ip,
                "dst": dst_ip,
                "protocol": protocol_name,
                "dst_port": dst_port,
                "size": packet_size,
                "packet_rate": packet_rate,
                "unique_ports": unique_ports,
                "status": threat_status,
                "severity": severity
            })

            if len(captured_packets) > 30:
                captured_packets.pop(0)

            if threat_status != "SAFE":

                previous = suspicious_ip_stats.get(src_ip, {
                    "occurrences": 0
                })

                suspicious_ip_stats[src_ip] = {
                    "ip": src_ip,
                    "dst": dst_ip,
                    "protocol": protocol_name,
                    "packet_rate": packet_rate,
                    "unique_ports": unique_ports,
                    "status": threat_status,
                    "severity": severity,
                    "last_seen": now_text,
                    "occurrences": previous.get("occurrences", 0) + 1
                }

    except Exception as e:
        print("Packet processing error:", e)


def start_sniffing():
    try:
        sniff(prn=process_packet, store=False)
    except Exception as e:
        print("Packet sniffing error:", e)


sniffer_thread = threading.Thread(target=start_sniffing)
sniffer_thread.daemon = True
sniffer_thread.start()

# =========================
# LOGIN PAGE
# =========================

@app.route("/", methods=["GET", "POST"])
def login():

    error = ""

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":
            return redirect(url_for("dashboard"))

        error = "Invalid Username or Password"

    return render_template("login.html", error=error)

# =========================
# DASHBOARD
# =========================

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    global last_upload_result

    result = ""
    severity = ""
    status = ""

    # =========================
    # LIVE TRAFFIC SIMULATOR
    # =========================

    protocols = ["TCP", "UDP", "HTTP", "FTP", "DNS"]

    activities = [
        "Normal Session",
        "Suspicious Activity",
        "Login Attempt",
        "Port Scanning",
        "Data Transfer",
        "API Request"
    ]

    ip = f"192.168.1.{random.randint(1, 255)}"
    protocol = random.choice(protocols)
    activity = random.choice(activities)

    if activity == "Port Scanning":
        traffic_status = "SUSPICIOUS"
    elif activity == "Suspicious Activity":
        traffic_status = "ALERT"
    else:
        traffic_status = random.choice(["SAFE", "SAFE", "SAFE", "SUSPICIOUS"])

    live_traffic.append({
        "ip": ip,
        "protocol": protocol,
        "activity": activity,
        "status": traffic_status
    })

    if len(live_traffic) > 12:
        live_traffic.pop(0)

    # =========================
    # POST ACTIONS
    # =========================

    if request.method == "POST":

        action = request.form.get("action")

        if action == "predict_sample":

            sample = X_test.sample(n=1)

            prediction = model.predict(sample)
            predicted_label = label_encoder.inverse_transform(prediction)

            result = predicted_label[0]
            severity, status = get_severity_and_status(result)

            save_detection_to_db(result, severity, status)

        elif action == "upload_file":

            uploaded_file = request.files.get("traffic_file")

            if uploaded_file and uploaded_file.filename != "":

                filename = secure_filename(uploaded_file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(filepath)

                try:
                    uploaded_data = pd.read_csv(filepath, header=None)

                    if uploaded_data.shape[1] >= 43:
                        uploaded_features = uploaded_data.iloc[:, :-2]

                    elif uploaded_data.shape[1] == 41:
                        uploaded_features = uploaded_data.iloc[:, :41]

                    else:
                        last_upload_result = {
                            "success": False,
                            "filename": filename,
                            "message": "Invalid file format. File must contain 41 feature columns or full NSL-KDD format."
                        }

                        uploaded_features = None

                    if uploaded_features is not None:

                        uploaded_features = uploaded_features.iloc[:, :41]
                        encoded_uploaded_features = encode_uploaded_features(uploaded_features)

                        upload_predictions = model.predict(encoded_uploaded_features)
                        predicted_labels = label_encoder.inverse_transform(upload_predictions)

                        prediction_counts = pd.Series(predicted_labels).value_counts()
                        most_common_threat = prediction_counts.index[0]

                        severity, status = get_severity_and_status(most_common_threat)

                        total_uploaded_records = len(uploaded_data)
                        normal_predictions = int((predicted_labels == "normal").sum())
                        attack_predictions = int((predicted_labels != "normal").sum())

                        save_detection_to_db(most_common_threat, severity, status)

                        last_upload_result = {
                            "success": True,
                            "filename": filename,
                            "total_records": total_uploaded_records,
                            "predicted_threat": most_common_threat,
                            "normal_predictions": normal_predictions,
                            "attack_predictions": attack_predictions,
                            "severity": severity,
                            "status": status
                        }

                except Exception as e:
                    last_upload_result = {
                        "success": False,
                        "filename": filename,
                        "message": str(e)
                    }

    # =========================
    # BUILD LIVE STATS
    # =========================

    with data_lock:
        packets_snapshot = captured_packets[::-1]
        suspicious_snapshot = list(suspicious_ip_stats.values())

    critical_alerts = len([p for p in packets_snapshot if p["severity"] == "CRITICAL"])
    high_alerts = len([p for p in packets_snapshot if p["severity"] == "HIGH"])
    medium_alerts = len([p for p in packets_snapshot if p["severity"] == "MEDIUM"])
    low_alerts = len([p for p in packets_snapshot if p["severity"] == "LOW"])

    total_captured_packets = len(packets_snapshot)
    monitored_ips = len(set(p["src"] for p in packets_snapshot)) if packets_snapshot else 0

    severity_order = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    suspicious_ips = sorted(
        suspicious_snapshot,
        key=lambda x: (
            -severity_order.get(x["severity"], 0),
            -x["occurrences"]
        )
    )

    if critical_alerts > 0:
        system_health = "CRITICAL"

    elif high_alerts > 0:
        system_health = "WARNING"

    else:
        system_health = "STABLE"

    history = load_history_from_db(limit=20)

    return render_template(
        "dashboard.html",

        accuracy=round(accuracy * 100, 2),

        result=result,
        severity=severity,
        status=status,

        total_records=len(data),
        total_attacks=attack_count,
        normal_records=normal_count,

        live_traffic=live_traffic[::-1],
        captured_packets=packets_snapshot,
        suspicious_ips=suspicious_ips,
        history=history,

        critical_alerts=critical_alerts,
        high_alerts=high_alerts,
        medium_alerts=medium_alerts,
        low_alerts=low_alerts,

        total_captured_packets=total_captured_packets,
        monitored_ips=monitored_ips,
        system_health=system_health,

        upload_result=last_upload_result,

        model_results=model_results,
        best_model_name=best_model_name,
        best_model_accuracy=round(best_model_accuracy * 100, 2),
        classification_metrics=classification_metrics,
        overall_accuracy_binary=overall_accuracy_binary
    )

# =========================
# PDF REPORT
# =========================

@app.route("/download_report")
def download_report():

    pdf_file = "CyberSecurity_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph(
        "AI-Based Intrusion Detection and Threat Intelligence Report",
        styles["Title"]
    )

    elements.append(title)
    elements.append(Spacer(1, 20))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with data_lock:
        packet_snapshot = captured_packets[-10:]
        suspicious_snapshot = list(suspicious_ip_stats.values())

    critical_count = len([p for p in captured_packets if p["severity"] == "CRITICAL"])
    high_count = len([p for p in captured_packets if p["severity"] == "HIGH"])
    medium_count = len([p for p in captured_packets if p["severity"] == "MEDIUM"])

    report_info = f"""
    <b>Generated Time:</b> {current_time}<br/>
    <b>Main Model Accuracy:</b> {round(accuracy * 100, 2)}%<br/>
    <b>Best Compared Model:</b> {best_model_name} ({round(best_model_accuracy * 100, 2)}%)<br/>
    <b>Total Dataset Records:</b> {len(data)}<br/>
    <b>Total Attack Records:</b> {attack_count}<br/>
    <b>Normal Records:</b> {normal_count}<br/>
    <b>Critical Alerts:</b> {critical_count}<br/>
    <b>High Alerts:</b> {high_count}<br/>
    <b>Medium Alerts:</b> {medium_count}<br/>
    """

    elements.append(Paragraph(report_info, styles["BodyText"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Model Comparison Results</b>", styles["Heading2"]))

    for item in model_results:
        elements.append(
            Paragraph(
                f"{item['name']} Accuracy: {item['accuracy']}%",
                styles["BodyText"]
            )
        )

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Recent Detection History</b>", styles["Heading2"]))

    history_rows = load_history_from_db(limit=10)

    for row in history_rows:
        text = f"""
        Time: {row['time']} |
        Threat: {row['threat']} |
        Severity: {row['severity']} |
        Status: {row['status']}
        """
        elements.append(Paragraph(text, styles["BodyText"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Recent Packet Monitoring Logs</b>", styles["Heading2"]))

    for packet in packet_snapshot:
        text = f"""
        Time: {packet['time']} |
        Source: {packet['src']} |
        Destination: {packet['dst']} |
        Protocol: {packet['protocol']} |
        Port: {packet['dst_port']} |
        Size: {packet['size']} |
        Rate: {packet['packet_rate']} |
        Status: {packet['status']} |
        Severity: {packet['severity']}
        """
        elements.append(Paragraph(text, styles["BodyText"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Suspicious IP Intelligence</b>", styles["Heading2"]))

    for ip_data in suspicious_snapshot[:10]:
        text = f"""
        IP: {ip_data['ip']} |
        Status: {ip_data['status']} |
        Severity: {ip_data['severity']} |
        Packet Rate: {ip_data['packet_rate']} |
        Unique Ports: {ip_data['unique_ports']} |
        Occurrences: {ip_data['occurrences']} |
        Last Seen: {ip_data['last_seen']}
        """
        elements.append(Paragraph(text, styles["BodyText"]))

    if last_upload_result:

        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<b>Uploaded Traffic File Analysis</b>", styles["Heading2"]))

        if last_upload_result["success"]:
            upload_text = f"""
            File Name: {last_upload_result['filename']}<br/>
            Total Records: {last_upload_result['total_records']}<br/>
            Predicted Threat: {last_upload_result['predicted_threat']}<br/>
            Normal Predictions: {last_upload_result['normal_predictions']}<br/>
            Attack Predictions: {last_upload_result['attack_predictions']}<br/>
            Severity: {last_upload_result['severity']}<br/>
            Status: {last_upload_result['status']}<br/>
            """
        else:
            upload_text = f"""
            File Name: {last_upload_result['filename']}<br/>
            Error: {last_upload_result['message']}<br/>
            """

        elements.append(Paragraph(upload_text, styles["BodyText"]))

    doc.build(elements)

    return send_file(pdf_file, as_attachment=True)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )