import os
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
import matplotlib
matplotlib.use("Agg")          # gunakan backend non-GUI
import matplotlib.pyplot as plt

# ==================== PARAMETER ====================
IMG_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42

BASE_DIR = os.path.join(
    "dataset",
    "Augmented IQ-OTHNCCD lung cancer dataset"
)

MODEL_PATH = os.path.join("models", "best_lung_cancer_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("models", "lung_cancer_model.h5")

# Kelas model (urutan harus sama dengan saat training)
MODEL_CLASSES = ["Benign cases", "Malignant cases", "Normal cases"]

# Label yang lebih enak dibaca di UI
DISPLAY_LABELS = {
    "Benign cases": "Benign (jinak)",
    "Malignant cases": "Malignant (ganas)",
    "Normal cases": "Normal"
}

print("Base dataset :", BASE_DIR)
print("Model path   :", MODEL_PATH)

# ==================== LOAD MODEL ====================
print("\nMemuat model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat.")

AUTOTUNE = tf.data.AUTOTUNE


# ==================== FUNGSI BANTUAN INFERENSI ====================
def preprocess_image(pil_img: Image.Image):
    """Resize & preprocess gambar untuk EfficientNet."""
    img = pil_img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(pil_img: Image.Image):
    """
    Kembalikan:
      - label prediksi utama (string, untuk tampilan),
      - confidence (%) prediksi utama,
      - list probabilitas tiap kelas (dalam persen).
    """
    batch = preprocess_image(pil_img)
    preds = model.predict(batch)
    probs = preds[0]  # array panjang 3 (Benign, Malignant, Normal)

    idx = int(np.argmax(probs))
    class_key = MODEL_CLASSES[idx]
    label = DISPLAY_LABELS.get(class_key, class_key)
    confidence = float(np.max(probs) * 100.0)

    prob_list = []
    for i, class_name in enumerate(MODEL_CLASSES):
        prob_list.append({
            "raw_name": class_name,
            "label": DISPLAY_LABELS.get(class_name, class_name),
            "value": float(probs[i] * 100.0)   # dalam persen (0-100)
        })

    return label, confidence, prob_list


def interpret_decision(prob_list):
    """
    Menentukan status risiko berdasarkan probabilitas:
      - high_risk: p_malignant >= 45%
      - borderline: p_malignant >= 35% dan selisih Normal-Malignant < 10%
      - low_risk: lainnya
    """
    p_normal = None
    p_malignant = None

    for p in prob_list:
        if "Normal" in p["label"]:
            p_normal = p["value"]
        if "Malignant" in p["label"]:
            p_malignant = p["value"]

    if p_normal is None or p_malignant is None:
        status = "unknown"
        note = ("Model tidak dapat menghitung status risiko dengan benar. "
                "Silakan ulangi atau konsultasikan dengan tenaga medis.")
        return status, note

    margin = abs(p_normal - p_malignant)

    # HIGH RISK – model sangat mencurigai ganas
    if p_malignant >= 45:
        status = "high_risk"
        note = (
            "Probabilitas kanker paru ganas (malignant) cukup tinggi. "
            "Hasil ini perlu dievaluasi segera oleh dokter atau radiolog. "
            "Model hanya alat bantu dan tidak menggantikan diagnosis klinis."
        )
    # BORDERLINE – model ragu antara Normal dan Malignant
    elif p_malignant >= 35 and margin < 10:
        status = "borderline"
        note = (
            "Model menunjukkan kemungkinan kanker paru ganas yang cukup bermakna "
            "dan hasil berada pada zona abu-abu (perbedaan dengan kelas Normal tidak besar). "
            "Disarankan untuk pemeriksaan lanjutan dan konsultasi dengan tenaga medis."
        )
    # LOW RISK – cenderung normal/jinak, tapi tetap bukan jaminan
    else:
        status = "low_risk"
        note = (
            "Model cenderung mengklasifikasikan sebagai Normal / jinak. "
            "Meski demikian, hasil ini tetap tidak boleh dijadikan dasar tunggal keputusan klinis. "
            "Jika terdapat gejala klinis, mohon konsultasi ke dokter."
        )

    return status, note


# ==================== FUNGSI EVALUASI MODEL (AKURASI + CM) ====================
def compute_eval_metrics():
    """
    Menghitung akurasi, per-class precision/recall/F1, dan menyimpan confusion matrix
    sebagai gambar PNG di folder static/. Dipanggil sekali saat aplikasi start.
    """
    print("\nMempersiapkan validation set untuk evaluasi...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = val_ds.class_names
    print("Kelas (validation):", class_names)

    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    y_true = []
    y_pred = []

    print("Menghitung prediksi di validation set...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds_idx = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix & akurasi
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Classification report (structured)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Per-class metrics untuk frontend
    per_class_metrics = []
    for cls_name in class_names:
        stats = report[cls_name]
        per_class_metrics.append({
            "name": DISPLAY_LABELS.get(cls_name, cls_name),
            "precision": float(stats["precision"] * 100.0),
            "recall": float(stats["recall"] * 100.0),
            "f1": float(stats["f1-score"] * 100.0),
            "support": int(stats["support"])
        })

    # Simpan confusion matrix sebagai gambar di static/
    os.makedirs("static", exist_ok=True)
    cm_filename = "confusion_matrix_val.png"
    cm_path = os.path.join("static", cm_filename)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Validation Set")

    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close(fig)

    print(f"Akurasi validasi: {acc:.4f}")
    print(f"Confusion matrix image disimpan di: {cm_path}")

    return acc, per_class_metrics, cm_filename


# ==================== INISIALISASI EVALUASI GLOBAL ====================
print("\nMenghitung metrik evaluasi (ini hanya sekali saat app start)...")
APP_ACCURACY, APP_CLASS_METRICS, CM_FILENAME = compute_eval_metrics()

# ==================== APP FLASK ====================
app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        app_accuracy=APP_ACCURACY,
        class_metrics=APP_CLASS_METRICS,
        cm_filename=CM_FILENAME
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Gagal membuka gambar: {str(e)}"}), 400

    label, confidence, prob_list = predict_image(img)
    status, note = interpret_decision(prob_list)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2),
        "probabilities": prob_list,
        "status": status,
        "note": note
    })


if __name__ == "__main__":
    app.run(debug=True)
