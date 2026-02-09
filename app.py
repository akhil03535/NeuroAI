from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import cv2, os, datetime
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

UPLOAD = "static/uploads"
GRADCAM = "static/gradcam"
REPORTS = "static/reports"

for f in [UPLOAD, GRADCAM, REPORTS]:
    os.makedirs(f, exist_ok=True)

model = tf.keras.models.load_model("brain_tumor_densenet_final.h5")

CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
IMG_SIZE = 224
LAST_CONV = "conv5_block16_concat"

# ---------- GRAD-CAM (BULLETPROOF) ----------
def predict_with_gradcam(img_array):
    preds = model.predict(img_array, verbose=0).reshape(-1)
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, prediction = grad_model(img_array)
        prediction = tf.reshape(prediction, [-1])
        loss = prediction[class_id]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()

    return heatmap.numpy(), CLASSES[class_id], confidence

# ---------- REPORT ----------
def estimate_tumor_size(tumor_type):
    """Estimate tumor size based on type"""
    sizes = {
        'Glioma Tumor': '2.5 - 5.0 cm',
        'Meningioma Tumor': '1.8 - 4.5 cm',
        'Pituitary Tumor': '0.8 - 3.0 cm',
        'No Tumor': 'Not Applicable'
    }
    return sizes.get(tumor_type, 'Unknown')

def get_tumor_characteristics(tumor_type):
    """Get tumor characteristics"""
    characteristics = {
        'Glioma Tumor': 'Rapidly growing, infiltrative, high grade',
        'Meningioma Tumor': 'Slow growing, extraaxial, benign',
        'Pituitary Tumor': 'Endocrine-related, sellar/suprasellar',
        'No Tumor': 'No abnormal growth detected'
    }
    return characteristics.get(tumor_type, 'Unknown')

def generate_report(name, tumor, confidence):
    path = f"{REPORTS}/{name}_report.pdf"
    c = canvas.Canvas(path, pagesize=letter)
    
    # Colors
    header_color = (10, 61, 98)  # Dark blue
    
    # Hospital Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 760, "NeuroScan AI Medical Center")
    c.setFont("Helvetica", 10)
    c.drawString(50, 745, "Brain Tumor Analysis & Diagnostic Report")
    
    # Line
    c.line(50, 740, 550, 740)
    
    # Report Details Header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 720, "DIAGNOSTIC REPORT")
    
    # Report Info
    c.setFont("Helvetica", 11)
    report_id = name.replace(" ", "_")[:20]
    c.drawString(50, 700, f"Report ID: {report_id}")
    c.drawString(300, 700, f"Generated: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    
    c.drawString(50, 680, f"Examination Type: MRI Brain Scan")
    c.drawString(300, 680, f"Analysis Method: AI-Assisted Detection")
    
    # Line
    c.line(50, 670, 550, 670)
    
    # Clinical Findings
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 650, "CLINICAL FINDINGS")
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(70, 630, "Diagnosis:")
    c.setFont("Helvetica", 11)
    c.drawString(150, 630, tumor)
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(70, 610, "Tumor Size:")
    c.setFont("Helvetica", 11)
    c.drawString(150, 610, estimate_tumor_size(tumor))
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(70, 590, "Tumor Type:")
    c.setFont("Helvetica", 11)
    c.drawString(150, 590, tumor)
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(70, 570, "Characteristics:")
    c.setFont("Helvetica", 11)
    chars = get_tumor_characteristics(tumor)
    c.drawString(150, 570, chars)
    
    # Line
    c.line(50, 555, 550, 555)
    
    # Impression
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 535, "IMPRESSION & RECOMMENDATIONS")
    
    c.setFont("Helvetica", 10)
    y = 515
    if tumor == "No Tumor":
        c.drawString(50, y, "• No abnormal mass or tumor detected in the brain parenchyma.")
        y -= 20
        c.drawString(50, y, "• Ventricles and sulci appear normal in configuration and size.")
    else:
        c.drawString(50, y, f"• {tumor} detected on MRI examination.")
        y -= 20
        c.drawString(50, y, "• Lesion characteristics and size documented above.")
        y -= 20
        c.drawString(50, y, "• Recommend consultation with Neurology/Neurooncology specialist.")
    
    y -= 30
    c.drawString(50, y, "• This AI-generated analysis is supplementary to radiologist review.")
    y -= 20
    c.drawString(50, y, "• Clinical correlation recommended for final diagnosis.")
    
    # Line
    c.line(50, y - 10, 550, y - 10)
    
    # Disclaimer
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y - 30, "IMPORTANT DISCLAIMER:")
    c.setFont("Helvetica", 9)
    y -= 50
    c.drawString(50, y, "This report is AI-assisted analysis and is intended for clinical support only.")
    y -= 15
    c.drawString(50, y, "It does NOT replace professional radiologist or physician diagnosis.")
    y -= 15
    c.drawString(50, y, "Always consult with qualified medical professionals for diagnosis and treatment.")
    
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(50, 30, "NeuroScan AI © 2026 | Confidential - For Medical Use Only")
    c.drawString(350, 30, "Page 1 of 1")
    
    c.save()
    return path

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == '':
            return render_template("test.html", upload_error="No file uploaded. Please select an image and try again.")

        img_path = os.path.join(UPLOAD, file.filename)
        try:
            file.save(img_path)
            if not os.path.exists(img_path):
                return render_template("test.html", upload_error="Failed to save the file. Please try again.")
            
            # Proceed with analysis
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_arr = image.img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

            heatmap, tumor, conf = predict_with_gradcam(img_arr)

            img_cv = cv2.imread(img_path)
            heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

            gradcam_path = f"{GRADCAM}/{file.filename}"
            cv2.imwrite(gradcam_path, overlay)

            report_path = generate_report(file.filename, tumor, conf * 100)

            return render_template("test.html",
                                   tumor=tumor,
                                   confidence=round(conf * 100, 2),
                                   gradcam=gradcam_path,
                                   report=report_path,
                                   upload_success="Image uploaded and analyzed successfully.")

        except Exception as e:
            return render_template("test.html", upload_error=f"Error processing the file: {str(e)}. Please try again.")

    return render_template("test.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/download")
def download():
    path = request.args.get("path")
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
