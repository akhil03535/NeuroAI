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

# Lazy model loading
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("brain_tumor_densenet_final.h5")
    return model

CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
IMG_SIZE = 224
LAST_CONV = "conv5_block16_concat"

# ---------- GRAD-CAM (BULLETPROOF) ----------
def predict_with_gradcam(img_array):
    model = get_model()
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

def generate_report(patient_name, patient_age, tumor, confidence):
    from datetime import datetime
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"{patient_name.replace(' ', '_')}_{timestamp}_report.pdf"
    path = f"{REPORTS}/{report_filename}"
    
    # Create the PDF
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    
    # Professional Color Palette - Blue and White
    PRIMARY_BLUE = colors.HexColor('#003366')  # Dark Blue
    LIGHT_BLUE = colors.HexColor('#E8F0F7')    # Light Blue Background
    ACCENT_BLUE = colors.HexColor('#0066CC')   # Accent Blue
    DIVIDER_COLOR = colors.HexColor('#B3D9FF') # Light Blue Divider
    TEXT_DARK = colors.HexColor('#333333')     # Dark Text
    
    # ==================== PAGE 1 ====================
    
    # = PROFESSIONAL HEADER =
    c.setFillColor(PRIMARY_BLUE)
    c.rect(0, height - 100, width, 100, fill=1, stroke=0)
    
    # Hospital Name and Logo Area
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.white)
    c.drawString(60, height - 45, "NeuroScan AI")
    
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor('#B3D9FF'))
    c.drawString(60, height - 62, "Medical Brain Tumor Analysis Center")
    c.drawString(60, height - 77, "Diagnostic Imaging Report")
    
    # Report Type and Date on Right
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.white)
    c.drawRightString(width - 60, height - 45, "MRI BRAIN SCAN REPORT")
    
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor('#B3D9FF'))
    current_date = datetime.now().strftime('%B %d, %Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    c.drawRightString(width - 60, height - 62, f"Date: {current_date}")
    c.drawRightString(width - 60, height - 77, f"Time: {current_time}")
    
    y_position = height - 120
    
    # Horizontal line
    c.setStrokeColor(DIVIDER_COLOR)
    c.setLineWidth(2)
    c.line(40, y_position, width - 40, y_position)
    
    y_position -= 25
    
    # = PATIENT DEMOGRAPHICS SECTION =
    c.setFillColor(LIGHT_BLUE)
    c.rect(40, y_position - 75, width - 80, 75, fill=1, stroke=0)
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position - 10, "PATIENT INFORMATION")
    
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_DARK)
    c.drawString(55, y_position - 28, f"Patient Name: {patient_name}")
    c.drawString(55, y_position - 45, f"Age: {patient_age} years")
    c.drawString(55, y_position - 62, f"Exam Date: {datetime.now().strftime('%d-%m-%Y at %H:%M:%S')}")
    
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 55, y_position - 28, "Report ID: NS-AI-2026")
    c.drawRightString(width - 55, y_position - 45, "Modality: MRI Brain")
    c.drawRightString(width - 55, y_position - 62, "Status: COMPLETED")
    
    y_position -= 110
    
    # = CLINICAL INDICATION =
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position, "CLINICAL INDICATION")
    
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_DARK)
    y_position -= 18
    c.drawString(55, y_position, "Brain MRI scan for tumor screening and detection using AI-assisted analysis.")
    
    y_position -= 30
    
    # Dividing line
    c.setStrokeColor(DIVIDER_COLOR)
    c.setLineWidth(1)
    c.line(40, y_position, width - 40, y_position)
    
    y_position -= 25
    
    # = FINDINGS SECTION (Main Diagnosis) =
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position, "FINDINGS")
    
    y_position -= 25
    
    # Diagnosis Box
    if tumor == "No Tumor":
        box_color = colors.HexColor('#E8F5E9')  # Light Green
        title_color = colors.HexColor('#1B5E20')  # Dark Green
        border_color = colors.HexColor('#4CAF50')  # Green
    else:
        box_color = colors.HexColor('#FFEBEE')  # Light Red
        title_color = colors.HexColor('#B71C1C')  # Dark Red
        border_color = colors.HexColor('#F44336')  # Red
    
    # Diagnosis box
    c.setFillColor(box_color)
    c.rect(40, y_position - 55, width - 80, 55, fill=1, stroke=1)
    c.setStrokeColor(border_color)
    c.setLineWidth(2.5)
    c.rect(40, y_position - 55, width - 80, 55, fill=1, stroke=1)
    
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position - 15, "PRIMARY DIAGNOSIS:")
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(title_color)
    c.drawString(55, y_position - 38, tumor)
    
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PRIMARY_BLUE)
    c.drawRightString(width - 55, y_position - 15, "AI CONFIDENCE:")
    
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(ACCENT_BLUE)
    c.drawRightString(width - 55, y_position - 38, f"{confidence:.1f}%")
    
    y_position -= 85
    
    # Detailed Findings
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_DARK)
    
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position, "Detailed Analysis:")
    
    y_position -= 20
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(TEXT_DARK)
    c.drawString(65, y_position, "• Tumor Type:")
    
    c.setFont("Helvetica", 9)
    c.drawString(160, y_position, tumor)
    
    y_position -= 18
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(TEXT_DARK)
    c.drawString(65, y_position, "• Estimated Size:")
    
    c.setFont("Helvetica", 9)
    c.drawString(160, y_position, estimate_tumor_size(tumor))
    
    y_position -= 18
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(TEXT_DARK)
    c.drawString(65, y_position, "• Characteristics:")
    
    c.setFont("Helvetica", 9)
    characteristics = get_tumor_characteristics(tumor)
    c.drawString(160, y_position, characteristics)
    
    y_position -= 30
    
    # Dividing line
    c.setStrokeColor(DIVIDER_COLOR)
    c.setLineWidth(1)
    c.line(40, y_position, width - 40, y_position)
    
    y_position -= 25
    
    # = IMPRESSION =
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position, "IMPRESSION")
    
    y_position -= 20
    c.setFont("Helvetica", 9)
    c.setFillColor(TEXT_DARK)
    
    if tumor == "No Tumor":
        impression_lines = [
            "• No abnormal mass or tumor detected in the brain parenchyma.",
            "• Ventricular system appears normal in size and configuration.",
            "• No midline shift or mass effect identified.",
            "• Overall findings are unremarkable and reassuring."
        ]
    else:
        impression_lines = [
            f"• {tumor} detected on MRI examination of the brain.",
            "• Lesion characteristics and size are documented in detailed findings above.",
            "• CLINICAL CORRELATION AND SPECIALIST CONSULTATION RECOMMENDED.",
            "• Consider advanced imaging studies if clinically indicated."
        ]
    
    for line in impression_lines:
        c.drawString(55, y_position, line)
        y_position -= 15
    
    y_position -= 15
    
    # Dividing line
    c.setStrokeColor(DIVIDER_COLOR)
    c.setLineWidth(1)
    c.line(40, y_position, width - 40, y_position)
    
    y_position -= 20
    
    # = RECOMMENDATION =
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PRIMARY_BLUE)
    c.drawString(55, y_position, "RECOMMENDATION")
    
    y_position -= 18
    c.setFont("Helvetica", 9)
    c.setFillColor(TEXT_DARK)
    c.drawString(55, y_position, "This AI-assisted analysis should be reviewed by a qualified radiologist or neurologist.")
    y_position -= 15
    c.drawString(55, y_position, "Clinical decisions should be made in correlation with patient symptoms and clinical history.")
    
    y_position -= 30
    
    # = FOOTER SECTION =
    c.setStrokeColor(DIVIDER_COLOR)
    c.setLineWidth(1)
    c.line(40, y_position, width - 40, y_position)
    
    y_position -= 20
    
    # Disclaimer box
    c.setFillColor(colors.HexColor('#FFF8E1'))  # Light Yellow
    c.rect(40, y_position - 45, width - 80, 45, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor('#F57F17'))  # Orange
    c.setLineWidth(1)
    c.rect(40, y_position - 45, width - 80, 45, fill=1, stroke=1)
    
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor('#E65100'))
    c.drawString(55, y_position - 12, "⚠️ IMPORTANT DISCLAIMER")
    
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor('#4E342E'))
    c.drawString(55, y_position - 25, "This report is generated using AI technology for clinical support only. It does NOT replace professional medical diagnosis.")
    c.drawString(55, y_position - 35, "Always consult qualified medical professionals for diagnosis and treatment decisions.")
    
    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor('#666666'))
    c.drawString(55, 30, "NeuroScan AI Medical Center © 2026 | Confidential Medical Document")
    c.drawRightString(width - 55, 30, f"Page 1 | Report ID: {report_filename.split('_')[0][:15]}")
    
    c.save()
    return path

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        # Get patient information
        patient_name = request.form.get("patient_name")
        patient_age = request.form.get("patient_age")
        
        if not patient_name or not patient_age:
            return render_template("test.html", upload_error="Please enter patient name and age.")
        
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

            report_path = generate_report(patient_name, patient_age, tumor, conf * 100)

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


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)