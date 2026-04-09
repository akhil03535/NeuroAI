# 🧠 NeuroScan AI - Brain Tumor Detection System

> **AI-Powered MRI Analysis for Medical Professionals**


## 🎯 About

NeuroScan AI is an intelligent medical imaging system that uses deep learning to detect brain tumors from MRI scans. The application provides clinicians with:

- **AI-Assisted Diagnosis** - Automatic tumor detection and classification
- **Visual Analysis** - Grad-CAM heatmaps showing tumor regions
- **Professional Reports** - Hospital-grade PDF diagnostic reports
- **Real-time Analysis** - Fast, file-based processing

This tool is designed to **support medical professionals** in their diagnostic process, not replace them.

---

## ✨ Key Features

### 🔬 **Advanced AI Model**
- Deep neural network trained on extensive medical imaging data
- Detects 4 tumor types: Glioma, Meningioma, Pituitary, No Tumor
- 95%+ accuracy on validation datasets
- Real-time inference (2-5 seconds per scan)

### 🎨 **Explainable AI**
- Grad-CAM visualization highlights tumor regions
- Visual heatmaps overlay on original MRI scans
- Confidence scores for each prediction
- Transparent decision-making process

### 📋 **Professional Reporting**
- Automated PDF report generation
- Patient name, age, and analysis timestamp
- Clinical findings and recommendations
- Medical-grade formatting (Blue & White theme)
- Color-coded results for easy interpretation

### � **File-Based Storage**
- MRI images saved in organized folders
- Generated reports stored with timestamps
- Grad-CAM visualizations auto-saved
- Simple folder-based file management

### 🎭 **Modern UI/UX**
- Responsive design (Desktop, Tablet, Mobile)
- Smooth loading animations
- Intuitive navigation
- Professional medical aesthetic

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Flask (Python) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **AI/ML** | TensorFlow, Keras, OpenCV |
| **Reports** | ReportLab |
| **Storage** | File-based (Folders & PDFs) |

---

## 📦 Installation

### Requirements
- Python 3.8 or higher
- 8GB RAM minimum
- 2GB free disk space
- Windows/Mac/Linux

### Quick Start

**1. Clone Repository**
```bash
git clone <repository-url>
cd BrainTumorDetection
```

**2. Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Model File**
Ensure `brain_tumor_densenet_final.h5` exists in project root

**5. Run Application**
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🚀 How to Use

### Step 1: Enter Patient Information
- Navigate to **Analyze** section
- Enter patient name
- Enter patient age

### Step 2: Upload MRI Image
- Click the upload box
- Select MRI brain scan image (JPG, PNG, BMP, TIFF)
- Image will preview

### Step 3: Analyze
- Click **"Analyze MRI Image"** button
- Wait for AI processing (loading animation shown)
- Results display in 2-5 seconds

### Step 4: View Results
- **Diagnosis**: Primary tumor classification
- **Confidence**: AI prediction confidence percentage
- **Visualization**: Grad-CAM heatmap with highlighted regions
- **Recommendations**: Clinical guidance

### Step 5: Download Report
- Click **"Download Medical Report (PDF)"**
- Professional report includes:
  - Patient demographics
  - Analysis timestamp
  - Diagnosis findings
  - Tumor characteristics
  - Clinical recommendations
  - Medical disclaimers

---

## 📁 Project Structure

```
BrainTumorDetection/
│
├── 📄 app.py                          # Main Flask application
├── 🧠 brain_tumor_densenet_final.h5  # Pre-trained AI model
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Documentation
├── 🚫 .gitignore                      # Git configuration
│
├── 📁 static/
│   ├── 🎨 css/
│   │   └── style.css                  # Responsive styling
│   ├── 📤 uploads/                    # User MRI scans
│   ├── 🔥 gradcam/                    # Visualization outputs
│   └── 📊 reports/                    # Generated PDFs
│
└── 📁 templates/
    ├── base.html                      # Base template
    ├── index.html                     # Home page
    ├── test.html                      # Analysis page
    ├── about.html                     # About page
    └── contact.html                   # Contact page
```

---

## 🎯 Supported Tumor Types

| Tumor Type | Description | Characteristics |
|-----------|-------------|-----------------|
| **Glioma** | Most common adult brain tumor | Rapidly growing, infiltrative |
| **Meningioma** | Originates from brain membrane | Slow growing, usually benign |
| **Pituitary** | Develops in pituitary gland | Endocrine-related symptoms |
| **No Tumor** | Healthy brain scan | Normal findings |

---

## 🔍 API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/test` | GET/POST | Analysis interface & processing |
| `/about` | GET | Information about tumors |
| `/contact` | GET | Contact & support |
| `/download` | GET | Download PDF reports |

---

## 💡 Model Architecture

**DenseNet (Densely Connected Convolution Networks)**

```
Input: 224 × 224 pixel MRI image
  ↓
DenseNet Backbone
  ↓
Feature Extraction
  ↓
Grad-CAM Layer (conv5_block16_concat)
  ↓
Classification Head
  ↓
Output: Tumor Class + Confidence Score
```

---

## ⚡ Performance

- **Inference Time**: 2-5 seconds per image
- **Accuracy**: 95%+ on test dataset
- **Max Image Size**: 10MB
- **Supported Formats**: JPG, PNG, BMP, TIFF
- **Report Generation**: 1-2 seconds

---

## � Data Storage

This application uses **file-based storage** - no database required:

- **MRI Images**: Stored in `static/uploads/` folder
- **Visualizations**: Grad-CAM heatmaps saved in `static/gradcam/`
- **Reports**: PDF files generated in `static/reports/`
- **All files**: Organized with timestamps for easy tracking

**Advantages:**
✓ No database installation needed  
✓ Simple folder-based organization  
✓ Easy to backup and migrate  
✓ Lightweight and fast  

---

## �🔐 Security & Privacy

✅ **Local Processing** - All images processed on your machine  
✅ **No Cloud Upload** - Data never leaves your system  
✅ **Patient Privacy** - Patient info stored locally  
✅ **Professional Disclaimers** - Clear medical limitations  

⚠️ **Important**: This tool is for **research and clinical support only**. Always consult qualified medical professionals for patient diagnosis.

---

## 🐛 Troubleshooting

### Problem: Model not loading
```bash
# Check model file exists
ls brain_tumor_densenet_final.h5
```

### Problem: Port 5000 already in use
```bash
# Use different port
# Edit app.py: app.run(port=5001)
```

### Problem: Out of memory
- Close other applications
- Restart the application
- Increase system RAM

### Problem: Image upload fails
- Check file format (JPG, PNG, BMP, TIFF)
- Verify file size < 10MB
- Ensure uploads folder has write permissions

---

## 📊 Example Workflow

```
1. User fills patient form (Name: John Doe, Age: 45)
   ↓
2. Uploads MRI image (scan.jpg)
   ↓
3. AI analyzes image with DenseNet model
   ↓
4. Generates Grad-CAM visualization
   ↓
5. Creates professional PDF report
   ↓
6. User downloads report and shares with medical team
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📝 Requirements

```
flask
tensorflow
opencv-python
numpy
matplotlib
reportlab
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚖️ Legal Disclaimer

```
⚠️ IMPORTANT: This application is for RESEARCH and EDUCATIONAL purposes.

- NOT a replacement for professional medical diagnosis
- Results must be reviewed by qualified radiologists
- AI predictions are supplementary only
- Always consult medical professionals for clinical decisions

For medical deployment, ensure compliance with:
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- Local medical data protection laws
```

---

## 📞 Support

Need help? Check the application's **About** and **Contact** pages within the app.

---

## 📄 License

This project is licensed under the MIT License - free for educational and research use.

---

## 🙏 Acknowledgments

- TensorFlow & Keras teams
- OpenCV community
- Flask web framework developers
- Medical imaging research community

---


