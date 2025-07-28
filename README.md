<p align="center">
  <img src="https://img.shields.io/badge/Model%20Size-20MB-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Runtime-Offline%20%7C%20CPU--Only-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Execution%20Time-~06s-yellow?style=for-the-badge"/>
 <img src="https://img.shields.io/badge/Tech%20Stack-Python%20%7C%20ONNX%20%7C%20YOLOv9--Tiny-ff69b4?style=for-the-badge"/>

</p>

#  Intelligent Offline PDF Structuring using ONNX-YOLOv9 & Heuristic Parsing

---

##  Abstract

This project presents an efficient, offline-capable, AI-driven pipeline for converting unstructured PDFs into structured, semantically rich JSON representations. We address the challenge of document understanding by fusing **visual layout detection** and **linguistic parsing** — optimized for constrained, CPU-only environments.

A compact ONNX-converted **YOLOv9-Tiny object detection model**, trained on heading-level detection (H1–H6), is employed for layout segmentation. Pages are rendered into images and passed through the model to identify structural zones without relying on unreliable metadata tags. Detected headings are then algorithmically aligned with text spans using **PyMuPDF** and **pdfplumber**, ensuring high-fidelity reconstruction of document outlines. The final output conforms to a custom-defined JSON schema suitable for downstream knowledge extraction or navigation applications.

The system is designed to operate **entirely offline**, supports **Docker-based deployment**, and processes documents rapidly with minimal resource footprint.

---

##  Flowchart  
![Image](https://github.com/user-attachments/assets/906885a1-4a84-4798-8f40-94e71b52cbf3)

---

##  Folder Structure

```
adobe_hackathon/
├── sample_dataset/
│   ├── outputs/                  # Stores processed output (e.g., JSONs, visualizations)
│   ├── pdfs/                     # Contains input PDF documents to be analyzed
│   └── schema/                   # Schema definitions or templates used for structuring outputs
├── Dockerfile                    # Dockerfile to containerize the app
├── doclaynet.yaml                # YOLOv5/YOLOv8 config file for DocLayNet model
├── process_pdfs.py               # Main script to process PDF files using the trained model
├── requirements.txt              # Python dependencies
├── yolo-doclaynet.onnx           # YOLOv5 ONNX model for inference
└── yolo-doclaynet.pt             # YOLOv5 PyTorch model file
```



## ⚙️ How to Run

### 🐳 Using Docker (Offline Execution)

> Ensure [Docker](https://docs.docker.com/get-docker/) is installed on your system.

```cmd
docker build --platform linux/amd64 -t pdf-processor .

docker run -v "C:\Users\iamjo\OneDrive\Desktop\final_adobbe_project\Adobe_hackathon\sample_dataset\pdfs:/app/input" ^
           -v "C:\Users\iamjo\OneDrive\Desktop\final_adobbe_project\Adobe_hackathon\sample_dataset\outputs:/app/output" ^
           --network none pdf-processor
```


##  Features

- ✅ Built using **YOLOv9-Tiny**, optimized with ONNX (~42MB) for fast and efficient visual document segmentation  
- ✅ Complete offline execution with no external API calls, running at ~0.5s/page on a standard **8-core CPU**  
- ✅ Robustly handles diverse PDF layouts, including documents with **mixed structures**, **images**, and **tables**  
- ✅ Automatically infers document hierarchy (H1–H6) from visual layout without relying on tags or metadata  
- ✅ Delivers structured, **schema-compliant** JSON outputs that are easy to integrate with downstream systems  
- ✅ Fully Dockerized for consistent, secure, and reproducible deployment across platforms  

---

###  Validation Checklist

The solution has been tested and verified against the following constraints and requirements:

- ✅ All PDFs in the input directory are successfully processed  
- ✅ JSON output is generated for each corresponding PDF  
- ✅ Output adheres to the defined schema (`sample_dataset/schema/output_schema.json`)  
- ✅ Output structure matches required hierarchical format  
- ✅ Processes 50-page PDF documents in under 10 seconds (on 8-core CPU)  
- ✅ Works entirely offline with no internet dependency  
- ✅ Memory usage consistently stays within a 16GB limit  
- ✅ Fully compatible with AMD64 (x86_64) architecture

      
---

##  References

- 📄 [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/pdf/1809.01477)
- 🗂️ [OmniDocBench: Benchmark Dataset for Document AI](https://github.com/opendatalab/OmniDocBench)  
  ↳ Dataset hosted on [HuggingFace Datasets](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- 🧠 [YOLOv9-based Document Layout Detection - yolo-doclaynet](https://github.com/ppaanngggg/yolo-doclaynet)
- 🏭 [MinerU: A Pretraining Framework for Structured Document Intelligence](https://github.com/ope)


##  Attention

- Since me and my teammates were thinking there were different rounds for 1A and 1B hence only the solution provided is 1A
