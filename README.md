# 🌳 CAIRO Wood Image Restoration: Data Acquisition Suite

An automated, high-precision GUI suite designed for the acquisition and synchronization of wood microscopic image datasets. This project supports the **Tualang Image Restoration** initiative at the CAIRO Lab, facilitating the creation of a robust dataset for AI model training (PyTorch).

## 📊 Dataset Status: Milestone 09 (March 12, 2026)

* **Total Records**: 6,800+ image pairs (Clear/Blur).
* 
**Species Diversity**: 31 initial wood species expanded to 35, including Tualang, Merbau, and Giam.


* **Botanical Mapping**: Full integration of Scientific Names (e.g., *Koompassia excelsa*) within the SQLite registry.
* 
**Storage Structure**: Semi-Flat directory logic (`Kayu/Species/BlockID/clear/`) optimized for PyTorch `ImageFolder` loaders.



## 🛠️ Technical Stack

* 
**Language**: Python 3.14.


* 
**GUI Framework**: PyQt6 for a multi-threaded, lag-free interface.


* 
**Computer Vision**: OpenCV 4.13 utilizing the Variance of Laplacian (VOL) for real-time sharpness quantification.


* 
**Database**: SQLite3 for metadata tracking and relational species mapping.


* 
**Hardware**: USB Microscope integrated with an Intel i7-10750H + GTX 1660 Ti environment.



## ✨ Key Features

* 
**Real-time VOL Metrics**: Quantifies image sharpness (High VOL > 1000 = Clear; Single digits = Synthetic Blur) to ensure training data quality.


* 
**Automated Dataset Repair**: Built-in scripts for filename padding (e.g., BAL3 → BAL03) and database-to-disk synchronization.


* 
**Species Management**: Dynamic registry for initials-to-name mapping with a safety lock to prevent folder misrouting.


* 
**Integrity Verification**: Automated "Verify Dataset" tool to cross-reference 6,800+ SQL paths against local physical storage.



## 🚀 Installation & Usage

1. **Clone the repository**:
```bash
git clone https://github.com/YourUsername/RepoName.git

```


2. **Install Dependencies**:
```bash
pip install opencv-python PyQt6 numpy

```


3. **Run Acquisition**:
```bash
python app/main.py

```



## 📝 License & Attribution

Developed as part of the **CAIRO Internship Program (2026)**. All wood samples provided by the CAIRO Lab.

---
