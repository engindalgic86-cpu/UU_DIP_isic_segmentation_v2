# ISIC 2018 Deri Lezyonu Segmentasyonu ve Öznitelik Çıkarımı

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULLANICI_ADI/isic_segmentation_project/blob/main/notebooks/ISIC_Segmentation_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Kapsamlı görüntü işleme pipeline'ı ile ISIC 2018 veri setinden otomatik lezyon segmentasyonu ve 28 öznitelik çıkarımı**

## 🔬 Genel Bakış

Bu proje, ISIC 2018 deri lezyonu veri seti üzerinde otomatik segmentasyon ve öznitelik çıkarımı için kapsamlı bir pipeline sunar. **2,239 görüntüden %100 başarı oranıyla ROI segmentasyonu** ve 28 adet öznitelik çıkarımı gerçekleştirilmiştir.

### 📊 Ana Sonuçlar

- ✅ **2,239 görüntü** başarıyla işlendi (%100 başarı)
- ✅ **28 öznitelik** (first-order, shape, GLCM)
- ✅ **%89 gürültü azaltma** (morfolojik operatörler)
- ✅ **9 sınıf** (melanoma, nevus, vb.)

## ✨ Pipeline Aşamaları

1. **RGB → Grayscale** dönüşümü
2. **Ön İşleme** (crop, kontrast, blur)
3. **Otsu Thresholding** segmentasyonu
4. **Post-Processing** (morfoloji, CCL)
5. **Öznitelik Çıkarımı** (28 feature)

## 🚀 Hızlı Başlangıç

### Google Colab (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULLANICI_ADI/isic_segmentation_project/blob/main/notebooks/ISIC_Segmentation_Colab.ipynb)

1. Yukarıdaki butona tıklayın
2. Runtime → Run All
3. ~15-30 dakika bekleyin
4. features.csv indirin!

### Lokal Kurulum

\`\`\`bash
git clone https://github.com/KULLANICI_ADI/isic_segmentation_project.git
cd isic_segmentation_project
pip install -r requirements.txt
python src/isic_segmentation_project.py
\`\`\`

## 📦 Veri Seti

**ISIC 2018:** [challenge.isic-archive.com](https://challenge.isic-archive.com/)

- 2,239 dermoskopik görüntü
- 9 sınıf (melanoma, nevus, basal cell carcinoma, vb.)
- 600×450 piksel (çoğunlukla)

## 📊 Sonuçlar

| Metrik | Değer |
|--------|-------|
| Başarı Oranı | %100 |
| Ortalama ROI | 315,982 piksel |
| Gürültü Azaltma | %89 |
| İşleme Süresi | ~2 sn/görüntü |

## 📁 Proje Yapısı

\`\`\`
isic_segmentation_project/
├── src/
│   └── isic_segmentation_project.py    # Ana script
├── notebooks/
│   └── ISIC_Segmentation_Colab.ipynb   # Colab notebook
├── outputs/                             # Çıktılar
│   ├── *.png                            # Görselleştirmeler
│   └── features.csv                     # Öznitelik tablosu
├── requirements.txt
└── README.md
\`\`\`

## 📚 Referanslar

- Codella et al. (2018). ISIC 2018 Challenge. *IEEE ISBI*
- Otsu, N. (1979). Threshold selection. *IEEE Trans.*
- Haralick et al. (1973). Textural features. *IEEE Trans.*

## 📄 Lisans

MIT License

## 📧 İletişim

**Mustafa Engin Dalgıç**  
Üsküdar Üniversitesi, Bilgisayar Mühendisliği

---

⭐ **Beğendiyseniz yıldız vermeyi unutmayın!**
