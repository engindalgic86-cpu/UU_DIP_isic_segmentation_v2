# 🚀 GitHub'a Yükleme ve Colab Entegrasyonu Rehberi

Bu rehber, projenizi GitHub'a yüklemek ve Google Colab ile entegre etmek için adım adım talimatlar içerir.

## 📋 İçindekiler

1. [GitHub Repository Oluşturma](#1-github-repository-oluşturma)
2. [Projeyi GitHub'a Yükleme](#2-projeyi-githuba-yükleme)
3. [Google Colab Entegrasyonu](#3-google-colab-entegrasyonu)
4. [Veri Seti Hazırlığı](#4-veri-seti-hazırlığı)
5. [Test ve Doğrulama](#5-test-ve-doğrulama)

---

## 1️⃣ GitHub Repository Oluşturma

### Adım 1.1: GitHub'da Yeni Repository

1. [github.com](https://github.com) adresine gidin
2. Sağ üstte **"+"** → **"New repository"** tıklayın
3. Repository ayarları:
   - **Repository name:** `isic-segmentation-project`
   - **Description:** "ISIC 2018 Deri Lezyonu Segmentasyonu ve Öznitelik Çıkarımı"
   - **Public** seçin (Colab için gerekli)
   - ✅ **Add README.md** işaretini KALDIRIN (zaten var)
   - ✅ **.gitignore** işaretini KALDIRIN (zaten var)
   - ✅ **Choose a license:** MIT
4. **"Create repository"** tıklayın

### Adım 1.2: Repository URL'sini Not Alın

```
https://github.com/KULLANICI_ADI/isic-segmentation-project.git
```

---

## 2️⃣ Projeyi GitHub'a Yükleme

### Seçenek A: GitHub Desktop (Kolay)

1. [GitHub Desktop](https://desktop.github.com/) indirin ve kurun
2. **File → Add Local Repository**
3. Proje klasörünü seçin
4. **Publish repository** tıklayın
5. Bitirdiniz! ✅

### Seçenek B: Komut Satırı (Terminal)

```bash
# Proje klasörüne gidin
cd isic_segmentation_project/

# Git'i başlatın
git init

# Dosyaları ekleyin
git add .

# İlk commit
git commit -m "Initial commit: ISIC 2018 Segmentation Project"

# GitHub repository'nizi bağlayın (URL'nizi güncelleyin)
git remote add origin https://github.com/KULLANICI_ADI/isic-segmentation-project.git

# Main branch'e push edin
git branch -M main
git push -u origin main
```

✅ **Başarılı!** Projeniz artık GitHub'da!

---

## 3️⃣ Google Colab Entegrasyonu

### Adım 3.1: README.md'yi Güncelleyin

1. GitHub'da repository'nizi açın
2. **README.md** dosyasını düzenleyin
3. **KULLANICI_ADI** yazan yerleri kendi kullanıcı adınızla değiştirin:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SIZIN_KULLANICI_ADI/isic-segmentation-project/blob/main/notebooks/ISIC_Segmentation_Colab.ipynb)
```

### Adım 3.2: Colab Notebook'u Güncelleyin

1. `notebooks/ISIC_Segmentation_Colab.ipynb` dosyasını açın
2. **İlk hücredeki** Git clone komutunu güncelleyin:

```python
!git clone https://github.com/SIZIN_KULLANICI_ADI/isic-segmentation-project.git
```

3. Değişiklikleri commit + push edin:

```bash
git add .
git commit -m "Update Colab notebook with correct GitHub URL"
git push
```

### Adım 3.3: Colab'da Test Edin

1. README.md'deki **"Open In Colab"** butonuna tıklayın
2. Veya direkt: `https://colab.research.google.com/github/SIZIN_KULLANICI_ADI/isic-segmentation-project/blob/main/notebooks/ISIC_Segmentation_Colab.ipynb`
3. İlk hücreyi çalıştırın - repository başarıyla klonlanmalı ✅

---

## 4️⃣ Veri Seti Hazırlığı

### Seçenek A: Google Drive'a Yükleme (Önerilen)

1. [ISIC 2018 veri setini](https://challenge.isic-archive.com/data/) indirin
2. Google Drive'ınıza yükleyin:
   ```
   Google Drive/
   └── ISIC_2018/
       ├── ISIC_0000001.jpg
       ├── ISIC_0000002.jpg
       └── ... (2,239 görüntü)
   ```
3. Colab notebook'ta `DRIVE_ISIC_PATH` değişkenini güncelleyin:
   ```python
   DRIVE_ISIC_PATH = '/content/drive/MyDrive/ISIC_2018'
   ```

### Seçenek B: Kaggle API

1. [Kaggle](https://www.kaggle.com/) hesabınızdan API token alın
2. Colab notebook'ta `USE_KAGGLE = True` yapın
3. kaggle.json dosyasını yükleyin
4. Otomatik indirilecek

---

## 5️⃣ Test ve Doğrulama

### Checklist ✅

- [ ] Repository GitHub'da görünüyor
- [ ] README.md düzgün render ediliyor
- [ ] "Open In Colab" butonu çalışıyor
- [ ] Colab notebook açılıyor
- [ ] İlk hücre (git clone) başarılı
- [ ] Veri yolu düzgün ayarlanmış
- [ ] Demo (9 örnek) çalışıyor
- [ ] Görselleştirmeler oluşuyor

### Hata Giderme

**Problem:** "Repository not found"
- **Çözüm:** Repository'nin **Public** olduğundan emin olun

**Problem:** "No such file or directory: ISIC"
- **Çözüm:** `DRIVE_ISIC_PATH` değişkenini kontrol edin

**Problem:** "ModuleNotFoundError"
- **Çözüm:** `requirements.txt` dosyasının yüklendiğinden emin olun

---

## 🎉 Tamamlandı!

Artık projeniz:
- ✅ GitHub'da public olarak paylaşılıyor
- ✅ Google Colab'da tek tıkla çalışıyor
- ✅ Herkes tarafından kullanılabilir

### 📊 Sonraki Adımlar:

1. **README.md'ye banner ekleyin:**
   ```markdown
   ![Banner](https://via.placeholder.com/1200x300?text=ISIC+2018+Segmentation)
   ```

2. **GitHub Topics ekleyin:**
   - Repository → Settings → Topics
   - Ekleyin: `computer-vision`, `image-processing`, `skin-cancer`, `segmentation`, `isic-2018`

3. **GitHub Pages ile dokümantasyon:**
   - Settings → Pages
   - Source: main branch / docs folder

4. **Releases oluşturun:**
   - Releases → Create a new release
   - Tag: v1.0.0
   - features.csv ve görselleştirmeleri ekleyin

---

## 📚 Ek Kaynaklar

- [GitHub Docs](https://docs.github.com/)
- [Colab Docs](https://colab.research.google.com/notebooks/intro.ipynb)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

## 📧 Destek

Sorun mu yaşıyorsunuz? [Issue açın](https://github.com/KULLANICI_ADI/isic-segmentation-project/issues)

---

**Son Güncelleme:** 10 Ocak 2026  
**Yazar:** Mustafa Engin Dalgıç
