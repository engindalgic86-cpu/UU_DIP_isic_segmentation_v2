# ==================== ISIC ROI SEGMENTASYON VE ÖZNİTELİK ÇIKARIMI ====================
# Mustafa Engin Dalgıç | 254309502
# Üsküdar Üniversitesi - Bilgisayar Mühendisliği YL
# Email: engindalgic86@gmail.com
#
# PROJE: ISIC 2018 Deri Lezyonu Görüntülerinde ROI Segmentasyonu + Öznitelik Çıkarımı
# =====================================================================================

# ==================== KÜTÜPHANELER ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 100

print("="*80)
print("🔬 ISIC ROI SEGMENTASYON VE ÖZNİTELİK ÇIKARIMI")
print("="*80)
print("\n✅ Tüm kütüphaneler başarıyla yüklendi!")
print(f"✅ OpenCV versiyonu: {cv2.__version__}")
print(f"✅ NumPy versiyonu: {np.__version__}")
print(f"✅ Pandas versiyonu: {pd.__version__}")


# ==================== VERİ SETİ YÜKLEME ====================
# ISIC klasör yolunu buraya yazın:
# Örnek Windows: r"C:\Users\ENGİN\Desktop\ISIC"
# Örnek Mac/Linux: "/home/engin/Desktop/ISIC"
DATA_PATH = "ISIC"  # Aynı klasörde ise böyle bırakın

def load_image_dataset(data_path):
    """
    ISIC klasöründeki tüm görüntüleri tarayıp DataFrame'e yükler
    
    Returns:
        pd.DataFrame: filename, filepath, width, height, class bilgilerini içeren DataFrame
    """
    print(f"\n{'='*80}")
    print("📂 VERİ SETİ YÜKLEME")
    print("="*80)
    
    image_data = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # ISIC klasörünü tara
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                
                # Sınıf bilgisi (klasör adından)
                class_name = os.path.basename(root) if root != data_path else "unknown"
                
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                    
                    image_data.append({
                        'filename': file,
                        'filepath': file_path,
                        'width': width,
                        'height': height,
                        'class': class_name,
                        'resolution': f"{width}x{height}"
                    })
                except Exception as e:
                    print(f"⚠️  Hata ({file}): {e}")
    
    df = pd.DataFrame(image_data)
    
    print(f"\n📊 Veri Seti İstatistikleri:")
    print(f"   Toplam görüntü: {len(df)}")
    print(f"   Sınıf sayısı: {df['class'].nunique()}")
    print(f"\n📋 Sınıf dağılımı:")
    print(df['class'].value_counts())
    
    return df


# ==================== AŞAMA 1: RGB → GRAYSCALE DÖNÜŞÜMÜ ====================
def stage1_rgb_to_grayscale(df, num_samples=9, save_output=True):
    """
    Aşama 1: RGB görüntüleri grayscale'e çevir ve görselleştir
    
    Args:
        df: Görüntü bilgilerini içeren DataFrame
        num_samples: Görselleştirilecek rastgele örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Grayscale görüntüleri içeren dictionary
    """
    print(f"\n{'='*80}")
    print("🎨 AŞAMA 1: RGB → GRAYSCALE DÖNÜŞÜMÜ")
    print("="*80)
    
    # Rastgele örnekler seç
    np.random.seed(42)
    sample_indices = np.random.choice(df.index, size=min(num_samples, len(df)), replace=False)
    samples = df.iloc[sample_indices]
    
    # Görselleştirme için grid oluştur
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*2, figsize=(20, 12))
    fig.suptitle('AŞAMA 1: RGB vs Grayscale Karşılaştırma', fontsize=16, fontweight='bold', y=0.995)
    
    grayscale_images = {}
    
    for idx, (i, row) in enumerate(samples.iterrows()):
        if idx >= rows * cols:
            break
            
        # Görüntüyü yükle
        img = cv2.imread(row['filepath'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grayscale dönüşümü
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Grayscale'i kaydet
        grayscale_images[row['filename']] = {
            'gray': img_gray,
            'rgb': img_rgb,
            'filepath': row['filepath'],
            'class': row['class']
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # RGB görüntü (sol)
        ax_rgb = axes[row_idx, col_idx*2]
        ax_rgb.imshow(img_rgb)
        ax_rgb.set_title(f'RGB\n{row["filename"][:20]}...\nClass: {row["class"]}', 
                         fontsize=9)
        ax_rgb.axis('off')
        
        # Grayscale görüntü (sağ)
        ax_gray = axes[row_idx, col_idx*2 + 1]
        ax_gray.imshow(img_gray, cmap='gray')
        ax_gray.set_title(f'Grayscale\n{row["width"]}x{row["height"]}', 
                          fontsize=9)
        ax_gray.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '01_rgb_vs_grayscale.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Görselleştirme kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # İstatistikler
    print(f"\n📊 Dönüşüm İstatistikleri:")
    print(f"   İşlenen görüntü sayısı: {len(grayscale_images)}")
    print(f"   Ortalama boyut: {df['width'].mean():.0f} x {df['height'].mean():.0f} piksel")
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 1 TAMAMLANDI!")
    print("="*80)
    
    return grayscale_images


# ==================== AŞAMA 2.1: DİNAMİK CROP ====================
def detect_background_threshold(img_gray):
    """
    Histogram analizi ile arka plan eşik değerini belirle
    
    Args:
        img_gray: Grayscale görüntü
        
    Returns:
        threshold: Arka plan için eşik değeri
    """
    # Histogram hesapla
    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=[0, 256])
    
    # En yüksek frekansa sahip piksel değeri (arka plan genelde en çok)
    # Genelde arka plan açık renk (yüksek değer) olduğundan üst yarıyı incele
    upper_half_hist = hist[128:]
    peak_idx = np.argmax(upper_half_hist) + 128
    
    # Eşik değeri: peak'in %80'i (arka plan genelde bu civarda)
    threshold = peak_idx * 0.8
    
    return threshold


def dynamic_crop(img_gray, margin=10):
    """
    Dinamik kırpma: Histogram analizi ile arka plan tespiti
    
    Strateji:
    1. Histogram analizi ile arka plan eşik değerini bul
    2. Arka plan piksellerinin konumlarını tespit et
    3. İlgi alanını (ROI) kapsayan minimum dikdörtgeni bul
    4. Margin ekleyerek kırp
    
    Args:
        img_gray: Grayscale görüntü
        margin: Kırpma sonrası eklenecek boşluk (piksel)
        
    Returns:
        cropped: Kırpılmış görüntü
        crop_coords: Kırpma koordinatları (x1, y1, x2, y2)
    """
    h, w = img_gray.shape
    
    # Arka plan eşiğini belirle
    bg_threshold = detect_background_threshold(img_gray)
    
    # Arka plan olmayan (ilgi alanı) piksellerini bul
    foreground_mask = img_gray < bg_threshold
    
    # İlgi alanının koordinatlarını bul
    rows = np.any(foreground_mask, axis=1)
    cols = np.any(foreground_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Eğer hiç ilgi alanı tespit edilmediyse, orijinal görüntüyü döndür
        return img_gray, (0, 0, w, h)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Margin ekle (ama görüntü sınırlarını aşma)
    y1 = max(0, y1 - margin)
    y2 = min(h, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(w, x2 + margin)
    
    # Kırp
    cropped = img_gray[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2, y2)


def stage2_1_dynamic_crop(grayscale_data, num_samples=9, save_output=True):
    """
    Aşama 2.1: Dinamik kırpma ile kenar gürültülerini temizle
    
    Args:
        grayscale_data: Aşama 1'den gelen grayscale görüntü dict'i
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Kırpılmış görüntüleri içeren dictionary
    """
    print(f"\n{'='*80}")
    print("✂️  AŞAMA 2.1: DİNAMİK CROP (KIRPMA)")
    print("="*80)
    print("\n📋 Strateji: Histogram analizi ile arka plan tespiti")
    print("   - Arka plan genelde açık renk (yüksek piksel değeri)")
    print("   - İlgi alanı (lezyon) daha koyu (düşük piksel değeri)")
    print("   - Sadece ilgi alanını kapsayan bölgeyi koru")
    
    # Rastgele örnekler seç
    sample_keys = list(grayscale_data.keys())[:num_samples]
    
    # Görselleştirme için grid oluştur
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*2, figsize=(20, 12))
    fig.suptitle('AŞAMA 2.1: Dinamik Crop - Öncesi vs Sonrası', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    cropped_data = {}
    crop_stats = {
        'original_sizes': [],
        'cropped_sizes': [],
        'pixels_removed': [],
        'percent_removed': []
    }
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        img_gray = grayscale_data[key]['gray']
        original_h, original_w = img_gray.shape
        
        # Dinamik kırpma uygula
        cropped, (x1, y1, x2, y2) = dynamic_crop(img_gray, margin=10)
        cropped_h, cropped_w = cropped.shape
        
        # İstatistikleri kaydet
        original_pixels = original_h * original_w
        cropped_pixels = cropped_h * cropped_w
        removed_pixels = original_pixels - cropped_pixels
        percent_removed = (removed_pixels / original_pixels) * 100
        
        crop_stats['original_sizes'].append(f"{original_w}x{original_h}")
        crop_stats['cropped_sizes'].append(f"{cropped_w}x{cropped_h}")
        crop_stats['pixels_removed'].append(removed_pixels)
        crop_stats['percent_removed'].append(percent_removed)
        
        # Veriyi kaydet
        cropped_data[key] = {
            'cropped': cropped,
            'original': img_gray,
            'rgb': grayscale_data[key]['rgb'],
            'crop_coords': (x1, y1, x2, y2),
            'class': grayscale_data[key]['class'],
            'stats': {
                'original_size': (original_w, original_h),
                'cropped_size': (cropped_w, cropped_h),
                'removed_pixels': removed_pixels,
                'percent_removed': percent_removed
            }
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Orijinal (sol)
        ax_orig = axes[row_idx, col_idx*2]
        ax_orig.imshow(img_gray, cmap='gray')
        ax_orig.set_title(f'Orijinal\n{original_w}x{original_h}', fontsize=9)
        
        # Kırpma bölgesini çiz
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         linewidth=2, edgecolor='red', facecolor='none')
        ax_orig.add_patch(rect)
        ax_orig.axis('off')
        
        # Kırpılmış (sağ)
        ax_crop = axes[row_idx, col_idx*2 + 1]
        ax_crop.imshow(cropped, cmap='gray')
        ax_crop.set_title(f'Kırpılmış\n{cropped_w}x{cropped_h}\n↓ %{percent_removed:.1f}', 
                          fontsize=9, color='green')
        ax_crop.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '02_dynamic_crop.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Görselleştirme kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # İstatistikler
    print(f"\n📊 Kırpma İstatistikleri ({len(crop_stats['percent_removed'])} örnek):")
    print(f"   Ortalama kırpılan alan: %{np.mean(crop_stats['percent_removed']):.1f}")
    print(f"   Min kırpılan alan: %{np.min(crop_stats['percent_removed']):.1f}")
    print(f"   Max kırpılan alan: %{np.max(crop_stats['percent_removed']):.1f}")
    
    print(f"\n💡 YORUM:")
    avg_removed = np.mean(crop_stats['percent_removed'])
    if avg_removed < 10:
        print(f"   - Kırpma oranı düşük (%{avg_removed:.1f})")
        print(f"   - Görüntülerde zaten az arka plan var")
        print(f"   - Lezyon merkeze yakın, iyi çerçevelenmiş")
    elif avg_removed < 30:
        print(f"   - Kırpma oranı orta (%{avg_removed:.1f})")
        print(f"   - Kenar gürültüleri başarıyla temizlendi")
        print(f"   - İlgi alanı korundu")
    else:
        print(f"   - Kırpma oranı yüksek (%{avg_removed:.1f})")
        print(f"   - Görüntülerde çok arka plan vardı")
        print(f"   - Dinamik kırpma etkili oldu")
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 2.1 TAMAMLANDI!")
    print("="*80)
    
    return cropped_data


# ==================== AŞAMA 2.2: KONTRAST İYİLEŞTİRME ====================
def contrast_stretching(img):
    """
    Kontrast germe (Min-Max normalizasyon)
    
    Args:
        img: Grayscale görüntü
        
    Returns:
        img_stretched: Kontrast gerilmiş görüntü
    """
    img_min = img.min()
    img_max = img.max()
    
    # Eğer görüntü zaten tam aralıkta ise, olduğu gibi döndür
    if img_min == 0 and img_max == 255:
        return img
    
    # Min-Max normalizasyon
    img_stretched = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return img_stretched


def histogram_equalization(img):
    """
    Histogram eşitleme
    
    Args:
        img: Grayscale görüntü
        
    Returns:
        img_equalized: Histogram eşitlenmiş görüntü
    """
    img_equalized = cv2.equalizeHist(img)
    return img_equalized


def stage2_2_contrast_enhancement(cropped_data, num_samples=9, save_output=True):
    """
    Aşama 2.2: Kontrast iyileştirme - Stretching vs Equalization karşılaştırma
    
    Args:
        cropped_data: Aşama 2.1'den gelen kırpılmış görüntü dict'i
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Kontrast iyileştirilmiş görüntüleri içeren dictionary
    """
    print(f"\n{'='*80}")
    print("🎨 AŞAMA 2.2: KONTRAST İYİLEŞTİRME")
    print("="*80)
    print("\n📋 İki yöntem test edilecek:")
    print("   A) Kontrast Germe (Stretching) - Min-Max normalizasyon")
    print("   B) Histogram Eşitleme (Equalization) - Histogram düzleştirme")
    
    # Rastgele örnekler seç
    sample_keys = list(cropped_data.keys())[:num_samples]
    
    # 3 sütunlu görselleştirme: Orijinal, Stretching, Equalization
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*3, figsize=(24, 12))
    fig.suptitle('AŞAMA 2.2: Kontrast İyileştirme Karşılaştırma', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    enhanced_data = {}
    comparison_stats = {
        'original_range': [],
        'stretched_range': [],
        'equalized_range': [],
        'original_std': [],
        'stretched_std': [],
        'equalized_std': []
    }
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        img_cropped = cropped_data[key]['cropped']
        
        # Her iki yöntemi uygula
        img_stretched = contrast_stretching(img_cropped)
        img_equalized = histogram_equalization(img_cropped)
        
        # İstatistikleri kaydet
        comparison_stats['original_range'].append(img_cropped.max() - img_cropped.min())
        comparison_stats['stretched_range'].append(img_stretched.max() - img_stretched.min())
        comparison_stats['equalized_range'].append(img_equalized.max() - img_equalized.min())
        comparison_stats['original_std'].append(img_cropped.std())
        comparison_stats['stretched_std'].append(img_stretched.std())
        comparison_stats['equalized_std'].append(img_equalized.std())
        
        # Veriyi kaydet (her iki yöntem de)
        enhanced_data[key] = {
            'cropped': img_cropped,
            'stretched': img_stretched,
            'equalized': img_equalized,
            'rgb': cropped_data[key]['rgb'],
            'class': cropped_data[key]['class'],
            'stats': {
                'original': {
                    'range': comparison_stats['original_range'][-1],
                    'std': comparison_stats['original_std'][-1],
                    'mean': img_cropped.mean()
                },
                'stretched': {
                    'range': comparison_stats['stretched_range'][-1],
                    'std': comparison_stats['stretched_std'][-1],
                    'mean': img_stretched.mean()
                },
                'equalized': {
                    'range': comparison_stats['equalized_range'][-1],
                    'std': comparison_stats['equalized_std'][-1],
                    'mean': img_equalized.mean()
                }
            }
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Orijinal (sol)
        ax_orig = axes[row_idx, col_idx*3]
        ax_orig.imshow(img_cropped, cmap='gray')
        ax_orig.set_title(f'Orijinal\nRange: {img_cropped.max()-img_cropped.min()}\nStd: {img_cropped.std():.1f}', 
                          fontsize=8)
        ax_orig.axis('off')
        
        # Stretching (orta)
        ax_stretch = axes[row_idx, col_idx*3 + 1]
        ax_stretch.imshow(img_stretched, cmap='gray')
        ax_stretch.set_title(f'Stretching\nRange: {img_stretched.max()-img_stretched.min()}\nStd: {img_stretched.std():.1f}', 
                            fontsize=8, color='blue')
        ax_stretch.axis('off')
        
        # Equalization (sağ)
        ax_equal = axes[row_idx, col_idx*3 + 2]
        ax_equal.imshow(img_equalized, cmap='gray')
        ax_equal.set_title(f'Equalization\nRange: {img_equalized.max()-img_equalized.min()}\nStd: {img_equalized.std():.1f}', 
                          fontsize=8, color='green')
        ax_equal.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '03_contrast_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Karşılaştırma kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # Histogram karşılaştırması (detaylı)
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('AŞAMA 2.2: Histogram Analizi (İlk 9 Örnek)', 
                  fontsize=14, fontweight='bold')
    
    for idx, key in enumerate(sample_keys[:9]):
        if idx >= 9:
            break
        
        img_cropped = cropped_data[key]['cropped']
        img_stretched = enhanced_data[key]['stretched']
        img_equalized = enhanced_data[key]['equalized']
        
        row_idx = idx // 3
        col_idx = idx % 3
        ax = axes2[row_idx, col_idx]
        
        # Histogramları çiz
        ax.hist(img_cropped.flatten(), bins=50, alpha=0.5, label='Orijinal', color='gray')
        ax.hist(img_stretched.flatten(), bins=50, alpha=0.5, label='Stretching', color='blue')
        ax.hist(img_equalized.flatten(), bins=50, alpha=0.5, label='Equalization', color='green')
        
        ax.set_title(f'Örnek {idx+1}', fontsize=9)
        ax.set_xlabel('Piksel Değeri', fontsize=8)
        ax.set_ylabel('Frekans', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_output:
        output_file2 = '03_histogram_analysis.png'
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"✅ Histogram analizi kaydedildi: {output_file2}")
    
    plt.show()
    plt.close()
    
    # İstatistiksel karşılaştırma
    print(f"\n📊 İstatistiksel Karşılaştırma ({len(comparison_stats['original_range'])} örnek):")
    print(f"\n   Piksel Aralığı (Range):")
    print(f"      Orijinal:     Ort: {np.mean(comparison_stats['original_range']):.1f}")
    print(f"      Stretching:   Ort: {np.mean(comparison_stats['stretched_range']):.1f} (+{np.mean(comparison_stats['stretched_range']) - np.mean(comparison_stats['original_range']):.1f})")
    print(f"      Equalization: Ort: {np.mean(comparison_stats['equalized_range']):.1f} (+{np.mean(comparison_stats['equalized_range']) - np.mean(comparison_stats['original_range']):.1f})")
    
    print(f"\n   Standart Sapma (Std - Kontrast göstergesi):")
    print(f"      Orijinal:     Ort: {np.mean(comparison_stats['original_std']):.1f}")
    print(f"      Stretching:   Ort: {np.mean(comparison_stats['stretched_std']):.1f} (+{np.mean(comparison_stats['stretched_std']) - np.mean(comparison_stats['original_std']):.1f})")
    print(f"      Equalization: Ort: {np.mean(comparison_stats['equalized_std']):.1f} (+{np.mean(comparison_stats['equalized_std']) - np.mean(comparison_stats['original_std']):.1f})")
    
    # Yöntem önerisi
    print(f"\n💡 OTOMATİK ANALİZ:")
    avg_original_std = np.mean(comparison_stats['original_std'])
    avg_stretched_std = np.mean(comparison_stats['stretched_std'])
    avg_equalized_std = np.mean(comparison_stats['equalized_std'])
    
    stretch_improvement = avg_stretched_std - avg_original_std
    equal_improvement = avg_equalized_std - avg_original_std
    
    print(f"\n   Kontrast İyileştirme Miktarı:")
    print(f"      Stretching:   +{stretch_improvement:.1f} std")
    print(f"      Equalization: +{equal_improvement:.1f} std")
    
    if stretch_improvement < 5 and equal_improvement < 5:
        print(f"\n   ⚠️  Her iki yöntem de az iyileştirme sağladı")
        print(f"   → Görüntüler zaten iyi kontrasta sahip")
        print(f"   → Orijinal görüntülerle devam edilebilir")
        recommendation = "original"
    elif stretch_improvement > equal_improvement * 1.2:
        print(f"\n   ✅ ÖNERİ: STRETCHING")
        print(f"   → Daha fazla kontrast iyileştirmesi (+{stretch_improvement:.1f})")
        print(f"   → Histogram daha dengeli dağılmış")
        print(f"   → Detay koruması daha iyi")
        recommendation = "stretching"
    elif equal_improvement > stretch_improvement * 1.2:
        print(f"\n   ✅ ÖNERİ: EQUALIZATION")
        print(f"   → Daha fazla kontrast iyileştirmesi (+{equal_improvement:.1f})")
        print(f"   → Lezyon-arka plan ayrımı daha net")
        print(f"   → Segmentasyon için daha uygun")
        recommendation = "equalization"
    else:
        print(f"\n   ⚖️  Her iki yöntem de benzer performans")
        print(f"   → Stretching: +{stretch_improvement:.1f} std")
        print(f"   → Equalization: +{equal_improvement:.1f} std")
        print(f"   → Görsel kontrole göre karar verilmeli")
        recommendation = "equalization"  # Segmentasyon için genelde daha iyi
    
    # Önerilen yöntemi kaydet
    for key in enhanced_data:
        enhanced_data[key]['recommended'] = recommendation
        if recommendation == "stretching":
            enhanced_data[key]['selected'] = enhanced_data[key]['stretched']
        elif recommendation == "equalization":
            enhanced_data[key]['selected'] = enhanced_data[key]['equalized']
        else:  # original
            enhanced_data[key]['selected'] = enhanced_data[key]['cropped']
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 2.2 TAMAMLANDI!")
    print(f"📌 Otomatik öneri: {recommendation.upper()}")
    print("="*80)
    
    return enhanced_data, recommendation


# ==================== AŞAMA 2.3: GÜRÜLTÜ AZALTMA (MEDIAN BLUR) ====================
def stage2_3_noise_reduction(enhanced_data, kernel_sizes=[3, 5, 7], 
                              num_samples=9, save_output=True):
    """
    Aşama 2.3: Median Blur ile gürültü azaltma
    
    Median Blur:
    - Tuz-biber gürültüsünü mükemmel temizler
    - Kenar koruması çok iyi (Gaussian'dan üstün)
    - Non-linear filtre (outlier'lara dayanıklı)
    - Segmentasyon öncesi ideal
    
    Args:
        enhanced_data: Aşama 2.2'den gelen kontrast iyileştirilmiş dict
        kernel_sizes: Test edilecek kernel boyutları (tek sayı olmalı)
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Gürültü azaltılmış görüntüleri içeren dictionary
    """
    print(f"\n{'='*80}")
    print("🔇 AŞAMA 2.3: GÜRÜLTÜ AZALTMA (MEDIAN BLUR)")
    print("="*80)
    print("\n📋 Median Blur Özellikleri:")
    print("   ✅ Tuz-biber gürültüsünü mükemmel temizler")
    print("   ✅ Kenar koruması çok iyi (Gaussian'dan üstün)")
    print("   ✅ Outlier'lara dayanıklı (non-linear)")
    print("   ✅ Segmentasyon için ideal")
    print(f"\n🔧 Test edilecek kernel boyutları: {kernel_sizes}")
    
    # Rastgele örnekler seç
    sample_keys = list(enhanced_data.keys())[:num_samples]
    
    # Görselleştirme: Orijinal + 3 farklı kernel boyutu
    rows = 3
    cols = 3
    n_kernels = len(kernel_sizes)
    fig, axes = plt.subplots(rows, cols*(n_kernels+1), figsize=(6*(n_kernels+1), 12))
    fig.suptitle('AŞAMA 2.3: Median Blur - Kernel Boyutu Karşılaştırma', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    blurred_data = {}
    blur_stats = {
        'original_std': [],
        'blur_std': {k: [] for k in kernel_sizes},
        'edge_preservation': {k: [] for k in kernel_sizes}
    }
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        # Seçilen (kontrast iyileştirilmiş) görüntüyü al
        img_enhanced = enhanced_data[key]['selected']
        
        # Her kernel boyutu için median blur uygula
        blurred_versions = {}
        for ksize in kernel_sizes:
            img_blurred = cv2.medianBlur(img_enhanced, ksize)
            blurred_versions[ksize] = img_blurred
            
            # İstatistikleri kaydet
            blur_stats['blur_std'][ksize].append(img_blurred.std())
            
            # Kenar koruması: Laplacian varyansı (yüksek = daha fazla kenar)
            laplacian_orig = cv2.Laplacian(img_enhanced, cv2.CV_64F).var()
            laplacian_blur = cv2.Laplacian(img_blurred, cv2.CV_64F).var()
            edge_preservation_ratio = laplacian_blur / laplacian_orig if laplacian_orig > 0 else 0
            blur_stats['edge_preservation'][ksize].append(edge_preservation_ratio)
        
        blur_stats['original_std'].append(img_enhanced.std())
        
        # Veriyi kaydet
        blurred_data[key] = {
            'enhanced': img_enhanced,
            'blurred': blurred_versions,
            'rgb': enhanced_data[key]['rgb'],
            'class': enhanced_data[key]['class'],
            'stats': {
                'original_std': blur_stats['original_std'][-1],
                'blur_std': {k: blur_stats['blur_std'][k][-1] for k in kernel_sizes},
                'edge_preservation': {k: blur_stats['edge_preservation'][k][-1] for k in kernel_sizes}
            }
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Orijinal (en sol)
        ax_orig = axes[row_idx, col_idx*(n_kernels+1)]
        ax_orig.imshow(img_enhanced, cmap='gray')
        ax_orig.set_title(f'Kontrast İyileştirilmiş\nStd: {img_enhanced.std():.1f}', 
                          fontsize=8)
        ax_orig.axis('off')
        
        # Her kernel boyutu
        for kidx, ksize in enumerate(kernel_sizes):
            img_blurred = blurred_versions[ksize]
            ax_blur = axes[row_idx, col_idx*(n_kernels+1) + kidx + 1]
            ax_blur.imshow(img_blurred, cmap='gray')
            
            # Kenar koruması yüzdesi
            edge_pres = blur_stats['edge_preservation'][ksize][-1] * 100
            color = 'green' if edge_pres > 85 else 'orange' if edge_pres > 70 else 'red'
            
            ax_blur.set_title(f'Kernel: {ksize}x{ksize}\nStd: {img_blurred.std():.1f}\nEdge: {edge_pres:.0f}%', 
                             fontsize=8, color=color)
            ax_blur.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '04_median_blur_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Karşılaştırma kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # Detaylı istatistikler
    print(f"\n📊 Gürültü Azaltma İstatistikleri ({len(blur_stats['original_std'])} örnek):")
    print(f"\n   Orijinal (Kontrast İyileştirilmiş):")
    print(f"      Ortalama Std: {np.mean(blur_stats['original_std']):.1f}")
    
    for ksize in kernel_sizes:
        avg_std = np.mean(blur_stats['blur_std'][ksize])
        avg_edge = np.mean(blur_stats['edge_preservation'][ksize]) * 100
        std_change = avg_std - np.mean(blur_stats['original_std'])
        
        print(f"\n   Kernel {ksize}x{ksize}:")
        print(f"      Ortalama Std: {avg_std:.1f} ({std_change:+.1f})")
        print(f"      Kenar Koruması: {avg_edge:.1f}%")
    
    # Optimal kernel boyutunu belirle
    print(f"\n💡 OTOMATİK KERNEL SEÇİMİ:")
    
    kernel_scores = {}
    for ksize in kernel_sizes:
        # Skor: Kenar koruması ağırlıklı
        avg_edge = np.mean(blur_stats['edge_preservation'][ksize])
        avg_std = np.mean(blur_stats['blur_std'][ksize])
        
        # Kenar koruması > 0.80 → iyi (ağırlık: 70%)
        # Std düşmesi → gürültü azaldı (ağırlık: 30%)
        edge_score = avg_edge
        noise_reduction_score = 1.0 - (avg_std / np.mean(blur_stats['original_std']))
        
        total_score = (edge_score * 0.7) + (noise_reduction_score * 0.3)
        kernel_scores[ksize] = total_score
    
    best_kernel = max(kernel_scores, key=kernel_scores.get)
    
    print(f"\n   Kernel Skorları:")
    for ksize in kernel_sizes:
        score = kernel_scores[ksize]
        edge_pres = np.mean(blur_stats['edge_preservation'][ksize]) * 100
        marker = "⭐ ÖNERILEN" if ksize == best_kernel else ""
        print(f"      {ksize}x{ksize}: {score:.3f} (Kenar: {edge_pres:.1f}%) {marker}")
    
    print(f"\n   ✅ ÖNERILEN KERNEL: {best_kernel}x{best_kernel}")
    
    # Yorum
    if best_kernel == 3:
        print(f"\n   💬 YORUM:")
        print(f"      - Küçük kernel (3x3) → minimal yumuşatma")
        print(f"      - Kenar koruması mükemmel")
        print(f"      - Hafif gürültüler temizlendi")
        print(f"      - Detay kaybı yok")
    elif best_kernel == 5:
        print(f"\n   💬 YORUM:")
        print(f"      - Orta kernel (5x5) → dengeli yaklaşım")
        print(f"      - İyi kenar koruması")
        print(f"      - Orta seviye gürültü temizliği")
        print(f"      - Segmentasyon için ideal")
    else:  # 7 veya daha büyük
        print(f"\n   💬 YORUM:")
        print(f"      - Büyük kernel ({best_kernel}x{best_kernel}) → güçlü yumuşatma")
        print(f"      - Ağır gürültüler temizlendi")
        print(f"      - Bazı detay kaybı olabilir")
        print(f"      - Çok gürültülü görüntüler için uygun")
    
    # Seçilen kernel ile final versiyonu oluştur
    for key in blurred_data:
        blurred_data[key]['selected_kernel'] = best_kernel
        blurred_data[key]['final'] = blurred_data[key]['blurred'][best_kernel]
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 2.3 TAMAMLANDI!")
    print(f"📌 Seçilen kernel: {best_kernel}x{best_kernel}")
    print("="*80)
    
    return blurred_data, best_kernel


# ==================== AŞAMA 3: THRESHOLDING İLE SEGMENTASYON ====================
def apply_global_threshold(img, threshold_value=127):
    """Global thresholding - Sabit eşik değeri"""
    _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary, threshold_value


def apply_otsu_threshold(img):
    """Otsu thresholding - Otomatik optimal eşik"""
    threshold_value, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary, threshold_value


def apply_adaptive_threshold(img, block_size=11, C=2):
    """Adaptive thresholding - Lokal adaptif eşik"""
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, block_size, C)
    return binary, None  # Adaptive'de tek bir threshold değeri yok


def stage3_thresholding_segmentation(blurred_data, num_samples=9, save_output=True):
    """
    Aşama 3: Thresholding ile binary segmentasyon
    
    3 yöntem test edilecek:
    1. Global Thresholding (Sabit eşik)
    2. Otsu Thresholding (Otomatik optimal eşik) - Önerilen
    3. Adaptive Thresholding (Lokal adaptif)
    
    Args:
        blurred_data: Aşama 2.3'ten gelen yumuşatılmış dict
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Binary maskeleri içeren dictionary
        str: Önerilen yöntem
    """
    print(f"\n{'='*80}")
    print("🎭 AŞAMA 3: THRESHOLDING İLE SEGMENTASYON")
    print("="*80)
    print("\n📋 3 Yöntem Test Edilecek:")
    print("   1️⃣  Global Thresholding - Sabit eşik değeri (örn. 127)")
    print("   2️⃣  Otsu Thresholding - Otomatik optimal eşik ⭐ ÖNERİLEN")
    print("   3️⃣  Adaptive Thresholding - Lokal adaptif eşik")
    print("\n🎯 Hedef: Lezyon (beyaz) vs Arka plan (siyah) ayrımı")
    
    # Rastgele örnekler seç
    sample_keys = list(blurred_data.keys())[:num_samples]
    
    # 4 sütunlu görselleştirme: Orijinal + 3 yöntem
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*4, figsize=(24, 12))
    fig.suptitle('AŞAMA 3.1-3.2: Thresholding Yöntemleri Karşılaştırma', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    segmented_data = {}
    threshold_stats = {
        'global_threshold': 127,  # Sabit
        'otsu_thresholds': [],
        'roi_pixels_global': [],
        'roi_pixels_otsu': [],
        'roi_pixels_adaptive': []
    }
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        img_blurred = blurred_data[key]['final']
        
        # 3 yöntemi uygula
        binary_global, threshold_global = apply_global_threshold(img_blurred, threshold_value=127)
        binary_otsu, threshold_otsu = apply_otsu_threshold(img_blurred)
        binary_adaptive, _ = apply_adaptive_threshold(img_blurred, block_size=11, C=2)
        
        # İstatistikleri kaydet
        threshold_stats['otsu_thresholds'].append(threshold_otsu)
        
        # ROI piksel sayısı (beyaz pikseller)
        roi_global = np.sum(binary_global == 255)
        roi_otsu = np.sum(binary_otsu == 255)
        roi_adaptive = np.sum(binary_adaptive == 255)
        
        threshold_stats['roi_pixels_global'].append(roi_global)
        threshold_stats['roi_pixels_otsu'].append(roi_otsu)
        threshold_stats['roi_pixels_adaptive'].append(roi_adaptive)
        
        # Veriyi kaydet
        segmented_data[key] = {
            'blurred': img_blurred,
            'binary_global': binary_global,
            'binary_otsu': binary_otsu,
            'binary_adaptive': binary_adaptive,
            'threshold_global': threshold_global,
            'threshold_otsu': threshold_otsu,
            'rgb': blurred_data[key]['rgb'],
            'class': blurred_data[key]['class'],
            'stats': {
                'roi_pixels_global': roi_global,
                'roi_pixels_otsu': roi_otsu,
                'roi_pixels_adaptive': roi_adaptive
            }
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Orijinal yumuşatılmış (en sol)
        ax_orig = axes[row_idx, col_idx*4]
        ax_orig.imshow(img_blurred, cmap='gray')
        ax_orig.set_title(f'Blur Sonrası', fontsize=8)
        ax_orig.axis('off')
        
        # Global Threshold
        ax_global = axes[row_idx, col_idx*4 + 1]
        ax_global.imshow(binary_global, cmap='gray')
        ax_global.set_title(f'Global (T={threshold_global})\nROI: {roi_global} px', 
                           fontsize=8, color='blue')
        ax_global.axis('off')
        
        # Otsu Threshold
        ax_otsu = axes[row_idx, col_idx*4 + 2]
        ax_otsu.imshow(binary_otsu, cmap='gray')
        ax_otsu.set_title(f'Otsu (T={threshold_otsu:.0f})\nROI: {roi_otsu} px', 
                         fontsize=8, color='green')
        ax_otsu.axis('off')
        
        # Adaptive Threshold
        ax_adaptive = axes[row_idx, col_idx*4 + 3]
        ax_adaptive.imshow(binary_adaptive, cmap='gray')
        ax_adaptive.set_title(f'Adaptive\nROI: {roi_adaptive} px', 
                             fontsize=8, color='orange')
        ax_adaptive.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '05_threshold_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Karşılaştırma kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # Eşik değerleri analizi
    print(f"\n📊 Eşik Değeri Analizi ({len(threshold_stats['otsu_thresholds'])} örnek):")
    print(f"\n   Global Thresholding:")
    print(f"      Eşik değeri: {threshold_stats['global_threshold']} (sabit)")
    print(f"      Ortalama ROI: {np.mean(threshold_stats['roi_pixels_global']):.0f} piksel")
    
    print(f"\n   Otsu Thresholding:")
    print(f"      Ortalama eşik: {np.mean(threshold_stats['otsu_thresholds']):.1f}")
    print(f"      Min eşik: {np.min(threshold_stats['otsu_thresholds']):.0f}")
    print(f"      Max eşik: {np.max(threshold_stats['otsu_thresholds']):.0f}")
    print(f"      Ortalama ROI: {np.mean(threshold_stats['roi_pixels_otsu']):.0f} piksel")
    
    print(f"\n   Adaptive Thresholding:")
    print(f"      Eşik değeri: Lokal (görüntü bölgelerine göre değişir)")
    print(f"      Ortalama ROI: {np.mean(threshold_stats['roi_pixels_adaptive']):.0f} piksel")
    
    # Yöntem önerisi
    print(f"\n💡 OTOMATİK YÖNTEM SEÇİMİ:")
    
    # ROI boyutları karşılaştır
    avg_roi_global = np.mean(threshold_stats['roi_pixels_global'])
    avg_roi_otsu = np.mean(threshold_stats['roi_pixels_otsu'])
    avg_roi_adaptive = np.mean(threshold_stats['roi_pixels_adaptive'])
    
    # ROI tutarlılığı (std deviation düşük = tutarlı)
    std_roi_global = np.std(threshold_stats['roi_pixels_global'])
    std_roi_otsu = np.std(threshold_stats['roi_pixels_otsu'])
    std_roi_adaptive = np.std(threshold_stats['roi_pixels_adaptive'])
    
    # Tutarlılık skoru (düşük std = yüksek tutarlılık)
    consistency_global = 1.0 / (1.0 + std_roi_global / avg_roi_global)
    consistency_otsu = 1.0 / (1.0 + std_roi_otsu / avg_roi_otsu)
    consistency_adaptive = 1.0 / (1.0 + std_roi_adaptive / avg_roi_adaptive)
    
    print(f"\n   ROI Tutarlılık Skoru (yüksek = iyi):")
    print(f"      Global:   {consistency_global:.3f}")
    print(f"      Otsu:     {consistency_otsu:.3f}")
    print(f"      Adaptive: {consistency_adaptive:.3f}")
    
    # Otsu genelde en iyi (equalization sonrası)
    scores = {
        'global': consistency_global * 0.8,  # Sabit eşik pek uygun değil
        'otsu': consistency_otsu * 1.0,      # Otsu ideal
        'adaptive': consistency_adaptive * 0.9  # Adaptive bazen aşırı hassas
    }
    
    best_method = max(scores, key=scores.get)
    
    print(f"\n   Toplam Skorlar:")
    for method, score in scores.items():
        marker = "⭐ ÖNERİLEN" if method == best_method else ""
        print(f"      {method.upper()}: {score:.3f} {marker}")
    
    print(f"\n   ✅ ÖNERILEN YÖNTEM: {best_method.upper()}")
    
    # Yorum
    if best_method == 'otsu':
        print(f"\n   💬 YORUM:")
        print(f"      - Otsu thresholding histogram analizi ile optimal eşik bulur")
        print(f"      - Equalization sonrası bimodal histogram → Otsu için ideal")
        print(f"      - Lezyon-arka plan ayrımı net")
        print(f"      - Tutarlı sonuçlar veriyor")
    elif best_method == 'global':
        print(f"\n   💬 YORUM:")
        print(f"      - Sabit eşik (127) tüm görüntüler için çalışmış")
        print(f"      - Basit ve hızlı")
        print(f"      - Ama optimal olmayabilir")
    else:  # adaptive
        print(f"\n   💬 YORUM:")
        print(f"      - Adaptive thresholding lokal varyasyonları yakalamış")
        print(f"      - İç detaylar korunmuş")
        print(f"      - Ama bazı gürültülü bölgeler oluşabilir")
    
    # Seçilen yöntemi kaydet
    for key in segmented_data:
        segmented_data[key]['recommended_method'] = best_method
        if best_method == 'global':
            segmented_data[key]['selected_binary'] = segmented_data[key]['binary_global']
        elif best_method == 'otsu':
            segmented_data[key]['selected_binary'] = segmented_data[key]['binary_otsu']
        else:  # adaptive
            segmented_data[key]['selected_binary'] = segmented_data[key]['binary_adaptive']
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 3 (THRESHOLDING) TAMAMLANDI!")
    print(f"📌 Önerilen yöntem: {best_method.upper()}")
    print("="*80)
    
    return segmented_data, best_method


# ==================== AŞAMA 4.1: MORFOLOJİK OPERATÖRLER ====================
def stage4_1_morphological_operations(segmented_data, num_samples=9, save_output=True):
    """
    Aşama 4.1: Morfolojik operatörler ile binary maske temizleme
    
    İşlemler:
    1. Opening (Erosion + Dilation) - Küçük gürültüleri temizle
    2. Closing (Dilation + Erosion) - Delikleri doldur
    
    Kernel: Ellipse (lezyonlar yuvarlak)
    
    Args:
        segmented_data: Aşama 3'ten gelen binary maskeli dict
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Morfoloji uygulanmış maskeleri içeren dictionary
        tuple: Seçilen kernel (şekil, boyut)
    """
    print(f"\n{'='*80}")
    print("🔬 AŞAMA 4.1: MORFOLOJİK OPERATÖRLER")
    print("="*80)
    print("\n📋 Uygulanacak İşlemler:")
    print("   1️⃣  Opening (Erosion + Dilation)")
    print("      → Küçük beyaz gürültüleri temizler")
    print("      → Lezyon dışındaki noktaları siler")
    print("   2️⃣  Closing (Dilation + Erosion)")
    print("      → Lezyon içindeki delikleri doldurur")
    print("      → Lezyon bütünlüğünü sağlar")
    print("\n🔧 Kernel: ELLIPSE (lezyonlar yuvarlak/oval)")
    print("   Test edilecek boyutlar: 5x5, 7x7")
    
    # Rastgele örnekler seç
    sample_keys = list(segmented_data.keys())[:num_samples]
    
    # Kernel boyutlarını test et
    kernel_sizes = [5, 7]
    
    # Görselleştirme için hazırlık
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*3, figsize=(18, 12))
    fig.suptitle('AŞAMA 4.1: Morfolojik Operatörler (Opening + Closing)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    morphed_data = {}
    morph_stats = {
        'original_components': [],
        'morphed_components_5': [],
        'morphed_components_7': []
    }
    
    # Önce en iyi kernel boyutunu belirlemek için istatistik topla
    for key in sample_keys[:9]:
        binary_mask = segmented_data[key]['selected_binary']
        
        # Orijinal bileşen sayısı
        num_labels_orig, _ = cv2.connectedComponents(binary_mask)
        morph_stats['original_components'].append(num_labels_orig - 1)  # -1: arka plan hariç
        
        # Her kernel boyutu için test
        for ksize in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            
            # Opening + Closing
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # Bileşen sayısı
            num_labels, _ = cv2.connectedComponents(closed)
            if ksize == 5:
                morph_stats['morphed_components_5'].append(num_labels - 1)
            else:
                morph_stats['morphed_components_7'].append(num_labels - 1)
    
    # En iyi kernel boyutunu seç
    # Hedef: Bileşen sayısını 1'e yaklaştırmak
    avg_comp_5 = np.mean(morph_stats['morphed_components_5'])
    avg_comp_7 = np.mean(morph_stats['morphed_components_7'])
    
    # 1'e yakınlık skoru
    score_5 = 1.0 / (1.0 + abs(avg_comp_5 - 1.0))
    score_7 = 1.0 / (1.0 + abs(avg_comp_7 - 1.0))
    
    best_kernel_size = 5 if score_5 >= score_7 else 7
    
    print(f"\n📊 Kernel Boyutu Seçimi:")
    print(f"   Kernel 5x5: Ortalama {avg_comp_5:.1f} bileşen (Skor: {score_5:.3f})")
    print(f"   Kernel 7x7: Ortalama {avg_comp_7:.1f} bileşen (Skor: {score_7:.3f})")
    print(f"   ✅ Seçilen: {best_kernel_size}x{best_kernel_size}")
    
    # Seçilen kernel ile tüm görüntüleri işle ve görselleştir
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (best_kernel_size, best_kernel_size))
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        binary_mask = segmented_data[key]['selected_binary']
        
        # Opening + Closing uygula
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Bileşen sayılarını hesapla
        num_orig, _ = cv2.connectedComponents(binary_mask)
        num_final, _ = cv2.connectedComponents(closed)
        
        # Veriyi kaydet
        morphed_data[key] = {
            'original_binary': binary_mask,
            'opened': opened,
            'final_morphed': closed,
            'rgb': segmented_data[key]['rgb'],
            'class': segmented_data[key]['class'],
            'kernel_size': best_kernel_size,
            'stats': {
                'components_before': num_orig - 1,
                'components_after': num_final - 1
            }
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Orijinal binary (sol)
        ax_orig = axes[row_idx, col_idx*3]
        ax_orig.imshow(binary_mask, cmap='gray')
        ax_orig.set_title(f'Binary\n{num_orig-1} bileşen', fontsize=8)
        ax_orig.axis('off')
        
        # Opening sonrası (orta)
        ax_open = axes[row_idx, col_idx*3 + 1]
        ax_open.imshow(opened, cmap='gray')
        num_open, _ = cv2.connectedComponents(opened)
        ax_open.set_title(f'Opening\n{num_open-1} bileşen', fontsize=8, color='blue')
        ax_open.axis('off')
        
        # Closing sonrası (sağ)
        ax_close = axes[row_idx, col_idx*3 + 2]
        ax_close.imshow(closed, cmap='gray')
        color = 'green' if (num_final-1) <= 1 else 'orange' if (num_final-1) <= 3 else 'red'
        ax_close.set_title(f'Opening+Closing\n{num_final-1} bileşen', fontsize=8, color=color)
        ax_close.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '06_morphology.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Görselleştirme kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # İstatistikler
    print(f"\n📊 Morfoloji İstatistikleri ({len(morphed_data)} örnek):")
    
    components_before = [d['stats']['components_before'] for d in morphed_data.values()]
    components_after = [d['stats']['components_after'] for d in morphed_data.values()]
    
    print(f"\n   Bileşen Sayısı Değişimi:")
    print(f"      Öncesi: Ortalama {np.mean(components_before):.1f} bileşen")
    print(f"      Sonrası: Ortalama {np.mean(components_after):.1f} bileşen")
    
    # 1 bileşenli görüntü sayısı
    single_component = sum(1 for c in components_after if c == 1)
    print(f"\n   Tek Bileşen (İdeal):")
    print(f"      {single_component}/{len(components_after)} görüntü (%{single_component/len(components_after)*100:.1f})")
    
    # Yorum
    print(f"\n💡 YORUM:")
    if np.mean(components_after) <= 1.5:
        print(f"      ✅ Mükemmel! Çoğu görüntü tek bileşene indirildi")
        print(f"      → Opening gürültüleri başarıyla temizledi")
        print(f"      → Closing delikleri doldurdu")
    elif np.mean(components_after) <= 3:
        print(f"      ⚠️  İyi ama bazı görüntülerde hala çoklu bileşen var")
        print(f"      → CCL ile en büyük bileşeni seçeceğiz")
    else:
        print(f"      ❌ Çok fazla bileşen kaldı")
        print(f"      → Daha büyük kernel veya farklı strateji gerekli")
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 4.1 TAMAMLANDI!")
    print(f"📌 Kullanılan kernel: ELLIPSE {best_kernel_size}x{best_kernel_size}")
    print("="*80)
    
    return morphed_data, ('ellipse', best_kernel_size)


# ==================== AŞAMA 4.2: CONNECTED COMPONENT LABELING (CCL) ====================
def stage4_2_connected_component_labeling(morphed_data, num_samples=9, save_output=True):
    """
    Aşama 4.2: Connected Component Labeling ve final ROI seçimi
    
    Strateji:
    1. Her maskede bağımsız bileşenleri bul (CCL)
    2. Bileşen sayısını analiz et
    3. Eğer birden fazla bileşen varsa → En büyüğünü seç
    4. Final ROI maskesi oluştur
    
    Args:
        morphed_data: Aşama 4.1'den gelen morfoloji uygulanmış dict
        num_samples: Görselleştirilecek örnek sayısı
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        dict: Final ROI maskeleri içeren dictionary
    """
    print(f"\n{'='*80}")
    print("🔢 AŞAMA 4.2: CONNECTED COMPONENT LABELING (CCL)")
    print("="*80)
    print("\n📋 İşlem Adımları:")
    print("   1️⃣  Her maskede bağımsız bileşenleri tespit et")
    print("   2️⃣  Bileşen sayısını analiz et")
    print("   3️⃣  Strateji: En büyük bileşeni seç (ana lezyon)")
    print("   4️⃣  Final ROI maskesi oluştur")
    
    # Rastgele örnekler seç
    sample_keys = list(morphed_data.keys())[:num_samples]
    
    # Görselleştirme: Orijinal, CCL renkli, Final ROI
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols*3, figsize=(18, 12))
    fig.suptitle('AŞAMA 4.2: Connected Component Labeling - Final ROI Seçimi', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    final_data = {}
    ccl_stats = {
        'num_components': [],
        'selected_areas': [],
        'removed_components': []
    }
    
    for idx, key in enumerate(sample_keys):
        if idx >= rows * cols:
            break
        
        morphed_mask = morphed_data[key]['final_morphed']
        
        # Connected Components uygula
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed_mask, connectivity=8)
        
        # Arka plan (label 0) hariç
        num_components = num_labels - 1
        ccl_stats['num_components'].append(num_components)
        
        # En büyük bileşeni seç (arka plan hariç)
        if num_components > 0:
            # Arka plan (0) hariç alanları al
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1  # +1: arka plan offset
            
            # Final ROI maskesi: Sadece en büyük bileşen
            final_roi = (labels == largest_idx).astype(np.uint8) * 255
            
            selected_area = areas[largest_idx - 1]
            removed_count = num_components - 1
            
            ccl_stats['selected_areas'].append(selected_area)
            ccl_stats['removed_components'].append(removed_count)
        else:
            # Hiç bileşen yoksa (boş maske)
            final_roi = np.zeros_like(morphed_mask)
            selected_area = 0
            removed_count = 0
            ccl_stats['selected_areas'].append(0)
            ccl_stats['removed_components'].append(0)
        
        # Veriyi kaydet
        final_data[key] = {
            'morphed_mask': morphed_mask,
            'labels': labels,
            'num_components': num_components,
            'final_roi': final_roi,
            'selected_area': selected_area,
            'removed_components': removed_count,
            'rgb': morphed_data[key]['rgb'],
            'class': morphed_data[key]['class']
        }
        
        # Görselleştirme
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Morfoloji sonrası (sol)
        ax_morph = axes[row_idx, col_idx*3]
        ax_morph.imshow(morphed_mask, cmap='gray')
        ax_morph.set_title(f'Morfoloji\n{num_components} bileşen', fontsize=8)
        ax_morph.axis('off')
        
        # CCL renkli (orta)
        ax_ccl = axes[row_idx, col_idx*3 + 1]
        # Renkli label görüntüsü oluştur
        label_hue = np.uint8(179 * labels / np.max(labels)) if np.max(labels) > 0 else np.zeros_like(labels, dtype=np.uint8)
        label_hue[labels == 0] = 0  # Arka plan siyah
        
        # HSV'ye çevir (renklendirme için)
        blank_channel = np.ones_like(label_hue) * 255
        label_img = cv2.merge([label_hue, blank_channel, blank_channel])
        label_img_rgb = cv2.cvtColor(label_img, cv2.COLOR_HSV2RGB)
        label_img_rgb[labels == 0] = 0  # Arka plan siyah
        
        ax_ccl.imshow(label_img_rgb)
        color = 'green' if num_components == 1 else 'orange' if num_components <= 3 else 'red'
        ax_ccl.set_title(f'CCL Renkli\n{num_components} bileşen', fontsize=8, color=color)
        ax_ccl.axis('off')
        
        # Final ROI (sağ)
        ax_roi = axes[row_idx, col_idx*3 + 2]
        ax_roi.imshow(final_roi, cmap='gray')
        ax_roi.set_title(f'Final ROI\nAlan: {selected_area} px\n↓ {removed_count} bileşen', 
                        fontsize=8, color='green')
        ax_roi.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = '07_ccl_final_roi.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Görselleştirme kaydedildi: {output_file}")
    
    plt.show()
    plt.close()
    
    # CCL istatistikleri
    print(f"\n📊 CCL İstatistikleri ({len(ccl_stats['num_components'])} örnek):")
    
    components_counts = np.array(ccl_stats['num_components'])
    
    print(f"\n   Bileşen Sayısı Dağılımı:")
    for n in range(1, max(components_counts) + 1 if len(components_counts) > 0 else 1):
        count = np.sum(components_counts == n)
        if count > 0:
            marker = "✅" if n == 1 else "⚠️" if n <= 3 else "❌"
            print(f"      {marker} {n} bileşen: {count} görüntü (%{count/len(components_counts)*100:.1f})")
    
    print(f"\n   ROI Alan İstatistikleri:")
    if len(ccl_stats['selected_areas']) > 0:
        print(f"      Ortalama: {np.mean(ccl_stats['selected_areas']):.0f} piksel")
        print(f"      Min: {np.min(ccl_stats['selected_areas']):.0f} piksel")
        print(f"      Max: {np.max(ccl_stats['selected_areas']):.0f} piksel")
    
    print(f"\n   Temizleme Özeti:")
    total_removed = np.sum(ccl_stats['removed_components'])
    print(f"      Toplam {total_removed} küçük bileşen temizlendi")
    
    # Yorum
    print(f"\n💡 YORUM:")
    single_roi_count = np.sum(components_counts == 1)
    single_roi_percent = single_roi_count / len(components_counts) * 100 if len(components_counts) > 0 else 0
    
    if single_roi_percent >= 80:
        print(f"      ✅ MÜKEMMEL! %{single_roi_percent:.0f} görüntüde tek ROI")
        print(f"      → Çoğu görüntü zaten temizdi")
        print(f"      → Segmentasyon pipeline başarılı")
    elif single_roi_percent >= 60:
        print(f"      ✅ İYİ! %{single_roi_percent:.0f} görüntüde tek ROI")
        print(f"      → Diğerlerinde en büyük bileşen seçildi")
        print(f"      → Kabul edilebilir sonuç")
    else:
        print(f"      ⚠️  ORTA: Sadece %{single_roi_percent:.0f} görüntüde tek ROI")
        print(f"      → Çok parçalı lezyonlar var")
        print(f"      → En büyük bileşen stratejisi uygulandı")
    
    if total_removed > 0:
        print(f"\n   🧹 Temizlik:")
        print(f"      → {total_removed} küçük bileşen (gürültü/artefakt) kaldırıldı")
        print(f"      → Ana lezyon korundu")
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 4.2 TAMAMLANDI!")
    print(f"📌 Strateji: En büyük bileşen seçimi")
    print("="*80)
    
    return final_data


# ==================== AŞAMA 5: ÖZNİTELİK ÇIKARIMI ====================
from scipy import stats as scipy_stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

def extract_first_order_features(gray_img, roi_mask):
    """
    First-order (İstatistiksel) öznitelikler
    
    Args:
        gray_img: Grayscale görüntü
        roi_mask: Binary ROI maskesi (255 = lezyon)
        
    Returns:
        dict: İstatistiksel öznitelikler
    """
    # ROI içindeki pikseller
    roi_pixels = gray_img[roi_mask == 255]
    
    if len(roi_pixels) == 0:
        return {f'first_order_{k}': 0.0 for k in ['mean', 'std', 'variance', 'min', 'max', 
                                                     'median', 'skewness', 'kurtosis', 'entropy', 'energy']}
    
    features = {
        'first_order_mean': float(np.mean(roi_pixels)),
        'first_order_std': float(np.std(roi_pixels)),
        'first_order_variance': float(np.var(roi_pixels)),
        'first_order_min': float(np.min(roi_pixels)),
        'first_order_max': float(np.max(roi_pixels)),
        'first_order_median': float(np.median(roi_pixels)),
        'first_order_skewness': float(scipy_stats.skew(roi_pixels)),
        'first_order_kurtosis': float(scipy_stats.kurtosis(roi_pixels)),
        'first_order_entropy': float(scipy_stats.entropy(np.histogram(roi_pixels, bins=256)[0] + 1e-10)),
        'first_order_energy': float(np.sum(roi_pixels.astype(np.float64) ** 2))
    }
    
    return features


def extract_shape_features(roi_mask):
    """
    2D Shape (Şekil) öznitelikleri
    
    Args:
        roi_mask: Binary ROI maskesi
        
    Returns:
        dict: Şekil öznitelikleri
    """
    # Contour bul
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {f'shape_{k}': 0.0 for k in ['area', 'perimeter', 'circularity', 'eccentricity', 
                                             'solidity', 'extent', 'major_axis', 'minor_axis', 
                                             'aspect_ratio', 'convex_area', 'equivalent_diameter', 'compactness']}
    
    cnt = contours[0]
    
    # Temel ölçüler
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Circularity
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    
    # Compactness (alternatif circularity)
    compactness = (perimeter ** 2 / area) if area > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(cnt)
    convex_area = cv2.contourArea(hull)
    solidity = (area / convex_area) if convex_area > 0 else 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    extent = (area / (w * h)) if (w * h) > 0 else 0
    
    # Ellipse fitting (major/minor axis, eccentricity)
    if len(cnt) >= 5:  # fitEllipse en az 5 nokta gerektirir
        try:
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (MA, ma), angle = ellipse
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        except:
            major_axis = minor_axis = aspect_ratio = eccentricity = 0
    else:
        major_axis = minor_axis = aspect_ratio = eccentricity = 0
    
    # Equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
    
    features = {
        'shape_area': float(area),
        'shape_perimeter': float(perimeter),
        'shape_circularity': float(circularity),
        'shape_compactness': float(compactness),
        'shape_eccentricity': float(eccentricity),
        'shape_solidity': float(solidity),
        'shape_extent': float(extent),
        'shape_major_axis': float(major_axis),
        'shape_minor_axis': float(minor_axis),
        'shape_aspect_ratio': float(aspect_ratio),
        'shape_convex_area': float(convex_area),
        'shape_equivalent_diameter': float(equivalent_diameter)
    }
    
    return features


def extract_glcm_features(gray_img, roi_mask, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16):
    """
    GLCM (Texture) öznitelikleri
    
    Args:
        gray_img: Grayscale görüntü
        roi_mask: Binary ROI maskesi
        distances: GLCM uzaklıkları
        angles: GLCM açıları
        levels: Gri seviye sayısı (quantization)
        
    Returns:
        dict: GLCM öznitelikleri
    """
    # ROI içindeki bölgeyi kes
    roi_pixels = gray_img[roi_mask == 255]
    
    if len(roi_pixels) < 10:
        return {f'glcm_{k}': 0.0 for k in ['contrast', 'dissimilarity', 'homogeneity', 
                                            'energy', 'correlation', 'ASM']}
    
    # ROI'yi quantize et (256 → levels)
    roi_quantized = (roi_pixels / 256.0 * levels).astype(np.uint8)
    roi_quantized = np.clip(roi_quantized, 0, levels - 1)
    
    # GLCM için 2D görüntü gerekli, ROI'yi yeniden şekillendir
    # Basit bir yaklaşım: ROI'yi kare matrise dönüştür
    side_length = int(np.sqrt(len(roi_quantized))) + 1
    roi_padded = np.zeros((side_length, side_length), dtype=np.uint8)
    roi_padded.flat[:len(roi_quantized)] = roi_quantized
    
    # GLCM hesapla
    try:
        glcm = graycomatrix(roi_padded, distances=distances, angles=angles, 
                           levels=levels, symmetric=True, normed=True)
        
        # GLCM özelliklerini hesapla
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        ASM = graycoprops(glcm, 'ASM').mean()
        
        features = {
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation),
            'glcm_ASM': float(ASM)
        }
    except:
        features = {f'glcm_{k}': 0.0 for k in ['contrast', 'dissimilarity', 'homogeneity', 
                                                 'energy', 'correlation', 'ASM']}
    
    return features


def stage5_feature_extraction(df, final_data_samples, save_output=True):
    """
    Aşama 5: TÜM veri setinden öznitelik çıkarımı
    
    Bu aşama TÜM görüntüleri işleyecek (sadece örnekler değil)
    
    Args:
        df: Görüntü bilgilerini içeren DataFrame (tüm veri seti)
        final_data_samples: Aşama 4.2'den gelen 9 örnek (referans için)
        save_output: Çıktıyı kaydetmek için True
        
    Returns:
        pd.DataFrame: Öznitelik tablosu
    """
    print(f"\n{'='*80}")
    print("📊 AŞAMA 5: ÖZNİTELİK ÇIKARIMI")
    print("="*80)
    print(f"\n⚠️  DİKKAT: Bu aşama TÜM veri setini işleyecek!")
    print(f"   Toplam görüntü: {len(df)}")
    print(f"   Tahmini süre: ~{len(df) * 2 / 60:.1f} dakika")
    print(f"\n📋 Çıkarılacak Öznitelikler:")
    print(f"   1️⃣  First-Order (İstatistiksel): 10 özellik")
    print(f"   2️⃣  2D Shape (Şekil): 12 özellik")
    print(f"   3️⃣  GLCM (Texture): 6 özellik")
    print(f"   📌 TOPLAM: 28 öznitelik + metadata")
    
    # Tüm pipeline'ı her görüntü için çalıştır
    feature_list = []
    errors = []
    
    print(f"\n🔄 İşleme başlıyor...")
    
    for idx, row in df.iterrows():
        try:
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   İşlenen: {idx + 1}/{len(df)} (%{(idx+1)/len(df)*100:.1f})")
            
            # 1. Görüntüyü yükle
            img = cv2.imread(row['filepath'])
            if img is None:
                raise ValueError("Görüntü yüklenemedi")
            
            # 2. RGB → Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # 3. Crop (basit - merkez crop) - BOYUT KONTROLÜ EKLE
            h, w = img_gray.shape
            margin = min(10, h // 20, w // 20)  # Görüntü çok küçükse margin'i azalt
            
            if h > 2 * margin and w > 2 * margin:
                img_cropped = img_gray[margin:h-margin, margin:w-margin]
            else:
                img_cropped = img_gray  # Çok küçükse crop yapma
            
            # 4. Equalization
            img_eq = cv2.equalizeHist(img_cropped)
            
            # 5. Median Blur
            img_blur = cv2.medianBlur(img_eq, 5)
            
            # 6. Otsu Thresholding
            _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 7. Morfoloji
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # 8. CCL - En büyük bileşeni seç
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_idx = np.argmax(areas) + 1
                final_roi = (labels == largest_idx).astype(np.uint8) * 255
            else:
                final_roi = np.zeros_like(closed)
            
            # 9. Öznitelik çıkarımı
            # ÖNEMLİ: img_cropped ve final_roi aynı boyutta!
            first_order_feats = extract_first_order_features(img_cropped, final_roi)
            shape_feats = extract_shape_features(final_roi)
            glcm_feats = extract_glcm_features(img_cropped, final_roi)
            
            # Metadata
            metadata = {
                'image_id': row['filename'],
                'class': row['class'],
                'width': row['width'],
                'height': row['height'],
                'roi_area': shape_feats['shape_area']
            }
            
            # Tüm özellikleri birleştir
            all_features = {**metadata, **first_order_feats, **shape_feats, **glcm_feats}
            feature_list.append(all_features)
            
        except Exception as e:
            errors.append({'filename': row['filename'], 'error': str(e)})
            if len(errors) <= 10:  # İlk 10 hatayı göster
                print(f"\n   ⚠️  Hata ({row['filename']}): {e}")
    
    print(f"\n✅ İşlem tamamlandı!")
    print(f"   Başarılı: {len(feature_list)}/{len(df)}")
    if errors:
        print(f"   ⚠️  Hatalar: {len(errors)}")
        if len(errors) > 10:
            print(f"   (İlk 10 hata gösterildi, toplam {len(errors)} hata)")
    
    # DataFrame oluştur
    features_df = pd.DataFrame(feature_list)
    
    # CSV olarak kaydet
    if save_output and len(features_df) > 0:
        csv_file = 'features.csv'
        features_df.to_csv(csv_file, index=False)
        print(f"\n✅ Öznitelik tablosu kaydedildi: {csv_file}")
        print(f"   Satır sayısı: {len(features_df)}")
        print(f"   Sütun sayısı: {len(features_df.columns)}")
    
    # Özet istatistikler
    if len(features_df) > 0:
        print(f"\n📊 Öznitelik Tablosu Özeti:")
        print(f"   Toplam görüntü: {len(features_df)}")
        print(f"   Toplam öznitelik: {len(features_df.columns) - 5}")  # metadata hariç
        print(f"   Sınıf dağılımı:")
        print(features_df['class'].value_counts())
        
        print(f"\n📋 İlk 5 satır:")
        print(features_df.head())
        
        # İstatistikler
        print(f"\n📈 Öznitelik İstatistikleri (ilk 10 sütun):")
        print(features_df.iloc[:, :10].describe())
    
    print(f"\n{'='*80}")
    print("✅ AŞAMA 5 TAMAMLANDI!")
    print("="*80)
    
    return features_df


# ==================== ANA PROGRAM ====================
if __name__ == "__main__":
    
    # Veri seti kontrolü
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ HATA: '{DATA_PATH}' klasörü bulunamadı!")
        print("\n📁 Çözümler:")
        print("   1. ISIC klasörünü Python dosyasıyla aynı dizine koyun")
        print(f"   2. Veya kodda DATA_PATH değişkenini tam yol olarak güncelleyin")
        print(f"\n💡 Şu an çalışılan dizin: {os.getcwd()}")
        print(f"\n📋 Bu dizindeki klasörler:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"   📁 {item}/")
        exit(1)
    
    # Veri setini yükle
    df = load_image_dataset(DATA_PATH)
    
    if len(df) == 0:
        print("\n❌ HATA: Hiç görüntü bulunamadı!")
        exit(1)
    
    # ==================== AŞAMA 1: RGB → GRAYSCALE ====================
    print("\n" + "="*80)
    print("🚀 AŞAMA 1 BAŞLIYOR...")
    print("="*80)
    
    grayscale_data = stage1_rgb_to_grayscale(df, num_samples=9, save_output=True)
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 1 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(grayscale_data)} görüntü işlendi")
    print(f"✅ Çıktı: 01_rgb_vs_grayscale.png")
    print(f"\n💡 KONTROL: Grayscale dönüşümü başarılı mı?")
    print(f"   - RGB ve Grayscale karşılaştırmasına bakın")
    print(f"   - Lezyon bölgeleri gri tonlamada görünüyor mu?")
    print(f"   - Detay kaybı var mı?")
    
    # ==================== AŞAMA 2.1: DİNAMİK CROP ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 2.1 BAŞLIYOR...")
    print("="*80)
    
    cropped_data = stage2_1_dynamic_crop(grayscale_data, num_samples=9, save_output=True)
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 2.1 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(cropped_data)} görüntü kırpıldı")
    print(f"✅ Çıktı: 02_dynamic_crop.png")
    print(f"\n💡 KONTROL: Dinamik kırpma başarılı mı?")
    print(f"   - Kırmızı çerçeve ilgi alanını doğru mu kapsıyor?")
    print(f"   - Lezyon bölgesi kayboldu mu?")
    print(f"   - Arka plan gürültüleri temizlendi mi?")
    print(f"   - Kırpma oranı mantıklı mı?")
    
    # ==================== AŞAMA 2.2: KONTRAST İYİLEŞTİRME ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 2.2 BAŞLIYOR...")
    print("="*80)
    
    enhanced_data, recommendation = stage2_2_contrast_enhancement(
        cropped_data, num_samples=9, save_output=True
    )
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 2.2 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(enhanced_data)} görüntü işlendi")
    print(f"✅ İki yöntem karşılaştırıldı:")
    print(f"   - Kontrast Germe (Stretching)")
    print(f"   - Histogram Eşitleme (Equalization)")
    print(f"✅ Çıktılar:")
    print(f"   - 03_contrast_comparison.png (yan yana karşılaştırma)")
    print(f"   - 03_histogram_analysis.png (histogram grafikleri)")
    print(f"\n🎯 OTOMATİK ÖNERİ: {recommendation.upper()}")
    print(f"\n💡 KONTROL:")
    print(f"   - Hangi yöntem lezyon-arka plan kontrastını daha iyi artırmış?")
    print(f"   - Detay kaybı var mı?")
    print(f"   - Histogramlara bakın - dağılım nasıl?")
    print(f"   - Otomatik öneriye katılıyor musunuz?")
    
    # ==================== AŞAMA 2.3: GÜRÜLTÜ AZALTMA ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 2.3 BAŞLIYOR...")
    print("="*80)
    
    blurred_data, best_kernel = stage2_3_noise_reduction(
        enhanced_data, 
        kernel_sizes=[3, 5, 7],
        num_samples=9, 
        save_output=True
    )
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 2.3 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(blurred_data)} görüntü işlendi")
    print(f"✅ Median Blur uygulandı")
    print(f"✅ 3 farklı kernel boyutu test edildi: 3x3, 5x5, 7x7")
    print(f"✅ Çıktı: 04_median_blur_comparison.png")
    print(f"\n🎯 ÖNERILEN KERNEL: {best_kernel}x{best_kernel}")
    print(f"\n💡 KONTROL:")
    print(f"   - Gürültüler temizlendi mi?")
    print(f"   - Lezyon kenarları korundu mu?")
    print(f"   - Hangi kernel boyutu en iyi?")
    print(f"   - Aşırı yumuşatma var mı?")
    
    # ==================== AŞAMA 3: THRESHOLDING SEGMENTASYON ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 3 BAŞLIYOR...")
    print("="*80)
    
    segmented_data, best_threshold_method = stage3_thresholding_segmentation(
        blurred_data,
        num_samples=9,
        save_output=True
    )
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 3 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(segmented_data)} görüntü segmente edildi")
    print(f"✅ 3 threshold yöntemi karşılaştırıldı:")
    print(f"   - Global Thresholding (T=127)")
    print(f"   - Otsu Thresholding (otomatik)")
    print(f"   - Adaptive Thresholding (lokal)")
    print(f"✅ Çıktı: 05_threshold_comparison.png")
    print(f"\n🎯 ÖNERILEN YÖNTEM: {best_threshold_method.upper()}")
    print(f"\n💡 KONTROL:")
    print(f"   - Binary maskeler doğru mu?")
    print(f"   - Lezyon beyaz, arka plan siyah mı?")
    print(f"   - Hangi yöntem en temiz maske üretmiş?")
    print(f"   - Lezyon bütünlüğü korunmuş mu (tek parça)?")
    
    # ==================== AŞAMA 4.1: MORFOLOJİK OPERATÖRLER ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 4.1 BAŞLIYOR...")
    print("="*80)
    
    morphed_data, kernel_info = stage4_1_morphological_operations(
        segmented_data,
        num_samples=9,
        save_output=True
    )
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 4.1 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(morphed_data)} görüntü işlendi")
    print(f"✅ Morfolojik operatörler uygulandı:")
    print(f"   - Opening (gürültü temizleme)")
    print(f"   - Closing (delik doldurma)")
    print(f"✅ Kernel: {kernel_info[0].upper()} {kernel_info[1]}x{kernel_info[1]}")
    print(f"✅ Çıktı: 06_morphology.png")
    print(f"\n💡 KONTROL:")
    print(f"   - Küçük gürültüler temizlendi mi?")
    print(f"   - Lezyon içindeki delikler doldu mu?")
    print(f"   - Bileşen sayısı azaldı mı?")
    
    # ==================== AŞAMA 4.2: CONNECTED COMPONENT LABELING ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 4.2 BAŞLIYOR...")
    print("="*80)
    
    final_data = stage4_2_connected_component_labeling(
        morphed_data,
        num_samples=9,
        save_output=True
    )
    
    print(f"\n{'='*80}")
    print("📋 AŞAMA 4.2 SONUÇ ÖZETİ")
    print("="*80)
    print(f"✅ {len(final_data)} görüntü için final ROI oluşturuldu")
    print(f"✅ Connected Component Labeling uygulandı")
    print(f"✅ Strateji: En büyük bileşen seçimi")
    print(f"✅ Çıktılar:")
    print(f"   - 07_ccl_final_roi.png (renkli CCL + final ROI)")
    print(f"\n💡 KONTROL:")
    print(f"   - Her görüntüde tek ROI var mı?")
    print(f"   - En büyük bileşen doğru seçilmiş mi?")
    print(f"   - Final ROI lezyon bütünlüğünü koruyor mu?")
    
    print(f"\n{'='*80}")
    print("🎉 SEGMENTa SYON TAMAMLANDI! (AŞAMA 1-4)")
    print("="*80)
    print("\n✅ Tamamlanan tüm adımlar:")
    print("   AŞAMA 1: RGB → Grayscale")
    print("   AŞAMA 2.1: Dinamik Crop")
    print("   AŞAMA 2.2: Kontrast İyileştirme (Equalization)")
    print("   AŞAMA 2.3: Gürültü Azaltma (Median Blur)")
    print("   AŞAMA 3: Thresholding (Otsu)")
    print("   AŞAMA 4.1: Morfolojik Operatörler")
    print("   AŞAMA 4.2: Connected Component Labeling")
    print("\n📊 Çıktılar:")
    print("   1. 01_rgb_vs_grayscale.png")
    print("   2. 02_dynamic_crop.png")
    print("   3. 03_contrast_comparison.png")
    print("   4. 03_histogram_analysis.png")
    print("   5. 04_median_blur_comparison.png")
    print("   6. 05_threshold_comparison.png")
    print("   7. 06_morphology.png")
    print("   8. 07_ccl_final_roi.png")
    print("\n🎯 Sonraki adım: AŞAMA 5 - ÖZNİTELİK ÇIKARIMI")
    print("   → First-order features (istatistiksel)")
    print("   → 2D Shape features (şekil)")
    print("   → GLCM features (texture)")
    print("   → Feature CSV oluşturma")
    print("\n" + "="*80)
    
    # ==================== AŞAMA 5: ÖZNİTELİK ÇIKARIMI ====================
    print(f"\n{'='*80}")
    print("🚀 AŞAMA 5 BAŞLIYOR (SON AŞAMA)...")
    print("="*80)
    print("\n⚠️  ÖNEMLİ: Bu aşama uzun sürebilir!")
    print("   Tüm veri seti işlenecek...")
    
    features_df = stage5_feature_extraction(
        df, 
        final_data,
        save_output=True
    )
    
    print(f"\n{'='*80}")
    print("🎊 TÜM PROJE TAMAMLANDI!")
    print("="*80)
    print("\n✅ BAŞARILI AŞAMALAR:")
    print("   ✅ AŞAMA 1: RGB → Grayscale")
    print("   ✅ AŞAMA 2: Pre-Processing (Crop, Kontrast, Blur)")
    print("   ✅ AŞAMA 3: Thresholding Segmentasyon")
    print("   ✅ AŞAMA 4: Post-Processing (Morfoloji, CCL)")
    print("   ✅ AŞAMA 5: Öznitelik Çıkarımı")
    print("\n📊 FINAL ÇIKTILAR:")
    print("   📁 Görselleştirmeler:")
    print("      - 01_rgb_vs_grayscale.png")
    print("      - 02_dynamic_crop.png")
    print("      - 03_contrast_comparison.png")
    print("      - 03_histogram_analysis.png")
    print("      - 04_median_blur_comparison.png")
    print("      - 05_threshold_comparison.png")
    print("      - 06_morphology.png")
    print("      - 07_ccl_final_roi.png")
    print("\n   📄 Öznitelik Tablosu:")
    print(f"      - features.csv ({len(features_df)} görüntü, {len(features_df.columns)} sütun)")
    print("\n🎯 ÖZNİTELİK TABLOSU İÇERİĞİ:")
    print(f"   - Metadata: image_id, class, width, height, roi_area")
    print(f"   - First-Order: 10 istatistiksel özellik")
    print(f"   - Shape: 12 şekil özelliği")
    print(f"   - GLCM: 6 texture özelliği")
    print(f"   - TOPLAM: {len(features_df.columns)} sütun")
    print("\n📌 KULLANIM:")
    print("   Bu CSV dosyasını makine öğrenmesi modellerinde kullanabilirsiniz!")
    print("   - Sınıflandırma (classification)")
    print("   - Kümeleme (clustering)")
    print("   - Öznitelik seçimi (feature selection)")
    print("\n" + "="*80)
    print("🎉 PROJE BAŞARIYLA TAMAMLANDI! 🎉")
    print("="*80)
