# 🐍 Snake Game — GA + Neural Network (Python)

Bu repo, **Genetic Algorithm (GA)** ve **Neural Network (YSA / ANN)** ile kendi kendine oynamayı öğrenen klasik **Snake** oyununu içerir.
Görselleştirme **pygame** ile yapılır; yılan çevresini 21 öznitelikli bir vektörle algılar, ağın çıktısına göre **düz / sağ / sol** kararları verir ve her jenerasyonda evrimleşir.

> Varsayılan ağ mimarisi: **\[21, 16, 3]** (giriş, gizli, çıkış).

---

## 🔧 Kurulum (Python 3.10)

Aşağıdaki adımlar, paket sürümleri ve sanal ortam bilgilerine göre hazırlanmıştır.

### 1️⃣ Sanal ortam oluştur

```bash
# Proje klasöründe
python3.10 -m venv venv
```

### 2️⃣ Ortamı aktive et

**Windows**:

```bash
venv\Scripts\activate
```

**macOS / Linux**:

```bash
source venv/bin/activate
```

### 3️⃣ Paketleri kur

```bash
pip install joblib==1.2.0
pip install llvmlite==0.39.1
pip install numba==0.56.4
pip install numpy==1.23.5
pip install pygame==2.1.2
pip install setuptools==63.2.0  # opsiyonel
```

> `pip` ve `setuptools` genellikle otomatik gelir; eşitlemek istersen bu sürümü kullan.

---

## 📂 Proje Yapısı

* `main.py` — Çalıştırma senaryoları (hazır ağla “izlet”, elle oyna, eğit).
* `game.py` — Oyun döngüsü, görünür/görünmez mod, klavye kontrolleri.
* `snake.py` — Yılan durumu, karar verme, fitness ve çizim.
* `map.py` — Harita, gıda yerleştirme, ışın tabanlı tarama (21 girişin üretimi).
* `neural_network.py` — YSA tanımı, ileri besleme, kaydet/yükle.
* `genetic_algorithm.py` — GA akışı; ebeveyn seçimi, çaprazlama, mutasyon, değerlendirme.
* `constants.py` — Sabitler (boyutlar, yönler, görsel dosya yolları).
* `img/` — `wall.png`, `apple.png`, `snake.png` gibi görseller.

---

## ▶️ Hızlı Başlangıç (main.py)


