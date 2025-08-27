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

### 1️⃣ Hazır ağı **izlet** (test modu)

```python
from game import *
from genetic_algorithm import *

net = NeuralNetwork()
game = Game()

# test
net.load(filename_weights='gen_100_weights.npy', filename_biases='gen_100_biases.npy')
game.start(display=True, neural_net=net)
```

* `display=True` → Oyunu pencerede izlersiniz (klavye kullanmadan).
* `net.load(...)` → Ağırlık/bias dosyalarını yükler.

---

### 2️⃣ **Elle oyna** (klavyeyle)

```python
# play
game = Game()
game.start(playable=True, display=True, speed=10)
```

* Sağ/Sol ok ile yön değiştir.
* ESC ile çıkış.
* `speed` adım hızını kontrol eder.

---

### 3️⃣ **Eğit** (GA)

```python
# train
gen = GeneticAlgorithm(population_size=1000, crossover_method='neuron', mutation_method='weight')
gen.start()
```

* Her jenerasyonda en iyi bireyin ağı **`gen_{N}_weights.npy / gen_{N}_biases.npy`** olarak kaydedilir.
* Değerlendirme **görünmez modda** yapılır (`display=False`).

> Görünür modda eğitim çok yavaş olabilir; yalnızca test için kullanın.

---

## 🧪 Paralel Değerlendirme

`genetic_algorithm.py` içindeki `evaluation(...)` fonksiyonu her ağı birden fazla kez çalıştırır ve ortalama skor hesaplar. **Joblib** ile paralel çalıştırma vardır.

```python
results1 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
# diğer sonuçlar...
for i in range(len(results1)):
    networks[i].score = int(np.mean([results1[i], results2[i], results3[i], results4[i]]))
```

* `n_jobs=num_cores` → CPU çekirdek sayısını otomatik kullanır.
* Tekrar sayısı 4 → rastgeleliği azaltır.
* `display=True` ile paralel çalıştırmayın, başsız mod önerilir.

---

## 💾 Ağı Kaydet / Yükle

```python
self.weights = np.load(filename_weights, allow_pickle=True)
self.biases  = np.load(filename_biases, allow_pickle=True)
```

* NumPy listeleri pickle ile kaydedildiği için `allow_pickle=True` gerekir.
* `save(...)` otomatik veya özel isimle kaydeder.

---

## 🕹️ Oyun Mekaniği

* **Girdi (21)**: 7 doğrultuda (duvar/beden/yemek) tarama.
* **Çıkış (3)**: 0 → düz, 1 → sağ, 2 → sol.
* **Fitness**: `len(body)^2 * age`.
* **Starve**: 500 adımda yemek bulamazsa ölür.
* Klavye (playable mod) → Sağ/Sol ok, ESC ile çık.

---

## ⚙️ GA Parametreleri

```python
GeneticAlgorithm(
    population_size=1000,
    generation_number=100,
    crossover_rate=0.3,
    crossover_method='neuron',
    mutation_rate=0.7,
    mutation_method='weight'
)
```

* population\_size → Birey sayısı
* generation\_number → Jenerasyon sayısı
* crossover / mutation rate → Çocuk ve mutasyon oranları
* crossover\_method → 'weight' veya 'neuron'
* mutation\_method → 'weight'

---

## ❗️ Sık Karşılaşılan Sorunlar

* **pygame penceresi açılmıyor** → display=True ile görünür mod, başsız sunucuda çalışmaz.
* **allow\_pickle hatası** → load(...) fonksiyonundaki allow\_pickle=True kullanın.
* **Yavaş eğitim** → Paralel değerlendirme ve görünmez mod kullanın.

---

## 📜 Lisans

Eğitim ve demo amaçlıdır. Dilediğiniz gibi kullanabilir veya genişletebilirsiniz.
