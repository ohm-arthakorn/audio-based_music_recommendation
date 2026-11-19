<!-- อย่าลืมมาใส่เทคนิคที่มีการใช้งานในการสร้างระบบแนะนำนะ ! -->
# Audio-Based Music Recommendation System 
<div align="right">
              Written by Ohm, Claude and Gemini
</div>

---
เอกสารฉบับนี้สรุปแนวคิด หลักการทางเทคนิค สถาปัตยกรรมระบบ และแผนการพัฒนา (Roadmap) สำหรับโปรเจกต์ระบบแนะนำเพลงโดยวิเคราะห์จากคุณสมบัติทางเสียง (Content-Based Audio Recommendation) ซึ่งผสานทักษะด้าน 

> - **Signal Processing**
> - **Audio Engineering**
> - **Data Science** 
> - **Backend Engineering (Go/Python)** 

เข้าด้วยกัน

---

## 1. แนวคิดหลักของโปรเจกต์ (Project Overview)

ระบบแนะนำเพลงทั่วไปมักอาศัยพฤติกรรมการฟังของผู้ใช้ (Collaborative Filtering) ซึ่งมีข้อจำกัดคือ **Cold Start Problem** (เพลงใหม่ที่ยังไม่มีคนฟังจะไม่ถูกแนะนำ) 

> **note** : ค้นหาข้อมูลเพิ่มเติมเกี่ยวกับเทคโนโลยีที่มีการใช้งานในปัจจุบันว่าจุดเด่น-จุดด้อยคืออะไร 

โปรเจกต์นี้จึงมุ่งเน้นไปที่ **Content-Based Filtering** โดยให้ระบบ "ฟัง" และวิเคราะห์โครงสร้างของคลื่นเสียงโดยตรง เพื่อสกัดเอกลักษณ์ทางดนตรีออกมาเป็นตัวเลข แล้วค้นหาเพลงที่มีลักษณะทางเสียงใกล้เคียงกัน **โดยไม่อิงจากชื่อศิลปินหรือป้ายกำกับแนวเพลง**

---

## 2. หลักการทางเทคนิค (Core Concepts)

### 2.1 การแปลงเสียงเป็นข้อมูล (Audio Feature Extraction - DSP)
สัญญาณเสียงดิบ (Time-Domain) จะถูกซอยแบ่งเป็นช่วงสั้นๆ (Framing) และแปลงเข้าสู่โดเมนความถี่ (Frequency-Domain) ผ่านกระบวนการ **Fast Fourier Transform (FFT)** เพื่อสกัดค่าสำคัญทางดนตรี (Music Information Retrieval - MIR):

* **MFCCs (Mel-Frequency Cepstral Coefficients):** จำลองการได้ยินของหูมนุษย์ (Mel Scale) เพื่อสกัด "เนื้อเสียง" (Timbre) ช่วยให้แยกแยะเครื่องดนตรีและเสียงร้องได้
* **Chroma Features:** วิเคราะห์ความเข้มข้นของระดับเสียงตัวโน้ต 12 ตัว (C ถึง B) เพื่อจับโครงสร้างคอร์ดและสเกลของเพลง
* **Tempo (BPM):** ความเร็วของจังหวะเพลง
* **Spectral Centroid:** จุดศูนย์ถ่วงของสเปกตรัมความถี่ บ่งบอกความ "สว่าง" หรือ "ทุ้ม" ของเสียง
* **Zero-Crossing Rate (ZCR):** อัตราการตัดผ่านจุดศูนย์ของคลื่น ช่วยแยกเสียงเครื่องเคาะ (Percussion) หรือเสียงพูดออกจากเสียงดนตรีที่ลากยาว

### 2.2 การคำนวณความเหมือน (Data Science & Similarity)
เมื่อสกัดค่าทางเสียงได้แล้ว จะนำค่าสถิติ (เช่น `mean` และ `std`) มารวมกันเป็นชุดตัวเลข 1 มิติ เรียกว่า **Feature Vector**:

1. **Feature Scaling:** ปรับมาตรฐานข้อมูลด้วย `StandardScaler` เพื่อไม่ให้ตัวแปรที่มีค่าตัวเลขสูง (เช่น BPM = 120) กลบตัวแปรที่มีค่าน้อย (เช่น ZCR = 0.05)
2. **Dimensionality Reduction:** ใช้เทคนิค **PCA (Principal Component Analysis)** ลดมิติข้อมูลเพื่อเพิ่มความเร็วในการประมวลผลและใช้พล็อตกราฟวิเคราะห์
3. **Similarity Metrics:** ใช้สมการ **Cosine Similarity** วัดมุมระหว่างเวกเตอร์ของเพลง 2 เพลง ($\text{Similarity} \to 1$ หมายถึงมีความคล้ายคลึงกันสูง):
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

---

## 3. สถาปัตยกรรมระบบและเครื่องมือ (Tech Stack & Architecture)

ระบบถูกออกแบบให้แยกส่วนการทำงาน (Decoupled Architecture) เพื่อดึงจุดเด่นของแต่ละภาษามาใช้:

```text
[Audio Files (.wav/.mp3)] 
       │
       ▼ (Batch Processing)
[Python Pipeline: librosa, pandas, scikit-learn]
       │  └─ สกัด Audio Features & ทำ Feature Scaling
       ▼
[Vector Database: PostgreSQL + pgvector / Qdrant]
       │  └─ เก็บ Feature Vectors ของเพลงทั้งหมด
       ▼ (Real-time Query)
[Go Backend API: Gin / Fiber]
       │  └─ รับ ID เพลง & คำนวณ Vector Search ความเร็วสูง
       ▼
[Frontend UI: Streamlit / HTML+JS]
```

* **Data & DSP Engine:** Python (`librosa`, `numpy`, `pandas`, `scikit-learn`)
* **Database:** PostgreSQL พร้อมส่วนขยาย `pgvector` (หรือ Qdrant / SQLite สำหรับช่วงเริ่มต้น)
* **Backend Service:** Go (`Gin` หรือ `Fiber`) สำหรับสร้าง REST API ที่รองรับการค้นหาได้รวดเร็ว
* **Dataset แนะนำสำหรับเริ่มต้น:** GTZAN Dataset (1,000 เพลง / 10 แนวเพลง) หรือ FMA (Free Music Archive)

---

## 4. แผนการเรียนรู้และพัฒนา (3-4 Weeks Roadmap)

### Phase 1: Audio DSP Foundations 
* **เป้าหมาย:** เข้าใจโครงสร้างข้อมูลเสียงและการแปลงสัญญาณเบื้องต้น
* **สิ่งที่ต้องทำ:** ทดลองใช้ `librosa` โหลดไฟล์เสียง, สร้างกราฟ Waveform และ Mel-Spectrogram, และทดลองสกัดค่า MFCCs, Chroma, ZCR ออกมาดูค่าตัวเลข

### Phase 2: Data Pipeline & Feature Engineering 
* **เป้าหมาย:** สร้างชุดข้อมูลมาตรฐาน (Structured Data) จากคลังเพลง
* **สิ่งที่ต้องทำ:** เขียนสคริปต์ Python วนลูปอ่านไฟล์เพลงทั้งหมดใน Dataset ยุบมิติเวลาของแต่ละ Feature ให้เป็นค่าทางสถิติ (`mean`, `std`), ทำ `StandardScaler` และบันทึกเป็นไฟล์ `.parquet` หรือ `.csv`

### Phase 3: Similarity & Data Analysis 
* **เป้าหมาย:** สร้างอัลกอริทึมแนะนำเพลงและตรวจสอบความแม่นยำ
* **สิ่งที่ต้องทำ:** คำนวณ Cosine Similarity เพื่อหาเพลงที่ใกล้เคียงที่สุด 5 อันดับแรก (Top-5), ทำ PCA ลดมิติข้อมูลเหลือ 2 มิติเพื่อพล็อตกราฟดูการกระจายตัวของแนวเพลง และวัดผลด้วยค่า `Precision@K`

### Phase 4: Database & Go Backend API 
* **เป้าหมาย:** ยกระดับโมเดลให้กลายเป็น Service ที่ใช้งานได้จริง
* **สิ่งที่ต้องทำ:** นำ Feature Vector บันทึกลงฐานข้อมูล และพัฒนา REST API ด้วยภาษา **Go** เพื่อรับค่า ID เพลงจากผู้ใช้ แล้วประมวลผลค้นหาเพลงที่ใกล้เคียงส่งกลับไปในรูปแบบ JSON

### Phase 5: UI & Portfolio Documentation 
* **เป้าหมาย:** สรุปผลงานให้สมบูรณ์และพร้อมนำเสนอ
* **สิ่งที่ต้องทำ:** สร้างหน้าเว็บอย่างง่าย (เช่น Streamlit หรือ Web UI พื้นฐาน) เพื่อกดเลือกเพลงและทดลองฟังเทียบเสียงจริง จากนั้นจัดทำไฟล์ `README.md` พร้อมภาพ Architecture Diagram ลงบน GitHub

---

## 5. แนวทางการต่อยอดในอนาคต (Future Enhancements)

* **Deep Learning Embeddings:** เปลี่ยนจากการสกัดค่าทางสถิติด้วยมือ มาใช้โมเดล Pre-trained อย่าง **CLAP (Contrastive Language-Audio Pretraining)** หรือ **CNN** อ่านภาพ Mel-Spectrogram เพื่อสร้าง Vector ที่เข้าใจอารมณ์เพลงได้ลึกซึ้งขึ้น
* **Real-time Audio Upload:** เพิ่มฟีเจอร์ให้ผู้ใช้อัปโหลดไฟล์เสียงหรืออัดเสียงฮัมเพลงสั้นๆ แล้วส่งผ่าน Go API ไปให้ Python สกัด Vector แบบ Real-time เพื่อค้นหาเพลงที่คล้ายกัน

---


# ส่วนเพิ่มเติม: ข้อเสนอแนะและการต่อยอดด้วย Kalman Filter

<div align="right">✍️ ส่วนนี้เขียนโดย Claude (Anthropic) 
</div>

---

## 6. ข้อประเมินและสิ่งที่ควรปรับ (Review)

### 6.1 จุดแข็งของโปรเจกต์
* ครอบคลุม Signal Processing, Audio Engineering, Data Science และ Programming ในชิ้นงานเดียว
* ผลลัพธ์ "ฟังด้วยหู" ได้ทันทีว่าแนะนำดีหรือไม่ ช่วยให้มีแรงจูงใจทำต่อเนื่อง
* Phase 1–3 เป็น Data Pipeline ครบวงจร (ingest → feature → analysis → evaluation) ใช้เป็น Portfolio สาย Data Analyst ได้

### 6.2 สิ่งที่ควรปรับ
1. **เข้าใจ DSP จริง ไม่ใช่แค่เรียก `librosa`:** เขียน STFT → Mel Filterbank → MFCC ด้วย NumPy เอง 1 รอบ แล้วเทียบผลกับ `librosa`
> **note** : เห็นด้วย อยากเรียนรู้เกี่ยวกับ DSP (Digital Signal Processing) 
2. **เปลี่ยน Dataset จาก GTZAN เป็น FMA Small:** GTZAN มีปัญหาเพลงซ้ำ ป้ายแนวเพลงผิด และไฟล์เสียงเพี้ยน (Sturm, 2013) ทำให้ตีความค่า Precision@K ได้ยาก ส่วน FMA Small มี 8,000 เพลง × 30 วินาที ใน 8 แนวที่สมดุลกัน
   * ข้อควรระวัง: "แนวเดียวกัน" ≠ "เสียงคล้ายกัน" ควรฟังตรวจด้วยหูประกอบเสมอ
4. **เลื่อน Go + pgvector ไปท้ายสุด:** เวกเตอร์หลักพันตัว คำนวณ Cosine Similarity ด้วย NumPy ก็เร็วพอ

---

## 7. การต่อยอดด้วย Kalman Filter: ติดตามรสนิยมผู้ใช้ (User Taste Tracking)

### 7.1 แนวคิด
Feature Vector ของเพลงเป็น **ค่าคงที่** จึงไม่เหมาะจะใส่ Kalman Filter (KF) โดยตรง สิ่งที่ **เปลี่ยนตามเวลา** คือ **รสนิยมของผู้ใช้** ซึ่งมีงานวิจัยรองรับ เช่น Collaborative Kalman Filter (Gultekin & Paisley) ที่มองรสนิยมผู้ใช้เป็นสถานะที่เลื่อนไปตามเวลา

### 7.2 การออกแบบโมเดล
| องค์ประกอบ | ความหมาย |
|---|---|
| **State** $x_k$ | เวกเตอร์รสนิยมผู้ใช้ใน PCA space (8–16 มิติ ต่อจาก Phase 3) |
| **Process Model** | $x_k = x_{k-1} + w_k$ (Random Walk) — $Q$ = ความเร็วที่รสนิยมเปลี่ยน |
| **Measurement** $z_k$ | Feature Vector ของเพลงที่ผู้ใช้ฟัง/กดชอบ, $H = I$ |
| **Measurement Noise** $R$ | ปรับตามสัญญาณ: กดชอบ → $R$ ต่ำ, ฟังจบเฉย ๆ → $R$ สูง |
| **Recommendation** | เลือกเพลงที่ใกล้ $\hat{x}_k$ มากที่สุด |
| **Explore / Exploit** | ใช้ Covariance $P$: ถ้า $P$ ใหญ่ (ไม่แน่ใจ) → กระจายคำแนะนำให้กว้างขึ้น |

### 7.3 การวัดผล (ไม่ต้องมีผู้ใช้จริง)
* จำลอง "ผู้ใช้สังเคราะห์" ที่รสนิยมค่อย ๆ เลื่อน (รู้ค่าจริง) แล้ววัด Tracking Error
* เปรียบเทียบกับ Baseline: ค่าเฉลี่ยธรรมดา และ Exponential Moving Average (EMA)

### 7.4 ข้อจำกัดที่ต้องรู้ไว้
* ใน Random Walk แบบง่าย KF ที่เข้า Steady State จะทำงานเทียบเท่า **Exponential Smoothing** (α = Kalman Gain) → KF อาจไม่ชนะ EMA ขาด ข้อได้เปรียบจริงคือ $P$ (ความไม่แน่นอน) และ $R$ ที่ปรับตามสัญญาณได้
* **รสนิยมหลายขั้ว:** ผู้ใช้ที่ชอบทั้ง Jazz และ Metal จะได้ค่าเฉลี่ยตกตรงกลาง → แก้ด้วย KF หลายตัวแยกตาม Cluster
* **การกดข้ามเพลง (Skip)** ไม่ใช่ค่าวัดแบบ Gaussian จึงใส่ KF ตรง ๆ ไม่ได้

### 7.5 Kalman Filter ฝั่ง DSP: Tempo Tracking
Tempo ในการเล่นจริงยืดหดได้ (Rubato) — งานของ Cemgil et al. มอง Tempo เป็น Hidden State แล้วประมาณด้วย KF ใช้ทำค่า BPM ให้เสถียรขึ้นได้

---

## 8. แนวคิดต่อยอดระยะยาว: ทฤษฎีดนตรี + Bayesian Filtering (ข้อสันนิษฐาน)

### 8.1 แบ่งเครื่องมือตามชนิดของสถานะ
| สถานะทางดนตรี | ชนิด | เครื่องมือที่เหมาะ |
|---|---|---|
| คอร์ด (Chord), คีย์ (Key) | ค่าแบบหมวด (Discrete) | **HMM** — โครงสร้าง Predict/Update เดียวกับ KF, Chroma + HMM เป็นมาตรฐานของ Chord Recognition |
| Tempo, Rubato | ค่าต่อเนื่อง (Continuous) | **Kalman Filter** |

**note** : HMM มาจากคำว่า "Hidden Markov Model" 
ref : https://en.wikipedia.org/wiki/Hidden_Markov_model


### 8.2 จากสถานะสู่ "ลักษณะการประพันธ์" (Compositional Style)
ดึงสถิติจากลำดับที่ตัวกรองประมาณได้ มาสร้าง **Style Vector**:
* **Chord Transition Probabilities** — เช่น ป๊อปวนคอร์ดซ้ำ, แจ๊สใช้ ii–V–I และคอร์ดแทนบ่อย
* **ความถี่ในการเปลี่ยนคีย์ (Modulation Rate)** — คลาสสิกมักเปลี่ยนคีย์บ่อยกว่าป๊อป
* **ความแปรปรวนของ Tempo** — แยกเพลงที่เล่นตาม Click Track ออกจากเพลงแบบ Rubato

→ ใช้แนะนำเพลงที่ "เรียงคอร์ดคล้ายกัน" ไม่ใช่แค่ "เสียงคล้ายกัน"

### 8.3 ข้อจำกัด
* ทฤษฎีดนตรีตะวันตก (ระบบ Major/Minor) ใช้ได้ไม่ดีกับดนตรีนอกระบบนี้ เช่น ดนตรีไทยเดิม
* เป็นงานระดับวิจัย MIR — เหมาะเป็นเฟสท้าย ๆ หรือหัวข้อ Senior Project

---

## 9. แหล่งอ้างอิง (References)
* Sturm, B. L. — *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use* — https://arxiv.org/abs/1306.1461
* Defferrard et al. — *FMA: A Dataset For Music Analysis* — https://github.com/mdeff/fma
* Gultekin, S. & Paisley, J. — *A Collaborative Kalman Filter for Time-Evolving Dyadic Processes* — https://arxiv.org/abs/1501.05624
* Gómez-Uribe, C. & Karrer, B. — *The Decoupled Extended Kalman Filter for Dynamic Exponential-Family Factorization Models* — https://arxiv.org/abs/1806.09976
* Cemgil, A. T. et al. — *On Tempo Tracking: Tempogram Representation and Kalman Filtering* — https://www.mcg.uva.nl/mmm-2003/abstracts/mmm-26.html
* Lee, K. & Slaney, M. — *A Unified System for Chord Transcription and Key Extraction using HMMs* (ISMIR 2007) — https://archives.ismir.net/ismir2007/paper/000245.pdf
* State Space Models & Kalman Filtering (EMA equivalence) — https://kundeng.github.io/bayeslearner-scikit-refresh/chapters/16-state-space-kalman.html
* หมายเหตุ: แผนรายสัปดาห์ การออกแบบโมเดลใน 7.2 และ Style Vector ใน 8.2 เป็นข้อเสนอของ Claude — ไม่มีแหล่งอ้างอิงชัดเจน

---
*— เขียนโดย Claude (Anthropic)*
