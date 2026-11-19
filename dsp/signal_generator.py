import numpy as np

def make_time(start_s, end_s, sr) -> np.ndarray:
    """
    ฟังก์ชันสำหรับการสร้างช่วงเวลาสำหรับการสร้างสัญญาณ 
    start_s: เวลาเริ่มต้น หน่วยเป็นวินาที (s)
    end_s: เวลาสิ้นสุด หน่วยเป็นวินาที (s)
    sr: Sampling Rate (Hz)

    ฟังก์ชันจะมีการ return เป็นระยะเวลาสำหรับการสร้างสัญญาณหน่วยเป็นวินาที (s)
    """

    duration = end_s - start_s  # ระยะเวลาของสัญญาณ
    n_sample = int(round(duration * sr))   

    t = np.arange(n_sample) 
    return (t/sr)+start_s


def create_sin(amplitude, frequency, t, phase=0):
    """
    ฟังก์ชันสำหรับการสร้างสัญญาณ Sine ใส่ Argument 4 ตัว แอมพลิจูด ความถี่ เวลา เฟส
    amplitude: ขนาดของแอมพลิจูดของสัญญาณ
    frequency: ความถี่ของสัญญาณ หน่วยเป็นเฮิร์ตซ์ (Hz)
    t: ระยะเวลาของสัญญาณ หน่วยเป็นวินาที (s)
    phase: มุมเฟสของสัญญาณ หน่วยเป็นเรเดียน (rad) หากไม่มีการใส่ค่าจะมี default = 0

    ฟังก์ชันจะมีการ return เป็นสัญญาณ Sine ประเภทของข้อมูลเป็น np.array
    """
    x = 2 * np.pi * frequency * t
    signal = amplitude * np.sin(x + phase)
    return signal


def normalize_signal(signal: np.array, target: float =0.9) -> np.array:
    """
        ฟังก์ชันสำหรับการ Normalize amplitude ของสัญญาณให้มี amplitude สูงสุดเท่ากับค่า parameter ตัวที่ 2
        signal: ตัวแปรสัญญาณที่ต้องการ Normalize 
        target: ขนาดของแอมพลิจูดหลังจากการ Normalize หากไม่มีการใส่ค่าจะมี default = 0.9
    
        ฟังก์ชันจะมีการ return เป็นสัญญาณที่มีการ Normalize แล้ว
    """
    peak = np.max(np.abs(signal))

    if peak == 0:
        return signal

    gain = target / peak

    return signal * gain

