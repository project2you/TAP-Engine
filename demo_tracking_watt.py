# run_tracker.py
# นำเข้าคลาส ResourceTracker จากไฟล์หลัก
from resource_tracker_ai import ResourceTracker, AIWorkloadProfiler, RegionalCarbonIntensity, HardwareProfile
import os
import time

def main():
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
    output_dir = "ai_resource_tracking"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== AI Resource Tracking Demo ===")
    
    # แสดงภูมิภาคที่รองรับ
    print("\nรองรับภูมิภาคสำหรับการคำนวณคาร์บอน:")
    for region in RegionalCarbonIntensity.available_regions():
        intensity = RegionalCarbonIntensity(region).get_intensity()
        print(f"  - {region}: {intensity} gCO2/kWh")
    
    # แสดงโปรไฟล์ AI ที่รองรับ
    print("\nโปรไฟล์การทำงาน AI ที่รองรับ:")
    for profile in AIWorkloadProfiler.available_profiles():
        desc = AIWorkloadProfiler.profile_description(profile)
        print(f"  - {profile}: {desc}")
    
    # เลือกใช้ภูมิภาคไทยใกล้เคียงกับเอเชีย
    region = 'global'  # สามารถเปลี่ยนเป็น 'china' หรือ 'india' ซึ่งใกล้เคียงไทยมากกว่า
    
    # สร้าง Resource Tracker พร้อมเปิดการติดตาม GPU (ถ้ามี)
    tracker = ResourceTracker(
        interval_sec=0.5,  # เก็บข้อมูลทุก 0.5 วินาที
        region=region,
        gpu_enabled=True,
        track_network=True,
        track_processes=True,
        output_dir=output_dir
    )
    
    # บันทึกข้อมูลระบบ
    system_info = tracker.record_system_info()
    print(f"\nข้อมูลระบบ:")
    print(f"  - OS: {system_info['system']} {system_info['release']}")
    print(f"  - CPU: {system_info['processor']}")
    print(f"  - RAM: {system_info['total_ram_gb']:.1f} GB")
    print(f"  - ความเข้มข้นคาร์บอน: {system_info['carbon_intensity']} gCO2/kWh ({region})")
    
    # เริ่มการติดตาม
    print("\nเริ่มติดตามการใช้ทรัพยากร...")
    tracker.start()
    
    # จำลองการทำงานของ LLM Inference
    print("\nจำลองการทำงาน LLM Inference...")
    with tracker.track_section("LLM_Inference", workload_type="llm_inference"):
        # จำลองการใช้งาน CPU และหน่วยความจำสำหรับ LLM inference
        data = [0] * 10000000  # จองหน่วยความจำ
        for i in range(3):
            # การคำนวณที่ใช้ CPU สูง
            _ = sum([i*i for i in range(500000)])
            time.sleep(1)
    
    # จำลองการทำ Data Preprocessing
    print("\nจำลองการทำงาน Data Preprocessing...")
    with tracker.track_section("Data_Preprocessing", workload_type="data_preprocessing"):
        # จำลองการใช้งานดิสก์และ CPU สำหรับการประมวลผลข้อมูล
        test_file = os.path.join(output_dir, "test_data.txt")
        with open(test_file, "w") as f:
            for i in range(50000):
                f.write(f"ข้อมูลทดสอบแถวที่ {i}\n")
        
        # อ่านไฟล์และประมวลผล
        with open(test_file, "r") as f:
            lines = f.readlines()
            word_count = sum(len(line.split()) for line in lines)
            print(f"  - อ่านไฟล์ {len(lines)} บรรทัด, {word_count} คำ")
        
        time.sleep(2)
    
    # จำลองการทำงานของ Computer Vision
    print("\nจำลองการทำงาน Computer Vision Inference...")
    with tracker.track_section("CV_Inference", workload_type="vision_inference"):
        # จำลองการประมวลผลภาพ
        image_data = [[0 for _ in range(1024)] for _ in range(1024)]  # จำลองภาพขนาด 1024x1024
        
        # จำลองการประมวลผลภาพ
        for _ in range(3):
            # การคำนวณที่ใช้ CPU สูง
            for i in range(100):
                for j in range(100):
                    # จำลองการประมวลผลพิกเซล
                    image_data[i][j] = (i * j) % 255
            time.sleep(1)
    
    # หยุดการติดตาม
    tracker.stop()
    
    # สร้างรายงานสรุป
    print("\nสรุปการใช้ทรัพยากรโดยรวม:")
    overall_summary = tracker.summarize()
    print(overall_summary.to_string(index=False))
    
    print("\nสรุปการใช้ทรัพยากรตามส่วน:")
    section_summary = tracker.summarize_sections()
    print(section_summary.to_string(index=False))
    
    # สร้างกราฟ
    plot_file = os.path.join(output_dir, "resource_metrics.png")
    tracker.plot_metrics(plot_file)
    
    # ส่งออกข้อมูล
    csv_file = os.path.join(output_dir, "resource_data.csv")
    tracker.export_data(csv_file, format='csv')
    json_file = os.path.join(output_dir, "resource_data.json")
    tracker.export_data(json_file, format='json')
    
    print(f"\nบันทึกผลลัพธ์ไปยัง {output_dir}/")
    print(f"  - ข้อมูลดิบ: {csv_file}")
    print(f"  - ข้อมูล JSON: {json_file}")
    print(f"  - กราฟ: {plot_file}")
    print(f"  - รายงานสรุป: {output_dir}/resource_summary_*.json")
    print(f"  - รายงานสรุปตามส่วน: {output_dir}/section_summary_*.json")
    print(f"  - ข้อมูลระบบ: {output_dir}/system_info.json")
    
    # คำนวณและแสดงคาร์บอนฟุตพริ้นท์
    total_co2 = overall_summary['Total CO2 (g)'].values[0]
    duration_sec = overall_summary['Duration (s)'].values[0]
    
    # แสดงข้อมูลในรูปแบบที่เข้าใจง่าย
    print("\n=== สรุปคาร์บอนฟุตพริ้นท์ ===")
    print(f"ระยะเวลาทั้งหมด: {duration_sec:.2f} วินาที ({duration_sec/60:.2f} นาที)")
    print(f"ปริมาณ CO2 ทั้งหมด: {total_co2:.4f} กรัม")
    print(f"อัตราการปล่อย CO2: {total_co2/(duration_sec/3600):.4f} กรัม/ชั่วโมง")
    
    # เปรียบเทียบกับกิจกรรมทั่วไป
    car_per_km = 120  # กรัม CO2 ต่อกิโลเมตร (รถยนต์ขนาดเล็ก)
    equivalent_driving = total_co2 / car_per_km
    
    print(f"\nเทียบเท่ากับ:")
    print(f"- การขับรถยนต์: {equivalent_driving*1000:.2f} เมตร")
    
    if os.path.exists(test_file):
        os.remove(test_file)  # ลบไฟล์ทดสอบ

if __name__ == "__main__":
    main()
