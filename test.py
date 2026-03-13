from src.pipeline.inspection_pipeline import run_inspection

image_path = "test_image.jpg"

report = run_inspection(image_path)

print("\n========== AI SITE INSPECTION REPORT ==========\n")
print(report)