from combine_models import PartDamageCombiner

IMAGE_PATH = "test_images/image.png"

combiner = PartDamageCombiner(
    part_model_path="models/yolov8_car_parts.pt",
    damage_model_path="models/yolov8_car_damage.pt"
)

result = combiner.predict(IMAGE_PATH)

# ------------------------
# PRINT PARTS
# ------------------------
print("\n=== PARTS DETECTED ===")
for p in result["parts"]:
    print(p)

# ------------------------
# PRINT DAMAGES
# ------------------------
print("\n=== DAMAGES DETECTED ===")
for d in result["damages"]:
    print(d)

# ------------------------
# PRINT ASSIGNMENTS
# ------------------------
print("\n=== DAMAGE → PART ASSIGNMENTS ===")
for a in result["assignments"]:
    print(f"{a['damage_type']} → {a['assigned_part']} (conf {a['damage_confidence']:.2f})")
