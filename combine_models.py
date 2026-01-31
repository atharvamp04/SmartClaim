from ultralytics import YOLO

class PartDamageCombiner:

    def __init__(self, part_model_path, damage_model_path):
        self.part_model = YOLO(part_model_path)
        self.damage_model = YOLO(damage_model_path)

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def assign_damage_to_part(self, part_boxes, damage_boxes):
        assignments = []

        for dmg in damage_boxes:
            dmg_center = self.get_center(dmg['bbox'])
            assigned_part = None

            for part in part_boxes:
                x1, y1, x2, y2 = part["bbox"]

                # Check if damage center lies inside this part's bounding box
                if x1 <= dmg_center[0] <= x2 and y1 <= dmg_center[1] <= y2:
                    assigned_part = part["name"]
                    break

            assignments.append({
                "damage_type": dmg["name"],
                "damage_confidence": dmg["conf"],
                "assigned_part": assigned_part
            })

        return assignments

    def predict(self, image_path):
        # ---------------------------------------------------------
        # 1️⃣ PARTS DETECTION
        # ---------------------------------------------------------
        part_results = self.part_model(image_path)[0]
        part_boxes = []

        for box in part_results.boxes:
            cls_id = int(box.cls)
            name = self.part_model.names[cls_id]
            bbox = box.xyxy[0].tolist()

            part_boxes.append({
                "name": name,
                "bbox": bbox
            })

        # ---------------------------------------------------------
        # 2️⃣ DAMAGES DETECTION
        # ---------------------------------------------------------
        dmg_results = self.damage_model(image_path)[0]
        damage_boxes = []

        for box in dmg_results.boxes:
            cls_id = int(box.cls)
            name = self.damage_model.names[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()

            damage_boxes.append({
                "name": name,
                "conf": conf,
                "bbox": bbox
            })

        # ---------------------------------------------------------
        # 3️⃣ ASSIGN DAMAGE → PART
        # ---------------------------------------------------------
        assignments = self.assign_damage_to_part(part_boxes, damage_boxes)

        return {
            "parts": part_boxes,
            "damages": damage_boxes,
            "assignments": assignments
        }
