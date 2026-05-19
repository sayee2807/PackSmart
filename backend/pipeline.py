from modules.detection import detect_front_object, detect_top_object
from modules.dimensions import get_real_dimensions
from modules.material import predict_material
from modules.weight import estimate_weight
from modules.packaging import get_packaging_recommendation
from modules.bom import generate_bom
import os
import base64
import cv2

def run_detection_pipeline(front_image_path, top_image_path, real_width_cm):
    """
    Stage 1: Detection, dimensions, material, weight estimation only
    """
    if real_width_cm <= 0:
        raise ValueError("Real width must be greater than zero")

    front_data = detect_front_object(front_image_path)
    top_data = detect_top_object(top_image_path)

    real_dimensions = get_real_dimensions(
        width_front_px=front_data["width_px"],
        height_front_px=front_data["height_px"],
        width_top_px=top_data["width_top_px"],
        height_top_px=top_data["height_top_px"],
        real_width_cm=real_width_cm
    )

    material_data = predict_material(front_data["crop_image_path"], front_data["object_name"])

    estimated_weight = estimate_weight(
        real_dimensions["volume_cm3"],
        material_data["materials"][0]["name"],
        front_data["object_name"]
    )

    # bbox_b64 = None
    # bbox_path = front_data.get("bbox_image_path")
    # if bbox_path and os.path.exists(bbox_path):
    #     _, buffer = cv2.imencode(".jpg", cv2.imread(bbox_path))
    #     bbox_b64 = base64.b64encode(buffer).decode("utf-8")
        # os.remove(bbox_path) 
    
    crop_path = front_data.get("crop_image_path")
    if crop_path and os.path.exists(crop_path):
        os.remove(crop_path)

    return {
        "bbox_image_path":front_data["bbox_image_path"],
        "object_name": front_data["object_name"],
        "object_confidence": front_data["confidence"],
        "real_dimensions": real_dimensions,
        "material": material_data,
        "estimated_weight": estimated_weight,
    }


def run_packaging_pipeline(material_data, real_dimensions, final_weight):
    """
    Stage 2: Packaging, BOM, pricing using final weight
    """
    packaging = get_packaging_recommendation(
        material_data["object_category"],
        material_data["fragility"],
        final_weight,
        real_dimensions["length_cm"],
        real_dimensions["width_cm"],
        real_dimensions["height_cm"]
    )

    bom_result = generate_bom(
        packaging["packaging_material"],
        packaging["adjusted_dimensions"],
        packaging["protection_layer"]
    )

    return {
        "packaging": packaging,
        "bom": bom_result["bom"],
        "grand_total": bom_result["grand_total"],
        "weight": final_weight,
    }
