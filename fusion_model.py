import numpy as np
from scipy.optimize import linear_sum_assignment

class FusionModel:
    def __init__(self, product_database):
        self.product_database = product_database

    def fuse_detections(self, visual_detections, weight_prediction):
        fused_results = []
        for det in visual_detections:
            product_id = det['class_id']
            visual_confidence = det['confidence']
            expected_weight = self.product_database[product_id]['weight']
            weight_confidence = self.calculate_weight_confidence(weight_prediction, expected_weight)
            fused_confidence = (visual_confidence + weight_confidence) / 2
            fused_results.append({
                'product_id': product_id,
                'product_name': det['class_name'],
                'confidence': fused_confidence,
                'bbox': det['bbox']
            })
        return fused_results

    def calculate_weight_confidence(self, predicted_weight, expected_weight, tolerance=0.1):
        weight_diff = abs(predicted_weight - expected_weight)
        if weight_diff <= tolerance * expected_weight:
            return 1.0
        else:
            return max(0, 1 - (weight_diff / expected_weight))

def main():
    # Example usage
    product_database = {
        0: {'name': 'Product A', 'weight': 100},
        1: {'name': 'Product B', 'weight': 200},
        2: {'name': 'Product C', 'weight': 150}
    }

    fusion_model = FusionModel(product_database)

    visual_detections = [
        {'class_id': 0, 'class_name': 'Product A', 'confidence': 0.8, 'bbox': (100, 100, 200, 200)},
        {'class_id': 1, 'class_name': 'Product B', 'confidence': 0.7, 'bbox': (300, 300, 400, 400)}
    ]

    weight_prediction = 180  # Example weight prediction

    fused_results = fusion_model.fuse_detections(visual_detections, weight_prediction)
    print("Fused Results:", fused_results)

if __name__ == "__main__":
    main()