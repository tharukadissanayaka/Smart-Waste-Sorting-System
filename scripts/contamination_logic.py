#!/usr/bin/env python3
"""
scripts/contamination_logic.py

Implements a rule-based contamination detection logic layer for a smart waste sorting system.
Given a detection's class, confidence score, and optional metadata, this layer identifies
possible contamination risks (e.g., misclassification of visually similar items like plastic
vs glass, or low confidence detections) and outputs warning messages and risk levels.

It is structured with an abstract base class interface so that it can be easily swapped for
a learned/ML-based model in the future.
"""

import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class ContaminationDetector(ABC):
    """
    Abstract base class for waste contamination risk detection.
    Defines a standard interface so heuristic rules can be seamlessly swapped
    with a learned/machine-learning model later.
    """

    @abstractmethod
    def evaluate_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single object detection for contamination risk.

        Args:
            detection (dict): A dictionary representing the detection. Must contain:
                - 'class_name' (str): The predicted class ('Plastic', 'Paper', 'Glass', 'Metal')
                - 'confidence' (float): The model's confidence score (0.0 to 1.0)
                Optionally can contain:
                    - 'bbox' (list of float): Bounding box [x1, y1, x2, y2]
                    - 'class_id' (int): The integer class ID reference
                    
        Returns:
            dict: Evaluation results containing:
                - 'is_risk' (bool): True if contamination risk is identified, False otherwise.
                - 'warning_message' (str): Description of the risk or validation status.
                - 'risk_level' (str): 'LOW', 'MEDIUM', 'HIGH', or 'NONE'.
                - 'suggested_action' (str): Recommended action (e.g., 'Sort manually', 'Automated sorting').
        """
        pass


class RuleBasedContaminationDetector(ContaminationDetector):
    """
    Heuristic-based contamination risk detector.
    Uses configurable confidence thresholds and visually similar class pairs
    to flag potential misclassifications.
    """

    def __init__(
        self,
        critical_threshold: float = 0.45,
        low_confidence_threshold: float = 0.70,
        similar_pairs: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initializes the detector with threshold rules.

        Args:
            critical_threshold (float): Detections below this confidence are flagged as high risk.
            low_confidence_threshold (float): Detections below this are inspected for similar-class confusions.
            similar_pairs (dict): Mapping of classes to other classes they visually resemble.
        """
        self.critical_threshold = critical_threshold
        self.low_confidence_threshold = low_confidence_threshold

        # Default visually similar mappings based on waste domain confusions:
        # - Plastic (PET bottles) vs Glass (bottles/jars)
        # - Paper (white plastic-coated cups) vs Plastic
        # - Metal (cans/foils) vs Glass/Plastic reflections
        if similar_pairs is None:
            self.similar_pairs = {
                "Plastic": ["Glass"],
                "Glass": ["Plastic"],
                "Paper": ["Plastic"],
                "Metal": ["Glass", "Plastic"]
            }
        else:
            self.similar_pairs = similar_pairs

    def evaluate_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies heuristic rules to evaluate contamination risk of a single detection.
        """
        # Validate essential inputs
        if "class_name" not in detection or "confidence" not in detection:
            raise ValueError("Detection dict must contain 'class_name' and 'confidence'")

        class_name = str(detection["class_name"])
        confidence = float(detection["confidence"])

        # Check confidence boundary limits
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence score {confidence} must be between 0.0 and 1.0")

        # Rule 1: Critical Confidence Threshold (HIGH RISK)
        if confidence < self.critical_threshold:
            return {
                "is_risk": True,
                "warning_message": (
                    f"Critical contamination risk: low detection confidence ({confidence:.2f}) "
                    f"for class '{class_name}'."
                ),
                "risk_level": "HIGH",
                "suggested_action": "Manual Sorting"
            }

        # Rule 2: Visual Similarity Confusions (MEDIUM RISK)
        if confidence < self.low_confidence_threshold:
            similar_classes = self.similar_pairs.get(class_name, [])
            if similar_classes:
                similar_str = " or ".join(similar_classes)
                return {
                    "is_risk": True,
                    "warning_message": (
                        f"Possible contamination: item classified as {class_name} "
                        f"but visually similar to {similar_str}, confidence {confidence:.2f}."
                    ),
                    "risk_level": "MEDIUM",
                    "suggested_action": "Verify or Manual Sorting"
                }

        # Rule 3: Low Risk (Safe for automated routing)
        return {
            "is_risk": False,
            "warning_message": f"Low contamination risk: high confidence ({confidence:.2f}) prediction for class '{class_name}'.",
            "risk_level": "LOW",
            "suggested_action": "Automated Sorting"
        }

    def evaluate_stream_contamination(
        self,
        detections: List[Dict[str, Any]],
        expected_class: str
    ) -> Dict[str, Any]:
        """
        Evaluates a batch of detections on a conveyor/stream designated for a single category,
        identifying both misclassification risks and physical stream cross-contamination.

        Args:
            detections (list of dict): Detections currently observed.
            expected_class (str): The class this conveyor stream is designated to route.

        Returns:
            dict: Stream evaluation summary.
        """
        contaminants = []
        uncertain_items = []
        safe_items = []

        for det in detections:
            res = self.evaluate_detection(det)
            class_name = det["class_name"]

            # Scenario A: Physical stream contamination (Class mismatch)
            if class_name != expected_class:
                severity = "CRITICAL" if det["confidence"] >= self.critical_threshold else "HIGH"
                contaminants.append({
                    "detection": det,
                    "reason": f"Mismatched class: found '{class_name}' in stream designated for '{expected_class}'",
                    "severity": severity
                })
            # Scenario B: High classification uncertainty/risk
            elif res["is_risk"]:
                uncertain_items.append({
                    "detection": det,
                    "reason": res["warning_message"],
                    "severity": res["risk_level"]
                })
            # Scenario C: Clean, confident match
            else:
                safe_items.append(det)

        is_contaminated = len(contaminants) > 0 or len(uncertain_items) > 0

        # Build descriptive stream summary
        if contaminants:
            summary_msg = f"Contamination detected! Found {len(contaminants)} items of incorrect classes in {expected_class} stream."
        elif uncertain_items:
            summary_msg = f"Stream at risk: {len(uncertain_items)} items have high classification uncertainty."
        else:
            summary_msg = f"Stream clear: all {len(safe_items)} items verified as {expected_class}."

        return {
            "is_contaminated": is_contaminated,
            "summary": summary_msg,
            "contaminants": contaminants,
            "uncertain_items": uncertain_items,
            "safe_items": safe_items
        }


def test_contamination_logic():
    """
    Unit tests covering a handful of example detection inputs to verify logic rules.
    """
    print("Running contamination logic unit tests...")
    
    detector = RuleBasedContaminationDetector(
        critical_threshold=0.45,
        low_confidence_threshold=0.70
    )

    # 1. High confidence prediction (No Risk / Low Risk)
    det_high_conf = {"class_name": "Plastic", "confidence": 0.85}
    res_high = detector.evaluate_detection(det_high_conf)
    assert not res_high["is_risk"], "High confidence prediction should not be marked as risk"
    assert res_high["risk_level"] == "LOW"
    assert res_high["suggested_action"] == "Automated Sorting"

    # 2. Low confidence, visually similar pair (Medium Risk)
    det_similar = {"class_name": "Plastic", "confidence": 0.55}
    res_similar = detector.evaluate_detection(det_similar)
    assert res_similar["is_risk"], "Low confidence visually similar class should flag a risk"
    assert res_similar["risk_level"] == "MEDIUM"
    assert "visually similar to Glass" in res_similar["warning_message"]
    assert res_similar["suggested_action"] == "Verify or Manual Sorting"

    # 3. Very low confidence below critical threshold (High Risk)
    det_critical = {"class_name": "Metal", "confidence": 0.35}
    res_critical = detector.evaluate_detection(det_critical)
    assert res_critical["is_risk"], "Confidence below critical threshold must flag a risk"
    assert res_critical["risk_level"] == "HIGH"
    assert "Critical contamination risk" in res_critical["warning_message"]
    assert res_critical["suggested_action"] == "Manual Sorting"

    # 4. Custom similar pairs override configuration
    custom_detector = RuleBasedContaminationDetector(
        similar_pairs={"Paper": ["Metal"]}
    )
    det_paper = {"class_name": "Paper", "confidence": 0.60}
    res_paper = custom_detector.evaluate_detection(det_paper)
    assert res_paper["is_risk"]
    assert "visually similar to Metal" in res_paper["warning_message"]

    # 5. Stream contamination check - mixed items
    detections = [
        {"class_name": "Plastic", "confidence": 0.80},  # Safe
        {"class_name": "Plastic", "confidence": 0.55},  # Uncertain (Medium Risk)
        {"class_name": "Glass", "confidence": 0.90},    # Contaminant (Class mismatch)
    ]
    stream_res = detector.evaluate_stream_contamination(detections, expected_class="Plastic")
    assert stream_res["is_contaminated"], "Stream should be flagged as contaminated"
    assert len(stream_res["contaminants"]) == 1
    assert stream_res["contaminants"][0]["detection"]["class_name"] == "Glass"
    assert len(stream_res["uncertain_items"]) == 1
    assert len(stream_res["safe_items"]) == 1
    assert "Contamination detected!" in stream_res["summary"]

    print("All unit tests passed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
        test_contamination_logic()
        sys.exit(0)

    test_contamination_logic()
    print("-" * 60)
    print("DEMO: Evaluating sample single detections:")
    print("-" * 60)

    detector = RuleBasedContaminationDetector()
    samples = [
        {"class_name": "Plastic", "confidence": 0.88},
        {"class_name": "Plastic", "confidence": 0.55},
        {"class_name": "Glass", "confidence": 0.92},
        {"class_name": "Metal", "confidence": 0.38},
    ]

    for i, s in enumerate(samples, 1):
        res = detector.evaluate_detection(s)
        print(f"Detection {i}: Class='{s['class_name']}', Conf={s['confidence']}")
        print(f"  Is Risk?    {res['is_risk']}")
        print(f"  Risk Level: {res['risk_level']}")
        print(f"  Warning:    {res['warning_message']}")
        print(f"  Action:     {res['suggested_action']}\n")

    print("-" * 60)
    print("DEMO: Evaluating stream contamination (Expected stream: Plastic):")
    print("-" * 60)
    
    stream_res = detector.evaluate_stream_contamination(samples, "Plastic")
    print(f"Is Stream Contaminated? {stream_res['is_contaminated']}")
    print(f"Summary:                {stream_res['summary']}")
    print(f"Contaminant count:      {len(stream_res['contaminants'])}")
    print(f"Uncertain item count:   {len(stream_res['uncertain_items'])}")
    print(f"Safe item count:        {len(stream_res['safe_items'])}")
