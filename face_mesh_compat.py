import os
import urllib.request
import mediapipe as mp

class LandmarksWrapper:
    def __init__(self, lms):
        self.landmark = lms

class FaceMeshResult:
    def __init__(self, face_landmarks):
        if face_landmarks and len(face_landmarks) > 0:
            self.multi_face_landmarks = [LandmarksWrapper(fl) for fl in face_landmarks]
        else:
            self.multi_face_landmarks = None

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.use_legacy = False
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.legacy_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.use_legacy = True
            except Exception:
                self.use_legacy = False

        if not self.use_legacy:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import config
            model_path = config.get_resource_path(os.path.join("models", "face_landmarker.task"))
            
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                print(f"[INFO] Downloading FaceLandmarker model to {model_path}...")
                urllib.request.urlretrieve(url, model_path)
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        if self.use_legacy:
            return self.legacy_mesh.process(rgb_frame)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.detector.detect(mp_image)
        return FaceMeshResult(res.face_landmarks)
