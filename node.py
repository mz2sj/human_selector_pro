import torch
import numpy as np
import folder_paths
from ultralytics import YOLO
import cv2
from insightface.app import FaceAnalysis


class HumanFrameSelectorPro:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),

                "pose_model": (folder_paths.get_filename_list("yolo"),),
                "face_model": (folder_paths.get_filename_list("yolo"),),

                "threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0}),
                "pose_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1}),
                "face_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1}),

                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "frame_stride": ("INT", {"default": 3, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "first_valid_index")
    FUNCTION = "run"
    CATEGORY = "custom/human"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pose_model = None
        self.face_model = None
        self.face_analyser = None

    def load_models(self, pose_path, face_path):
        if self.pose_model is None:
            self.pose_model = YOLO(pose_path)
            self.pose_model.to(self.device)

        if self.face_model is None:
            self.face_model = YOLO(face_path)
            self.face_model.to(self.device)

        if self.face_analyser is None:
            self.face_analyser = FaceAnalysis(name="buffalo_l")
            self.face_analyser.prepare(ctx_id=0)

    # ===============================
    # 👁️ 眼睛检测（68点 EAR）
    # ===============================
    def compute_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def is_eye_open(self, kps):
        if kps is None:
            return False, 0.0

        kps = np.array(kps)

        if kps.shape[0] >= 68:
            try:
                left = kps[36:42]
                right = kps[42:48]
                ear = (self.compute_ear(left) + self.compute_ear(right)) / 2
                return ear > 0.2, ear
            except:
                return False, 0.0

        return False, 0.0

    # ===============================
    # 🙂 正脸评分
    # ===============================
    def frontal_score(self, kps):
        if kps is None or len(kps) < 3:
            return 0.0

        kps = np.array(kps)

        if kps.shape[0] >= 68:
            l, r, n = kps[36], kps[45], kps[30]
        else:
            l, r, n = kps[0], kps[1], kps[2]

        mid = (l + r) / 2
        dist = np.linalg.norm(l - r)

        return max(0.0, 1 - np.linalg.norm(n - mid) / (dist + 1e-6))

    # ===============================
    # 📸 清晰度
    # ===============================
    def face_quality(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        return sharpness

    # ===============================
    # 🦴 Pose完整度（关键点数量）
    # ===============================
    def pose_score(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return 0.0

        return float((keypoints > 0.5).sum()) / len(keypoints)

    # ===============================
    # 🚀 主函数
    # ===============================
    def run(self, images, pose_model, face_model,
            threshold, pose_weight, face_weight,
            batch_size, frame_stride):

        print("===== Human Selector START =====")

        pose_path = folder_paths.get_full_path("yolo", pose_model)
        face_path = folder_paths.get_full_path("yolo", face_model)

        self.load_models(pose_path, face_path)

        imgs = (images.cpu().numpy() * 255)
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)

        indexed_imgs = [(i, img) for i, img in enumerate(imgs) if img is not None]
        indexed_imgs = indexed_imgs[::max(1, frame_stride)]

        best_img = None
        best_score = -1

        first_valid_index = -1

        for i in range(0, len(indexed_imgs), batch_size):

            batch = indexed_imgs[i:i + batch_size]
            batch_imgs = [x[1] for x in batch]

            pose_results = self.pose_model(batch_imgs, verbose=False)

            for (orig_idx, img), p_res in zip(batch, pose_results):

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # ========= face =========
                try:
                    faces = self.face_analyser.get(img_bgr)
                except Exception as e:
                    print(f"❌ insightface失败: {e}")
                    continue

                if len(faces) == 0:
                    continue

                # ========= pose =========
                if p_res.keypoints is None:
                    continue

                kp = p_res.keypoints.conf.cpu().numpy()
                if len(kp) == 0:
                    continue

                # ✅ 同时满足才算有效
                if first_valid_index == -1:
                    first_valid_index = orig_idx
                    print(f"🎯 首次 人脸+人体: index={first_valid_index}")

                face_obj = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                kps = face_obj.kps

                # ===== scoring =====
                p_score = self.pose_score(kp[0])
                f_score = self.frontal_score(kps)
                q_score = self.face_quality(img, face_obj.bbox)

                final = pose_weight * p_score + face_weight * (f_score + q_score)

                if final > best_score:
                    best_score = final
                    best_img = img

                if final >= threshold:
                    return (
                        torch.from_numpy(best_img / 255.0).unsqueeze(0).float(),
                        int(first_valid_index)
                    )

        # fallback
        if best_img is not None:
            return (
                torch.from_numpy(best_img / 255.0).unsqueeze(0).float(),
                int(first_valid_index)
            )

        print("❌ 所有帧都失败，fallback")

        return (
            torch.from_numpy(imgs[0] / 255.0).unsqueeze(0).float(),
            int(first_valid_index)
        )


NODE_CLASS_MAPPINGS = {
    "HumanFrameSelectorPro": HumanFrameSelectorPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanFrameSelectorPro": "Human Selector PRO (Face + Pose Required)"
}