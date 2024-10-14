from typing import List, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
import cv2
from albumentations import Compose
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset, DlibDataset  # noqa: E402
from src.models.dlib_module import DlibLitModule

from src import utils
import mediapipe as mp
from PIL import Image

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig):

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating model <{cfg.data.transform_val._target_}>")
    transform: Compose = hydra.utils.instantiate(cfg.data.transform_val)

    log.info(f"Loading model from checkpoint {cfg.ckpt_path}")
    model = DlibLitModule.load_from_checkpoint(cfg.ckpt_path, net=model.net)
    model.eval()

    cap = cv2.VideoCapture(0)

    mp_face_detection = mp.solutions.face_detection


    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        # Lật khung hình theo chiều ngang (trái-phải)
        frame = cv2.flip(frame, 1)

        # Chuyển từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = []
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                # Lấy bounding box của khuôn mặt đầu tiên
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    x, y, w, h = max(0, x), max(0, y), min(w, iw - x), min(h, ih - y) 
                    
                    if w > 0 and h > 0: 
                        cropped_face = frame[y:y+h, x:x+w]
                        # Kiểm tra xem hình ảnh đã cắt có trống hay không
                        if cropped_face.size != 0:
                            transformed = transform(image=cropped_face)['image']
                            input_transformed = transformed.unsqueeze(0)

                            predictions = model(input_transformed)
                            predictions = predictions.detach().numpy()
                            predictions = ((predictions + 0.5) * np.array([w, h])) + np.array([x, y])

                            landmarks.extend(predictions)
                        else:
                            print(f"Empty cropped face with shape: {cropped_face.shape}")
        
        if len(landmarks) > 0:
            landmarks = np.stack(landmarks)

            annotated_image = Image.fromarray(frame)

            for lm in landmarks:
                annotated_image = DlibDataset.annotate_image_triplot(annotated_image, lm)
                
            frame = np.array(annotated_image)
        else: 
            print("landmark detection fail!")

        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

    # Trả về giá trị rỗng để tránh lỗi unpack
    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_on_cam.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
