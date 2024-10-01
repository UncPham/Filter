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
from src.data.dlib_datamodule import TransformDataset  # noqa: E402

from src import utils
from mtcnn.mtcnn import MTCNN # take too much time
import mediapipe as mp 

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig):

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating model <{cfg.data.transform_val._target_}>")
    transform: Compose = hydra.utils.instantiate(cfg.data.transform_val)

    log.info(f"Loading model from checkpoint {cfg.ckpt_path}")
    model = model.load_from_checkpoint(cfg.ckpt_path, net=model.net)
    model.eval()

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    cap = cv2.VideoCapture(0)
    # detector = MTCNN()
    mp_face_detection = mp.solutions.face_detection
    # mp_drawing = mp.solutions.drawing_utils


    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        # Lật khung hình theo chiều ngang (trái-phải)
        frame = cv2.flip(frame, 1)

        # print(type(frame), frame.shape)
        # 1. Chuyển từ BGR (OpenCV) sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = []
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                # Lấy bounding box của khuôn mặt đầu tiên
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    if w > 0 and h > 0 and y+h <= ih and x+w <= iw: 
                        cropped_face = frame[y:y+h, x:x+w]
                        # Kiểm tra xem hình ảnh đã cắt có trống hay không
                        if cropped_face.size != 0:
                            transformed = transform(image=cropped_face)['image']
                            faces.append(transformed)
                        else:
                            print(f"Empty cropped face with shape: {cropped_face.shape}")
        # Chuyển danh sách các tensor thành một tensor duy nhất với batch sizesudo nano /usr/local/hadoop/etc/hadoop/mapred-site.xml

        if len(faces) > 0:
            faces = torch.stack(faces)  # Kết hợp tất cả các tensors trong danh sách thành một tensor (batch_size, channels, height, width)
            # 3. Chuyển từ NumPy array sang PyTorch Tensor
            frame_tensor = torch.from_numpy(frame_rgb)

            # 4. Chuyển đổi kiểu dữ liệu sang float và chuẩn hóa giá trị pixel về [0, 1]
            frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255  # Đổi thứ tự thành (C, H, W)

            # 5. Thêm chiều batch size
            frame_tensor = frame_tensor.unsqueeze(0)  # Thêm batch size = 1

            h, w = frame.shape[:2]

            # Apply preprocessing
            # transformed = transform(image=frame)['image']
            # input_transformed = transformed.unsqueeze(0)  # Add batch dimension

            # print(type(input_transformed), input_transformed.shape)

            output_h, output_w = faces.shape[2:]

            # Perform prediction
            # log.info("Starting predictions!")
            predictions = model(faces)

            predictions = predictions.detach().numpy()
            
            scale_x = 1/w
            scale_y = 1/h
            
            scaled_predictions = []
            for i in range(predictions.shape[0]):  # Duyệt qua batch size b
                # scaled_points = [(int(x * scale_x), int(y * scale_y)) for (x, y) in predictions[i]]
                scaled_predictions.extend(predictions[i])
            scaled_predictions = np.array(scaled_predictions)
            scaled_predictions = np.expand_dims(scaled_predictions, axis=0)
        

            annotated_image = TransformDataset.annotate_tensor(frame_tensor, scaled_predictions)
            annotated_image = annotated_image.squeeze(0)
            annotated_image = annotated_image.permute(1, 2, 0).numpy()
            # Process predictions
            # for pred in predictions:
            #     pred = pred[0]  # Get the prediction
            #     
            #     break

            # scaled_image = cv2.resize(annotated_image, (w, h), interpolation=cv2.INTER_LINEAR)
            

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            # Hiển thị khung hình với dự đoán
            cv2.imshow('Camera Feed', annotated_image)
        else:
            print("No faces detected")
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
