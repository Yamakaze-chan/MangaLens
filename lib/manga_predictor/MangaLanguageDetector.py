import onnxruntime as ort
import numpy as np
from PIL import Image

class MangaLanguageDetector:
    def __init__(self, model_path, device='cpu'):
        """
        Khởi tạo detector sử dụng ONNX Runtime.
        :param model_path: Đường dẫn tới file .onnx
        :param device: 'cpu' hoặc 'cuda' (nếu máy có GPU)
        """
        self.class_names = ['Chinese', 'Japanese', 'Korean']
        self.input_size = (224, 224)
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, image):
        """Tiền xử lý ảnh sử dụng Padding để giữ nguyên tỉ lệ (Aspect Ratio)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            image = image.convert('RGB')

        w, h = image.size
        ratio = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", self.input_size, (255, 255, 255))
        
        paste_x = (self.input_size[0] - new_w) // 2
        paste_y = (self.input_size[1] - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))

        img_data = np.array(new_image).astype(np.float32) / 255.0
        img_data = (img_data - self.mean) / self.std

        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        
        return img_data

    def _softmax(self, x):
        """Tính xác suất phần trăm."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image):
        """
        Dự đoán ngôn ngữ từ ảnh.
        :param image: Có thể là đường dẫn file (str), PIL Image, hoặc numpy array
        :return: Dictionary chứa tên lớp và xác suất (%)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]
        probs = self._softmax(logits)
        result = {self.class_names[i]: float(probs[i]) for i in range(3)}
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def get_top_label(self, image):
        """Trả về duy nhất nhãn có tỉ lệ cao nhất."""
        results = self.predict(image)
        return list(results.keys())[0]