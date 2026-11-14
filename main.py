from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


# ── 파일 경로 설정 ──────────────────────────────────────
MODEL_PATH = Path(r"your model path")
LABELS_PATH = Path(r"your label path")

# 이미지 한 장 또는 테스트 폴더를 지정
IMAGE_PATH = Path(r"your test folder path")

# 테스트 폴더 평가 시 생성되는 결과 파일
RESULT_CSV_PATH = Path(r"write path and file name")

# 이미지 한 장을 평가할 때 정답을 알고 있으면 "ai" 또는 "real" 입력
# 정답을 입력하지 않으면 예측 결과만 출력
SINGLE_IMAGE_TRUE_LABEL = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ("ai", "real")


# ── 클래스 이름 및 이미지 전처리 ────────────────────────
def normalize_label(label):
    normalized = label.strip().lower().replace("-", "").replace("_", "")
    normalized = normalized.replace(" ", "")

    aliases = {
        "ai": "ai",
        "aigenerated": "ai",
        "aiartdata": "ai",
        "real": "real",
        "realart": "real",
    }

    if normalized not in aliases:
        raise ValueError(f"인식할 수 없는 클래스 이름: {label}")

    return aliases[normalized]


def read_model_labels(labels_path):
    labels = []

    with labels_path.open("r", encoding="utf-8-sig") as label_file:
        for line in label_file:
            line = line.strip()
            if not line:
                continue

            # 예: "0 ai" → index=0, name="ai"
            index_text, name = line.split(maxsplit=1)
            index = int(index_text)

            if index != len(labels):
                raise ValueError("labels.txt의 클래스 번호가 0부터 순서대로여야 합니다.")

            labels.append(normalize_label(name))

    if len(labels) != 2 or set(labels) != set(CLASS_NAMES):
        raise ValueError("labels.txt에는 ai와 real 두 클래스가 있어야 합니다.")

    return labels


def preprocess_image(image_path):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        image = ImageOps.fit(
            image,
            (224, 224),
            Image.Resampling.LANCZOS,
        )
        image_array = np.asarray(image, dtype=np.float32)

    normalized_image = image_array / 127.5 - 1.0
    return np.expand_dims(normalized_image, axis=0)


def predict_image(model, model_labels, image_path):
    input_data = preprocess_image(image_path)
    prediction = np.asarray(model.predict(input_data, verbose=0))

    if prediction.shape != (1, len(model_labels)):
        raise ValueError(f"예상하지 못한 모델 출력 형태: {prediction.shape}")

    if not np.all(np.isfinite(prediction)):
        raise ValueError("모델 출력에 NaN 또는 무한대가 포함되어 있습니다.")

    output_scores = prediction[0]
    predicted_index = int(np.argmax(output_scores))
    predicted_label = model_labels[predicted_index]

    scores = {
        label: float(output_scores[index])
        for index, label in enumerate(model_labels)
    }

    return predicted_label, scores


# ── 테스트 폴더에서 이미지와 정답 수집 ──────────────────
def collect_test_images(test_root):
    samples = []
    class_folders = {}

    for folder in sorted(test_root.iterdir()):
        if not folder.is_dir():
            continue

        try:
            label = normalize_label(folder.name)
        except ValueError:
            continue

        if label in class_folders:
            raise ValueError(f"같은 클래스의 폴더가 중복되었습니다: {label}")

        class_folders[label] = folder

    if set(class_folders) != set(CLASS_NAMES):
        raise ValueError(
            "테스트 폴더 바로 아래에 AI-generated와 Real 폴더가 있어야 합니다.\n"
            "예: 01_TEST_DO_NOT_TRAIN/AI-generated, 01_TEST_DO_NOT_TRAIN/Real"
        )

    for true_label in CLASS_NAMES:
        folder = class_folders[true_label]
        image_paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(f"이미지가 없는 클래스 폴더: {folder}")

        samples.extend((path, true_label) for path in image_paths)

    return samples


# ── 이미지 한 장 예측 ───────────────────────────────────
def evaluate_single_image(model, model_labels):
    predicted_label, scores = predict_image(
        model, model_labels, IMAGE_PATH
    )

    print(f"이미지: {IMAGE_PATH}")
    print(f"예측 클래스: {predicted_label}")
    print(f"AI 출력 점수:   {scores['ai']:.4f}")
    print(f"Real 출력 점수: {scores['real']:.4f}")

    if SINGLE_IMAGE_TRUE_LABEL is not None:
        true_label = normalize_label(SINGLE_IMAGE_TRUE_LABEL)
        print(f"실제 클래스: {true_label}")
        print(f"정답 여부: {predicted_label == true_label}")


# ── 테스트 폴더 평가 및 결과 저장 ────────────────────────
def evaluate_test_folder(model, model_labels):
    samples = collect_test_images(IMAGE_PATH)
    class_to_index = {
        label: index for index, label in enumerate(CLASS_NAMES)
    }

    confusion_matrix = np.zeros((2, 2), dtype=int)
    prediction_rows = []

    for number, (image_path, true_label) in enumerate(samples, start=1):
        # 오류가 있는 이미지를 조용히 제외하지 않고 실행을 중단
        try:
            predicted_label, scores = predict_image(
                model, model_labels, image_path
            )
        except Exception as error:
            raise RuntimeError(
                f"이미지 처리 중 오류가 발생했습니다: {image_path}"
            ) from error

        true_index = class_to_index[true_label]
        predicted_index = class_to_index[predicted_label]
        confusion_matrix[true_index, predicted_index] += 1

        prediction_rows.append({
            "image_path": str(image_path.relative_to(IMAGE_PATH)),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "ai_score": scores["ai"],
            "real_score": scores["real"],
            "correct": int(predicted_label == true_label),
        })

        if number % 20 == 0 or number == len(samples):
            print(f"평가 진행: {number}/{len(samples)}")

    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    print("\n[평가 결과]")
    print(f"테스트 이미지 수: {len(samples)}")
    print(f"전체 Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    for label, index in class_to_index.items():
        class_count = int(confusion_matrix[index].sum())
        correct_count = int(confusion_matrix[index, index])
        recall = correct_count / class_count

        print(
            f"{label} Recall: {recall:.4f} ({recall:.2%})"
            f" — {class_count}장 중 {correct_count}장 정답"
        )

    print("\n[혼동행렬: 행=실제 정답, 열=모델 예측]")
    print("                 예측 ai    예측 real")
    print(
        f"실제 ai       {confusion_matrix[0, 0]:9d}"
        f"    {confusion_matrix[0, 1]:9d}"
    )
    print(
        f"실제 real     {confusion_matrix[1, 0]:9d}"
        f"    {confusion_matrix[1, 1]:9d}"
    )

    RESULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=[
                "image_path",
                "true_label",
                "predicted_label",
                "ai_score",
                "real_score",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\n이미지별 예측 결과 저장: {RESULT_CSV_PATH}")


# ── 모델 로딩 및 평가 실행 ───────────────────────────────
def main():
    for required_path in (MODEL_PATH, LABELS_PATH, IMAGE_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {required_path}")

    model_labels = read_model_labels(LABELS_PATH)
    model = load_model(str(MODEL_PATH), compile=False)

    print(f"모델: {MODEL_PATH.name}")
    print(f"모델 출력 순서: {model_labels}")

    if IMAGE_PATH.is_file():
        evaluate_single_image(model, model_labels)
    else:
        evaluate_test_folder(model, model_labels)


if __name__ == "__main__":
    main()
