import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import json
from mean import get_mean, get_std
from datasets.ucf101 import load_annotation_data, get_class_labels
from model import generate_model
from opts import parse_opts
from spatial_transforms import Compose, Normalize, Scale, ToTensor


def resume_model(opt, model):
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


def get_spatial_transform(opt):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        Scale((150, 150)),
        ToTensor(opt.norm_value),
        norm_method
    ])
    return spatial_transform


@torch.no_grad()
def predict(clip, model, spatial_transform, device):
    clip = [spatial_transform(img) for img in clip]
    clip = torch.stack(clip, dim=0).unsqueeze(0)
    clip = clip.to(device)
    output = model(clip)
    softmax_output = torch.softmax(output, dim=1)
    score, idx = torch.topk(softmax_output, k=1)
    mask = score > 0.55
    preds = idx[mask]
    confidences = score[mask]
    return preds, confidences, softmax_output


def main():
    opt = parse_opts()
    print(opt)

    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    idx_to_class = {label: name for name, label in class_to_idx.items()}
    print("클래스 인덱스 매핑:", idx_to_class)

    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
    model = generate_model(opt, device)

    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        model.eval()

        test_root = "./invert_data/video_data/image_data"
        spatial_transform = get_spatial_transform(opt)

        tp = fp = fn = tn = 0
        total_samples = 0
        times = []

        json_results = []

        for label in os.listdir(test_root):
            label_root = os.path.join(test_root, label)
            if not os.path.isdir(label_root):
                continue
            for clip_folder in os.listdir(label_root):
                frame_folder = os.path.join(label_root, clip_folder)
                if not os.path.isdir(frame_folder):
                    continue

                image_files = sorted([
                    os.path.join(frame_folder, f)
                    for f in os.listdir(frame_folder)
                    if f.endswith('.jpg') or f.endswith('.png')
                ])
                if len(image_files) < 16:
                    print(f"{frame_folder}: insufficient frames ({len(image_files)}), skipping")
                    continue

                clip = [Image.open(f).convert("RGB") for f in image_files[:16]]

                start = time.time()
                preds, confidences, softmax_output = predict(clip, model, spatial_transform, device)
                elapsed = (time.time() - start) * 1000  # ms
                times.append(elapsed)

                if preds.size(0) > 0:
                    result_str = idx_to_class[preds.item()]
                    confidence = confidences.item()
                else:
                    result_str = "no confident prediction"
                    confidence = 0.0

                print(f"{frame_folder}: {result_str} (confidence: {confidence:.4f}) | 추론 시간: {elapsed:.2f} ms")

                total_samples += 1

                classification_label = "fall-off" if confidence > 0.6 else "no-accident"

                # 성능 통계 계산
                if result_str == "no confident prediction":
                    if label == "fall-off":
                        fn += 1
                    elif label == "no-accident":
                        fp += 1
                else:
                    if label == "fall-off":
                        if result_str == "fall-off":
                            tp += 1
                        else:
                            fn += 1
                    elif label == "no-accident":
                        if result_str == "fall-off":
                            fp += 1
                        elif result_str == "no-accident":
                            tn += 1

                json_results.append({
                    "clip_id": clip_folder,
                    "predicted_label": result_str,
                    "confidence": confidence,
                    "test_time_ms": elapsed,
                    "classification": classification_label
                })

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        print('\n====[성능 통계]====')
        print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn} | 합계: {tp + tn + fp + fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1_score:.3f}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"전체 입력 샘플 수(제외 없이): {total_samples}")

        if times:
            avg_time = sum(times) / len(times)
            print(f"\n클립당 평균 추론 시간: {avg_time:.2f} ms")
        else:
            print("\n평균 추론 시간 계산에 사용할 데이터가 없습니다.")

        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        json_path = os.path.join(result_dir, "inference_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)

        print(f"\n추론 결과가 JSON 파일로 저장되었습니다: {json_path}")
        print(f"총 {total_samples}개의 샘플을 처리했습니다.")

if __name__ == "__main__":
    main()
