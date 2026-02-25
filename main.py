import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np
import time
import sys
import copy
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from train import train_epoch
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from dataset import get_training_set, get_validation_set
from mean import get_mean
from spatial_transforms import (
    Compose, Normalize, Scale, ToTensor, Grayscale)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel

INCIDENT_CLASSES = ['bump', 'fall-down', 'fall-off', 'hit', 'jam']


def log_print(message):
    print(message)
    sys.stdout.flush()


def resume_model(opt, model, optimizer):
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_print(f"Model Restored from Epoch {checkpoint['epoch']}")
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def get_loaders(opt):
    log_print("데이터로더 생성 시작...")
    log_print("Training 데이터 전처리 설정 중...")
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform_list = [Scale((opt.sample_size, opt.sample_size))]
    if getattr(opt, 'use_grayscale', False):
        spatial_transform_list.append(Grayscale())
    spatial_transform_list.extend([ToTensor(opt.norm_value), norm_method])
    spatial_transform = Compose(spatial_transform_list)
    temporal_transform = TemporalRandomCrop(20)
    target_transform = ClassLabel()

    log_print("Training 데이터셋 로딩 중...")
    training_data = get_training_set(opt, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)
    log_print(f"Training 데이터셋 로딩 완료: {len(training_data)}개 샘플, {len(train_loader)}개 배치")

    log_print("Validation 데이터 전처리 설정 중...")
    spatial_transform_list = [Scale((opt.sample_size, opt.sample_size))]
    if getattr(opt, 'use_grayscale', False):
        spatial_transform_list.append(Grayscale())
    spatial_transform_list.extend([ToTensor(opt.norm_value), norm_method])
    spatial_transform = Compose(spatial_transform_list)
    target_transform = ClassLabel()
    temporal_transform = LoopPadding(20)

    log_print("Validation 데이터셋 로딩 중...")
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True)

    log_print(f"Validation 데이터셋 로딩 완료: {len(validation_data)}개 샘플, {len(val_loader)}개 배치")
    log_print("데이터로더 생성 완료!\n")

    return train_loader, val_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_to_param_ratio(model, train_loader):
    num_params = count_parameters(model)
    num_samples = len(train_loader.dataset)
    ratio = num_params / num_samples if num_samples > 0 else float('inf')
    return ratio, num_params, num_samples


def plot_loss(train_losses, val_losses, output_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Train & Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    return cm, f1, precision, recall


def run_single_training(opt, run_label=None):
    run_name = run_label or infer_positive_class(opt.annotation_path) or 'single'
    log_print("학습 프로세스 시작!")
    log_print(f"러닝 태그: {run_name}")
    log_print("="*60)
    log_print("설정 파라미터:")
    for key, value in vars(opt).items():
        log_print(f"   {key}: {value}")
    log_print("="*60)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_print(f"시드 설정 완료: {seed}")

    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
    log_print(f"디바이스 설정: {device}")

    if torch.cuda.is_available() and opt.use_cuda:
        log_print(f"GPU 정보: {torch.cuda.get_device_name(opt.gpu)}")
        log_print(f"GPU 메모리: {torch.cuda.get_device_properties(opt.gpu).total_memory / 1024**3:.1f}GB")

    log_dir = getattr(opt, 'tensorboard_dir', 'tf_logs')
    summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    log_print("TensorBoard 초기화 완료")

    log_print("모델 생성 중...")
    model = generate_model(opt, device)
    log_print("모델 생성 완료")
    log_print(f"현재 백본: {model.backbone_name}")

    train_loader, val_loader = get_loaders(opt)

    ratio, num_params, num_samples = data_to_param_ratio(model, train_loader)
    log_print(f"훈련 샘플 수: {num_samples}")
    log_print(f"파라미터 총 개수: {num_params}")
    log_print(f"파라미터 대비 데이터 수 비율: {ratio:.2f} (파라미터 개수 / 데이터 샘플 수)")

    optimizer = optim.Adam(model.parameters(), lr=opt.lr_rate, weight_decay=opt.weight_decay)
    log_print(f"옵티마이저 설정 완료: Adam (lr={opt.lr_rate}, weight_decay={opt.weight_decay})")

    criterion = nn.CrossEntropyLoss()
    log_print("손실함수 설정 완료: CrossEntropyLoss")

    if opt.resume_path:
        log_print(f"모델 복원 중: {opt.resume_path}")
        start_epoch = resume_model(opt, model, optimizer)
    else:
        start_epoch = 1
        log_print("새로운 모델로 학습 시작")

    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = start_epoch - 1
    last_metrics = {}
    last_epoch = start_epoch - 1

    log_print("\n" + "="*60)
    log_print("학습 시작!")
    log_print("="*60)

    for epoch in range(start_epoch, opt.n_epochs + 1):
        epoch_start_time = time.time()
        log_print(f"\n[Epoch {epoch}/{opt.n_epochs}] 시작")
        log_print("-" * 50)
        log_print("Training 단계 시작...")
        train_start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)

        train_elapsed = time.time() - train_start_time
        log_print(f"Training 완료!")
        log_print(f"   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        log_print(f"   소요시간: {train_elapsed:.1f}초")

        log_print("\nValidation 단계 시작...")
        log_print("   이 과정은 배치 크기에 따라 시간이 달라집니다.")
        val_start_time = time.time()

        try:
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            val_elapsed = time.time() - val_start_time
            log_print(f"Validation 완료!")
            log_print(f"   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
            log_print(f"   소요시간: {val_elapsed:.1f}초")
        except KeyboardInterrupt:
            log_print("사용자가 학습을 중단했습니다.")
            break
        except Exception as e:
            log_print(f"Validation 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            log_print("오류가 발생했지만 학습을 계속 진행합니다...")
            val_loss, val_acc = 0.0, 0.0

        epoch_elapsed = time.time() - epoch_start_time
        log_print(f"\n[Epoch {epoch}] 완료!")
        log_print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        log_print(f"   Valid: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        log_print(f"   총 소요시간: {epoch_elapsed:.1f}초")

        summary_writer.add_scalar('losses/train_loss', train_loss, epoch)
        summary_writer.add_scalar('losses/val_loss', val_loss, epoch)
        summary_writer.add_scalar('accuracy/train_acc', train_acc * 100, epoch)
        summary_writer.add_scalar('accuracy/val_acc', val_acc * 100, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        last_epoch = epoch

        if (epoch) % opt.save_interval == 0:
            log_print(f"\n모델 저장 중... (Epoch {epoch})")
            snapshot_dir = getattr(opt, 'snapshot_dir', 'snapshots')
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)
                log_print(f"{snapshot_dir} 디렉토리 생성")

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            model_filename = f'{opt.model}-{model.backbone_name}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth'
            model_path = os.path.join(snapshot_dir, model_filename)
            try:
                torch.save(state, model_path)
                log_print(f"모델 저장 완료: {model_path}")
                file_size = os.path.getsize(model_path) / (1024*1024)
                log_print(f"파일 크기: {file_size:.1f}MB")
                log_print(f"Epoch {epoch} model saved!\n")
            except Exception as e:
                log_print(f"모델 저장 실패: {e}")

        log_print("-" * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            log_print(f"\nEarly stopping! {early_stopping.patience} 에포크 동안 Validation loss 개선이 없었습니다.")
            break

        # 혼동 행렬, F1 score, Precision, Recall 계산 및 출력
        train_cm, train_f1, train_precision, train_recall = compute_metrics(model, train_loader, device)
        val_cm, val_f1, val_precision, val_recall = compute_metrics(model, val_loader, device)
        train_accuracy = calculate_accuracy_from_cm(train_cm)
        val_accuracy = calculate_accuracy_from_cm(val_cm)

        log_print(f"Epoch {epoch} - Train F1 Score: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
        log_print(f"Epoch {epoch} - Validation F1 Score: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
        log_print(f"Epoch {epoch} - Train Confusion Matrix:\n{train_cm}")
        log_print(f"Epoch {epoch} - Validation Confusion Matrix:\n{val_cm}")

        last_metrics = {
            'train': {
                'f1': float(train_f1),
                'accuracy': float(train_accuracy),
                'cm': train_cm
            },
            'val': {
                'f1': float(val_f1),
                'accuracy': float(val_accuracy),
                'cm': val_cm
            }
        }

    summary_writer.close()

    loss_curve_path = f"loss_curve_{run_name}.png"
    plot_loss(train_losses, val_losses, loss_curve_path)
    log_print(f"학습 종료 후 Loss 곡선 그래프 저장: {loss_curve_path}")

    snapshot_dir = getattr(opt, 'snapshot_dir', 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    save_final_artifacts(opt, model, optimizer, best_model_state, last_metrics, run_name, snapshot_dir, last_epoch, best_epoch)

    log_print("\n전체 학습 완료!")
    log_print("="*60)


def prepare_run_options(base_opt, positive_class):
    run_opt = copy.deepcopy(base_opt)
    dataset_dir = Path(run_opt.annotation_path).parent if run_opt.annotation_path else Path('.')
    annotation_file = dataset_dir / f"annotation_{positive_class}.json"
    if not annotation_file.exists():
        log_print(f"[경고] 어노테이션 파일을 찾을 수 없습니다: {annotation_file}. {positive_class} 학습을 건너뜁니다.")
        return None

    run_opt.annotation_path = str(annotation_file)
    run_opt.selected_classes = [positive_class, 'no-accident']
    run_opt.train_all_incidents = False
    run_opt.snapshot_dir = os.path.join('snapshots', positive_class)
    run_opt.tensorboard_dir = os.path.join('tf_logs', positive_class)
    if hasattr(run_opt, 'resume_path'):
        run_opt.resume_path = None
    return run_opt


def infer_positive_class(annotation_path):
    if not annotation_path:
        return None
    name = Path(annotation_path).stem  # e.g., annotation_jam
    if name.startswith('annotation_'):
        return name.split('annotation_')[-1]
    return None


def run_sequential_training(opt):
    base_dir = Path(opt.annotation_path).parent if opt.annotation_path else Path('datasets')
    if not base_dir.exists():
        log_print(f"[경고] 어노테이션 디렉토리를 찾을 수 없습니다: {base_dir}")
        return

    for positive_class in INCIDENT_CLASSES:
        run_opt = prepare_run_options(opt, positive_class)
        if run_opt is None:
            continue
        run_single_training(run_opt, run_label=positive_class)


def main_worker():
    opt = parse_opts()

    if getattr(opt, 'train_all_incidents', False):
        log_print("[INFO] train_all_incidents 옵션이 활성화되었습니다. 5개 사고 유형을 순차 학습합니다.")
        run_sequential_training(opt)
    else:
        if not hasattr(opt, 'snapshot_dir'):
            opt.snapshot_dir = 'snapshots'
        if not hasattr(opt, 'tensorboard_dir'):
            opt.tensorboard_dir = 'tf_logs'
        run_single_training(opt)


if __name__ == "__main__":
    main_worker()


def calculate_accuracy_from_cm(cm):
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def extract_binary_counts(cm):
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return int(tp), int(tn), int(fp), int(fn)
    else:
        return None, None, None, None


def save_metrics_csv(csv_path, run_name, metrics):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['run', 'split', 'f1_score', 'accuracy', 'tp', 'tn', 'fp', 'fn']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for split, info in metrics.items():
            tp, tn, fp, fn = extract_binary_counts(info['cm'])
            writer.writerow({
                'run': run_name,
                'split': split,
                'f1_score': info['f1'],
                'accuracy': info['accuracy'],
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })


def save_final_artifacts(opt, model, optimizer, best_model_state, metrics, run_name, snapshot_dir, last_epoch, best_epoch):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_filename = f"{opt.model}-{model.backbone_name}-{run_name}-{timestamp}"

    checkpoint_path = os.path.join(snapshot_dir, f"{base_filename}-checkpoint.pth")
    weights_path = os.path.join(snapshot_dir, f"{base_filename}-weights.pth")
    weights_hparams_path = os.path.join(snapshot_dir, f"{base_filename}-weights-hparams.pth")
    best_model_path = os.path.join(snapshot_dir, f"{opt.model}-{model.backbone_name}-{run_name}-best.pth")
    metrics_csv_path = os.path.join(snapshot_dir, f"metrics_{run_name}.csv")

    checkpoint_state = {
        'epoch': last_epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': vars(opt)
    }
    try:
        torch.save(checkpoint_state, checkpoint_path)
        log_print(f"최종 체크포인트 저장: {checkpoint_path}")
    except Exception as e:
        log_print(f"체크포인트 저장 실패: {e}")

    try:
        torch.save(model.state_dict(), weights_path)
        log_print(f"순수 가중치 저장: {weights_path}")
    except Exception as e:
        log_print(f"순수 가중치 저장 실패: {e}")

    try:
        torch.save({'state_dict': model.state_dict(), 'hyperparameters': vars(opt)}, weights_hparams_path)
        log_print(f"가중치+하이퍼파라미터 저장: {weights_hparams_path}")
    except Exception as e:
        log_print(f"가중치+하이퍼파라미터 저장 실패: {e}")

    if best_model_state is not None:
        try:
            torch.save(best_model_state, best_model_path)
            log_print(f"베스트 모델 저장 (Epoch {best_epoch}): {best_model_path}")
        except Exception as e:
            log_print(f"베스트 모델 저장 실패: {e}")

    if metrics:
        try:
            save_metrics_csv(metrics_csv_path, run_name, metrics)
            log_print(f"성능 지표 저장: {metrics_csv_path}")
        except Exception as e:
            log_print(f"성능 지표 저장 실패: {e}")
