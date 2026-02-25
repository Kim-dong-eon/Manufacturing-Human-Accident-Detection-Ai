import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='CNN-LSTM for AI Hub Fall Detection (UCF101 Format)')

    # 데이터셋 타입 (필수)
    parser.add_argument('--dataset', type=str,
                        default='ucf101',
                        help='dataset type (ucf101 format compatible)')

    # 프로젝트 및 데이터 경로
    parser.add_argument('--root_path', default='./', type=str,
                        help='Project root directory')
    parser.add_argument('--video_path', default=r'I:\Smart\human-accident',
                        type=str, help='Train/Validation image data folder (for training/validation only)')
    parser.add_argument('--annotation_path', default=r'I:\Smart\cnn-lstm-master\datasets\annotation_jam.json',
                        type=str, help='Annotation JSON file for all splits')
    parser.add_argument('--test_video_path', default='./invert_data/video_data/image_data/',
                        type=str, help='Test image frames folder (for testing only)')
    parser.add_argument('--test_mp4_path', default='./invert_data/video_data/mp4/',
                        type=str, help='Test mp4 video folder (for testing only)')

    # 클래스 설정 (변환된 실제 라벨명에 맞게)
    parser.add_argument('--selected_classes', default=['jam', 'no-accident'],
                        type=list, help='Class names')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')

    # 데이터 구분명 (annotation 내 subset 필드와 매칭)
    parser.add_argument('--train_subset', default='training', type=str, help='Training subset name')
    parser.add_argument('--val_subset', default='validation', type=str, help='Validation subset name')
    parser.add_argument('--test_subset', default='testing', type=str, help='Test subset name')

    # 테스트 입력 타입 선택
    parser.add_argument('--test_type', default='image', choices=['image', 'mp4'], type=str,
                        help='Test on image frames or mp4 videos')

    # 모델 구조 및 입력 크기
    parser.add_argument('--model', default='cnnlstm', type=str,
                        help='Model architecture (cnnlstm | cnnlstm_attn)')

    # --- 백본 모델 옵션 추가!
    parser.add_argument('--backbone_name',
                        default='resnet18',
                        type=str,
                        choices=['resnet18', 'resnet101', 'efficientnet_b0'],
                        help='CNN 백본 모델 선택 (resnet18, resnet101, efficientnet_b0)')

    parser.add_argument('--sample_duration', default=20, type=int,
                        help='Number of frames per sample (temporal window)')
    parser.add_argument('--sample_size', default=224, type=int,
                        help='Height and width of input images')

    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)

    # 검증 샘플 개수 옵션
    parser.add_argument('--n_val_samples', default=1, type=int,
                        help='Number of validation samples for each activity')

    # 옵티마이저 및 스케줄러
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dampening', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_patience', default=10, type=int)

    # 데이터 로딩 관련
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--norm_value', default=1, type=int)
    parser.add_argument('--no_mean_norm', action='store_true')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--mean_dataset', default='activitynet', type=str)

    # 모델 저장/로깅
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--pretrain_path', default='', type=str)
    parser.add_argument('--log_interval', default=5, type=int)

    # GPU 설정
    parser.add_argument('--use_cuda', action='store_true')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--gpu', default=0, type=int)

    # 기타 옵션
    parser.add_argument('--early_stopping_patience', default=15, type=int)
    parser.add_argument('--save_best_only', action='store_true')
    parser.set_defaults(save_best_only=True)
    parser.add_argument('--class_weights', action='store_true')
    parser.set_defaults(class_weights=False)
    parser.add_argument('--data_augmentation', action='store_true')
    parser.set_defaults(data_augmentation=True)
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate model (no training)')
    parser.set_defaults(eval_only=False)
    parser.add_argument('--train_all_incidents', action='store_true',
                        help='bump, fall-down, fall-off, hit, jam 순차 학습 실행')
    parser.set_defaults(train_all_incidents=False)
    parser.add_argument('--use_grayscale', action='store_true',
                        help='입력 이미지를 그레이스케일(RGB 변환 포함)로 사용')
    parser.set_defaults(use_grayscale=False)

    args = parser.parse_args()

    # 옵션 출력
    print("\n[옵션 설정 확인]")
    print(f" - 데이터셋 타입: {args.dataset}")
    print(f" - 프로젝트 루트: {args.root_path}")
    print(f" - 학습/검증 이미지 데이터: {args.video_path}")
    print(f" - 테스트 이미지 데이터: {args.test_video_path}")
    print(f" - 테스트 mp4 데이터: {args.test_mp4_path}")
    print(f" - annotation JSON: {args.annotation_path}")
    print(f" - 클래스: {args.selected_classes} (총 {args.n_classes}개)")
    print(f" - 학습/검증/테스트 split명: {args.train_subset} / {args.val_subset} / {args.test_subset}")
    print(f" - 테스트 입력 타입: {args.test_type}")
    print(f" - 모델: {args.model}, 백본: {args.backbone_name}, 입력 크기: {args.sample_size}x{args.sample_size}, 시퀀스 길이: {args.sample_duration}")
    print(f" - 배치 크기: {args.batch_size}, epochs: {args.n_epochs}, lr: {args.lr_rate}")
    print(f" - 검증용 샘플 수: {args.n_val_samples}")
    print(f" - GPU 사용: {args.use_cuda}, GPU 번호: {args.gpu}")
    print(f" - train_all_incidents: {args.train_all_incidents}")
    print(f" - use_grayscale: {args.use_grayscale}")

    return args
