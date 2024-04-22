import super_gradients
from super_gradients import init_trainer, Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models, dataloaders
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad
import super_gradients

# from super_gradients.training.datasets import CWZCustomDetectionDataset

def main():
    init_trainer()
    setup_device(device=None, multi_gpu="Off", num_gpus=1)

    trainer = Trainer(experiment_name="cwz_headnobg_yolo_nas_s", ckpt_root_dir="None")

    num_classes = 2
    arch_params = {
        "in_channels": 3,
        "backbone": {
            "NStageBackbone": {
                "stem": {"YoloNASStem": {"out_channels": 16}},
                "stages": [
                    {"YoloNASStage": {"out_channels": 32, "num_blocks": 2, "activation_type": "relu", "hidden_channels": 32, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 64, "num_blocks": 3, "activation_type": "relu", "hidden_channels": 24, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 128, "num_blocks": 4, "activation_type": "relu", "hidden_channels": 32, "concat_intermediates": False}},
                    {"YoloNASStage": {"out_channels": 256, "num_blocks": 2, "activation_type": "relu", "hidden_channels": 64, "concat_intermediates": False}},
                ],
                "context_module": {"SPP": {"output_channels": 256, "activation_type": "relu", "k": [5, 9, 13]}},
                "out_layers": ["stage1", "stage2", "stage3", "context_module"],
            }
        },
        "neck": {
            "YoloNASPANNeckWithC2": {
                "neck1": {
                    "YoloNASUpStage": {
                        "out_channels": 64,
                        "num_blocks": 2,
                        "hidden_channels": 24,
                        "width_mult": 1,
                        "depth_mult": 1,
                        "activation_type": "relu",
                        "reduce_channels": True,
                    }
                },
                "neck2": {
                    "YoloNASUpStage": {
                        "out_channels": 32,
                        "num_blocks": 2,
                        "hidden_channels": 16,
                        "width_mult": 1,
                        "depth_mult": 1,
                        "activation_type": "relu",
                        "reduce_channels": True,
                    }
                },
                "neck3": {
                    "YoloNASDownStage": {
                        "out_channels": 64,
                        "num_blocks": 2,
                        "hidden_channels": 24,
                        "activation_type": "relu",
                        "width_mult": 1,
                        "depth_mult": 1,
                    }
                },
                "neck4": {
                    "YoloNASDownStage": {
                        "out_channels": 128,
                        "num_blocks": 2,
                        "hidden_channels": 24,
                        "activation_type": "relu",
                        "width_mult": 1,
                        "depth_mult": 1,
                    }
                },
            }
        },
        "heads": {
            "NDFLHeads": {
                "num_classes": 2,
                "reg_max": 16,
                "heads_list": [
                    {"YoloNASDFLHead": {"inter_channels": 64, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 8}},
                    {"YoloNASDFLHead": {"inter_channels": 128, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 16}},
                    {"YoloNASDFLHead": {"inter_channels": 256, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 32}},
                ],
            }
        },
        "bn_eps": 0.001,
        "bn_momentum": 0.03,
        "inplace_act": True,
        "_convert_": "all",
        "num_classes": 2,
    }

    model = models.get(
        model_name="yolo_nas_s",
        num_classes=num_classes,
        arch_params=arch_params,
        strict_load=StrictLoad.NO_KEY_MATCHING,
        pretrained_weights=None,
        checkpoint_path=None,
        load_backbone=False,
        checkpoint_num_classes=None,
    )

    train_dataloader = dataloaders.get(
        name=None,
        dataset_params={
            "dataset_filenames_txts": ["/home/cwz/Lenovo/HeadBody/crowdhuman_train_3989.txt"],
            "input_dim": [640, 640],
            "cache_annotations": True,
            "ignore_empty_annotations": False,
            "transforms": [
                {"DetectionMosaic": {"input_dim": [640, 640], "prob": 1.0}},
                {"DetectionHSV": {"prob": 1.0, "hgain": 5, "sgain": 30, "vgain": 30}},
                {"DetectionHorizontalFlip": {"prob": 0.5}},
                {"DetectionPaddedRescale": {"input_dim": [640, 640], "pad_value": 114}},
                {"DetectionStandardize": {"max_value": 255.0}},
                {"DetectionTargetsFormatTransform": {"output_format": "LABEL_CXCYWH"}},
            ],
        },
        dataloader_params={
            "dataset": "CWZCustomDetectionDataset",
            "batch_size": 32,
            "num_workers": 8,
            "drop_last": True,
            "shuffle": True,
            "pin_memory": True,
            "collate_fn": "DetectionCollateFN",
        },
    )
    
    train_dataset_params = {
            "dataset_filenames_txts": ["/home/cwz/Lenovo/HeadBody/crowdhuman_train_320.txt"],
            "input_dim": [640, 640],
            "cache_annotations": True,
            "ignore_empty_annotations": False,
            "transforms": [
                {"DetectionMosaic": {"input_dim": [640, 640], "prob": 1.0}},
                {"DetectionHSV": {"prob": 1.0, "hgain": 5, "sgain": 30, "vgain": 30}},
                {"DetectionHorizontalFlip": {"prob": 0.5}},
                {"DetectionPaddedRescale": {"input_dim": [640, 640], "pad_value": 114}},
                {"DetectionStandardize": {"max_value": 255.0}},
                {"DetectionTargetsFormatTransform": {"output_format": "LABEL_CXCYWH"}},
            ],
        }
    # trainset = CWZCustomDetectionDataset(**train_dataset_params)
    train_dataloader.dataset.plot(1, 4)

    val_dataloader = dataloaders.get(
        name=None,
        dataset_params={
            "dataset_filenames_txts": ["/home/cwz/Lenovo/HeadBody/crowdhuman_val_1235.txt"],
            "input_dim": [636, 636],
            "cache_annotations": True,
            "ignore_empty_annotations": False,
            "transforms": [
                {"DetectionPadToSize": {"output_size": [640, 640], "pad_value": 114}},
                {"DetectionStandardize": {"max_value": 255.0}},
                {"DetectionTargetsFormatTransform": {"input_dim": [640, 640], "output_format": "LABEL_CXCYWH"}},
            ],
        },
        dataloader_params={
            "dataset": "CWZCustomDetectionDataset",
            "batch_size": 32,
            "num_workers": 8,
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": "DetectionCollateFN",
        },
    )

    _lr_updates = super_gradients.training.utils.utils.empty_list()

    _valid_metrics_list_0_detectionmetrics_post_prediction_callback = super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback(
        score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7
    )

    training_hyperparams = {
        "resume": False,
        "run_id": None,
        "resume_path": None,
        "resume_from_remote_sg_logger": False,
        "ckpt_name": "ckpt_latest.pth",
        "lr_mode": "CosineLRScheduler",
        "lr_schedule_function": None,
        "lr_warmup_epochs": 0,
        "lr_warmup_steps": 100,
        "lr_cooldown_epochs": 0,
        "warmup_initial_lr": 1e-06,
        "step_lr_update_freq": None,
        "cosine_final_lr_ratio": 0.1,
        "warmup_mode": "LinearBatchLRWarmup",
        "lr_updates": _lr_updates,
        "pre_prediction_callback": None,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 1e-05},
        "load_opt_params": True,
        "zero_weight_decay_on_bias_and_bn": True,
        "loss": "PPYoloELoss",
        "criterion_params": {"use_static_assigner": False, "num_classes": 2},
        "ema": True,
        "ema_params": {"decay": 0.9997, "decay_type": "threshold", "beta": 15},
        "train_metrics_list": [],
        "valid_metrics_list": [
            {
                "DetectionMetrics": {
                    "score_thres": 0.1,
                    "top_k_predictions": 300,
                    "num_cls": 2,
                    "normalize_targets": True,
                    "post_prediction_callback": _valid_metrics_list_0_detectionmetrics_post_prediction_callback,
                }
            }
        ],
        "metric_to_watch": "mAP@0.50:0.95",
        "greater_metric_to_watch_is_better": True,
        "launch_tensorboard": False,
        "tensorboard_port": None,
        "tb_files_user_prompt": False,
        "save_tensorboard_to_s3": False,
        "precise_bn": False,
        "precise_bn_batch_size": None,
        "sync_bn": True,
        "silent_mode": False,
        "mixed_precision": False,
        "save_ckpt_epoch_list": [5, 10, 15, 20, 25, 30, 50, 100, 200, 250],
        "average_best_models": True,
        "dataset_statistics": False,
        "batch_accumulate": 1,
        "run_validation_freq": 1,
        "run_test_freq": 1,
        "save_model": True,
        "seed": 42,
        "phase_callbacks": [],
        "log_installed_packages": True,
        "clip_grad_norm": None,
        "ckpt_best_name": "ckpt_best.pth",
        "max_train_batches": None,
        "max_valid_batches": None,
        "sg_logger": "base_sg_logger",
        "sg_logger_params": {
            "tb_files_user_prompt": False,
            "launch_tensorboard": False,
            "tensorboard_port": None,
            "save_checkpoints_remote": False,
            "save_tensorboard_remote": False,
            "save_logs_remote": False,
            "monitor_system": True,
        },
        "torch_compile": False,
        "torch_compile_loss": False,
        "torch_compile_options": {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False, "backend": "inductor", "options": None, "disable": False},
        "finetune": False,
        "_convert_": "all",
        "max_epochs": 300,
        "initial_lr": 0.0002,
    }

    # TRAIN
    result = trainer.train(
        model=model,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        training_params=training_hyperparams,
    )

    print(result)


if __name__ == "__main__":
    main()
