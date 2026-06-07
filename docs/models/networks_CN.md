# 完整网络架构

[English](networks.md)

本项目支持 128 个完整网络架构（136 注册，合并尺寸变体；123 个标准 + 13 个文本引导），通过 `architecture` 字段直接使用。

## CNN (35)

卷积神经网络系列，经典 UNet 及其变体。

| 名称 | 论文 | 发表 | GitHub | YAML |
|---|---|---|---|---|
| `attention_unet` | Attention U-Net | MIDL 2018 | [ozan-oktay/Attention-Gating-Network](https://github.com/ozan-oktay/Attention-Gating-Network) | [basic](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `unetpp` | UNet++ | DLMIA 2018 | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) | [emcad](../../configs/architectures/combinations/general/unetpp_emcad.yaml) |
| `r2unet` | R2U-Net | IEEE Access 2018 | - | [basic](../../configs/architectures/combinations/general/r2unet_basic.yaml) |
| `multiresunet` | MultiResUNet | Neural Networks 2020 | - | [basic](../../configs/architectures/combinations/general/multiresunet_basic.yaml) |
| `resunet_a` | ResUNet-a | ISPRS 2020 | - | [basic](../../configs/architectures/combinations/general/resunet_a_basic.yaml) |
| `resunetpp` | ResUNet++ | ISM 2019 | - | [basic](../../configs/architectures/combinations/general/resunetpp_basic.yaml) |
| `unet3plus` | UNet 3+ | ICASSP 2020 | [ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version) | [basic](../../configs/architectures/combinations/general/unet3plus_basic.yaml) |
| `denseunet` | DenseUNet | - | - | [basic](../../configs/architectures/combinations/general/denseunet_basic.yaml) |
| `scseunet` | scSE-UNet (Squeeze-Excitation) | MICCAI 2018 | - | [basic](../../configs/architectures/combinations/general/scseunet_basic.yaml) |
| `sa_unet` | SA-UNet (Spatial Attention) | IEEE TIM 2021 | - | [basic](../../configs/architectures/combinations/general/sa_unet_basic.yaml) |
| `kiunet` | KiU-Net | MICCAI 2020 | [jeya-maria-jose/KiU-Net-pytorch](https://github.com/jeya-maria-jose/KiU-Net-pytorch) | [basic](../../configs/architectures/combinations/general/kiunet_basic.yaml) |
| `pan` | PAN (Pyramid Attention Network) | BMVC 2018 | - | [basic](../../configs/architectures/combinations/general/pan_basic.yaml) |
| `linknet` | LinkNet | VCIP 2017 | - | [basic](../../configs/architectures/combinations/general/linknet_basic.yaml) |
| `pspnet` | PSPNet | CVPR 2017 | - | [basic](../../configs/architectures/combinations/general/pspnet_basic.yaml) |
| `fr_unet` | FR-UNet (Full-Resolution) | IEEE TMI 2022 | - | [basic](../../configs/architectures/combinations/general/fr_unet_basic.yaml) |
| `dcsaunet` | DCSAU-Net | Computers in Biology and Medicine 2023 | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) | [basic](../../configs/architectures/combinations/general/dcsaunet_basic.yaml) |
| `cfanet` | CFA-Net | Computers in Biology and Medicine 2024 | [ZhangJD-ong/CFA-Net](https://github.com/ZhangJD-ong/CFA-Net) | [basic](../../configs/architectures/combinations/general/cfanet_basic.yaml) |
| `mednext` | MedNeXt | MICCAI 2023 | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) | [emcad](../../configs/architectures/combinations/general/mednext_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/mednext_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `nnunet_2d` | nnU-Net (2D) | Nature Methods 2021 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | [basic](../../configs/architectures/combinations/general/nnunet_2d_basic.yaml) |
| `acc_unet` | ACC-UNet | MICCAI 2023 | - | [basic](../../configs/architectures/combinations/general/acc_unet_basic.yaml) |
| `cmunext` | CMUNeXt | arXiv 2023 | - | [basic](../../configs/architectures/combinations/general/cmunext_basic.yaml) |
| `mew_unet` | MEW-UNet | arXiv 2024 | - | [basic](../../configs/architectures/combinations/general/mew_unet_basic.yaml) |
| `lv_unet` | LV-UNet (Lightweight) | - | - | [basic](../../configs/architectures/combinations/general/lv_unet_basic.yaml) |
| `ege_unet` | EGE-UNet | arXiv 2023 | [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) | [basic](../../configs/architectures/combinations/general/ege_unet_basic.yaml) |
| `malunet` | MALUNet | arXiv 2022 | - | [basic](../../configs/architectures/combinations/general/malunet_basic.yaml) |
| `lite_unet` | Lite-UNet | - | - | [basic](../../configs/architectures/combinations/general/lite_unet_basic.yaml) |
| `mk_unet` | MK-UNet | - | - | [basic](../../configs/architectures/combinations/general/mk_unet_basic.yaml) |
| `u_lite` | U-Lite | arXiv 2022 | - | [basic](../../configs/architectures/combinations/general/u_lite_basic.yaml) |
| `aau_net` | AAU-Net | IEEE JBHI 2023 | [CGPxy/AAU-net](https://github.com/CGPxy/AAU-net) | [basic](../../configs/architectures/combinations/general/aau_net_basic.yaml) |
| `cmu_net` | CMU-Net | Bioinformatics 2024 | - | [basic](../../configs/architectures/combinations/general/cmu_net_basic.yaml) |
| `dscnet` | DSCNet | MICCAI 2023 | - | [basic](../../configs/architectures/combinations/general/dscnet_basic.yaml) |
| `dconnnet` | DconnNet | MICCAI 2023 | - | [basic](../../configs/architectures/combinations/general/dconnnet_basic.yaml) |
| `stu_net` | STU-Net | arXiv 2023 | - | [basic](../../configs/architectures/combinations/general/stu_net_basic.yaml) |
| `polyper` | Polyper | - | - | [basic](../../configs/architectures/combinations/general/polyper_basic.yaml) |
| `hovernet_lite` | HoverNet Lite | - | - | [basic](../../configs/architectures/combinations/general/hovernet_lite_basic.yaml) |

## Transformer (35)

基于 Transformer 的分割网络。

| 名称 | 论文 | 发表 | GitHub | YAML |
|---|---|---|---|---|
| `transunet` | TransUNet | arXiv 2021 | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | [cascade_full](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `swinunet` | Swin-UNet | ECCV 2022 | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | [segformer](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `medt` | MedT (Medical Transformer) | MICCAI 2021 | [jeya-maria-jose/Medical-Transformer](https://github.com/jeya-maria-jose/Medical-Transformer) | [basic](../../configs/architectures/combinations/general/medt_basic.yaml) |
| `daeformer` | DAEFormer | ICLR 2023 | - | [emcad](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `missformer` | MISSFormer | IEEE TMI 2022 | - | [basic](../../configs/architectures/combinations/general/missformer_basic.yaml) |
| `h2former` | H2Former | IEEE TMI 2023 | - | [basic](../../configs/architectures/combinations/general/h2former_basic.yaml) |
| `hiformer` | HiFormer | WACV 2023 | - | [cascade](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `mctrans` | MCTrans | MICCAI 2021 | - | [cascade_emcad](../../configs/architectures/combinations/general/mctrans_cascade_emcad.yaml) |
| `mtunet` | MT-UNet | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/mtunet_basic.yaml) |
| `scaleformer` | ScaleFormer | MICCAI 2022 | - | [cascade_full](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `fatnet` | FAT-Net | IEEE TMI 2022 | - | [basic](../../configs/architectures/combinations/general/fatnet_basic.yaml) |
| `nnformer_2d` | nnFormer (2D) | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/nnformer_2d_basic.yaml) |
| `transfuse` | TransFuse | MICCAI 2021 | - | [basic](../../configs/architectures/combinations/general/transfuse_basic.yaml) |
| `levit_unet` | LeViT-UNet | ML4H 2022 | - | [basic](../../configs/architectures/combinations/general/levit_unet_basic.yaml) |
| `transatt_unet` | TransAttUNet | arXiv 2022 | - | [basic](../../configs/architectures/combinations/general/transatt_unet_basic.yaml) |
| `da_transunet` | DA-TransUNet | arXiv 2023 | - | [basic](../../configs/architectures/combinations/general/da_transunet_basic.yaml) |
| `ds_transunet` | DS-TransUNet | arXiv 2022 | - | [basic](../../configs/architectures/combinations/general/ds_transunet_basic.yaml) |
| `uctransnet_full` | UCTransNet (full) | AAAI 2022 | - | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |
| `uctransnet_enc` | UCTransNet (encoder-only) | AAAI 2022 | - | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |
| `mobile_u_vit` | Mobile-UViT | - | - | [basic](../../configs/architectures/combinations/general/mobile_u_vit_basic.yaml) |
| `cswin_unet` | CSWin-UNet | - | - | [basic](../../configs/architectures/combinations/general/cswin_unet_basic.yaml) |
| `fcbformer` | FCBFormer | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/fcbformer_basic.yaml) |
| `pvt_unet` | PVT-UNet | - | - | [emcad](../../configs/architectures/combinations/general/pvtv2_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/pvtv2_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/pvtv2_cfm.yaml) |
| `transnetr` | TransNetR | IEEE Access 2023 | - | [basic](../../configs/architectures/combinations/general/transnetr_basic.yaml) |
| `polyp_pvt` | Polyp-PVT | MICCAI 2021 | - | [basic](../../configs/architectures/combinations/general/polyp_pvt_basic.yaml) |
| `cascade` | CASCADE | MICCAI 2023 | - | [resnet34](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `hsnet` | HSNet | MedIA 2023 | - | [basic](../../configs/architectures/combinations/general/hsnet_basic.yaml) |
| `ssformer` | SSFormer | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/ssformer_basic.yaml) |
| `ldnet` | LDNet | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/ldnet_basic.yaml) |
| `esfpnet` | ESFPNet | MICCAI 2022 | - | [basic](../../configs/architectures/combinations/general/esfpnet_basic.yaml) |
| `mist` | MIST | IEEE TMI 2023 | - | [basic](../../configs/architectures/combinations/general/mist_basic.yaml) |
| `double_unet` | DoubleU-Net | CBMS 2020 | - | [basic](../../configs/architectures/combinations/general/double_unet_basic.yaml) |
| `sepnet` | SEPNet | - | - | [basic](../../configs/architectures/combinations/general/sepnet_basic.yaml) |
| `ctnet` | CTNet | - | - | [basic](../../configs/architectures/combinations/general/ctnet_basic.yaml) |
| `nulite` | NuLite | - | - | [basic](../../configs/architectures/combinations/general/nulite_basic.yaml) |

## Mamba / SSM (24)

基于 Mamba (Selective State Space Model) 的网络。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `mamba_unet` | Mamba-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/mamba_unet_basic.yaml) |
| `h_vmunet` | H-vmunet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/h_vmunet_basic.yaml) |
| `lightm_unet` | LightM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/lightm_unet_basic.yaml) |
| `swin_umamba` | Swin-UMamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/swin_umamba_basic.yaml) |
| `umamba_bot` | U-Mamba (bottleneck) | arXiv 2024 | [cascade_full](../../configs/architectures/combinations/general/umamba_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/umamba_cfm.yaml), [emcad](../../configs/architectures/combinations/general/umamba_emcad.yaml) |
| `umamba_enc` | U-Mamba (encoder) | arXiv 2024 | [cascade_full](../../configs/architectures/combinations/general/umamba_cascade_full.yaml) |
| `ultralight_vmunet` | UltraLight VM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/ultralight_vmunet_basic.yaml) |
| `vm_unet` | VM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/vm_unet_basic.yaml) |
| `vm_unet_v2` | VM-UNet V2 | arXiv 2024 | [basic](../../configs/architectures/combinations/general/vm_unet_v2_basic.yaml) |
| `lkm_unet` | LKM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/lkm_unet_basic.yaml) |
| `log_vmamba` | LoG-VMamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/log_vmamba_basic.yaml) |
| `vmkla_unet` | VMKLA-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/vmkla_unet_basic.yaml) |
| `ultralbm_unet` | UltraLBM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/ultralbm_unet_basic.yaml) |
| `nnmamba_2d` | nnMamba (2D) | arXiv 2024 | [basic](../../configs/architectures/combinations/general/nnmamba_2d_basic.yaml) |
| `polyp_mamba` | Polyp-Mamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/polyp_mamba_basic.yaml) |
| `hc_mamba` | HC-Mamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/hc_mamba_basic.yaml) |
| `ac_mambaseg` | AC-MambaSeg | arXiv 2024 | [basic](../../configs/architectures/combinations/general/ac_mambaseg_basic.yaml) |
| `dcm_net` | DCM-Net | arXiv 2024 | [basic](../../configs/architectures/combinations/general/dcm_net_basic.yaml) |
| `dermomamba` | DermoMamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/dermomamba_basic.yaml) |
| `mucm_net` | MUCM-Net | arXiv 2024 | [basic](../../configs/architectures/combinations/general/mucm_net_basic.yaml) |
| `serp_mamba` | Serp-Mamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/serp_mamba_basic.yaml) |
| `skin_mamba` | SkinMamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/skin_mamba_basic.yaml) |
| `mamba_vesselnet_pp` | Mamba-VesselNet++ | arXiv 2024 | [basic](../../configs/architectures/combinations/general/mamba_vesselnet_pp_basic.yaml) |
| `vim_unet` | ViM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/vim_unet_basic.yaml) |
| `uu_mamba` | UU-Mamba | arXiv 2024 | [basic](../../configs/architectures/combinations/general/uu_mamba_basic.yaml) |

## SAM (10)

基于 Segment Anything Model 的网络。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `sam_b` | SAM ViT-Base | ICCV 2023 | [cascade_full](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/sam_vit_cfm.yaml), [emcad](../../configs/architectures/combinations/general/sam_vit_emcad.yaml) |
| `sam_l` | SAM ViT-Large | ICCV 2023 | [cascade_full](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml) |
| `mobile_sam` | MobileSAM | arXiv 2023 | [basic](../../configs/architectures/combinations/general/mobile_sam_basic.yaml) |
| `sam2` | SAM 2 | arXiv 2024 | [basic](../../configs/architectures/combinations/general/sam2_basic.yaml) |
| `medsam` | MedSAM | Nature Comms 2024 | [emcad](../../configs/architectures/combinations/general/medsam_encoder_emcad.yaml) |
| `samus` | SAMUS | arXiv 2023 | [basic](../../configs/architectures/combinations/general/samus_basic.yaml) |
| `sam_med2d` | SAM-Med2D | arXiv 2023 | [basic](../../configs/architectures/combinations/general/sam_med2d_basic.yaml) |
| `sammed2d_wrapper` | SAMMed2D (wrapper) | arXiv 2023 | [basic](../../configs/architectures/combinations/general/sammed2d_wrapper_basic.yaml) |
| `medical_sam_adapter` | Medical SAM Adapter | arXiv 2023 | [basic](../../configs/architectures/combinations/general/medical_sam_adapter_basic.yaml) |
| `samed` | SAMed | arXiv 2023 | [basic](../../configs/architectures/combinations/general/samed_basic.yaml) |
| `auto_sam` | AutoSAM | arXiv 2023 | [basic](../../configs/architectures/combinations/general/auto_sam_basic.yaml) |
| `lite_medsam` | Lite-MedSAM | arXiv 2024 | [basic](../../configs/architectures/combinations/general/lite_medsam_basic.yaml) |

## KAN / MLP (4)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `ukan` | U-KAN | arXiv 2024 | [basic](../../configs/architectures/combinations/general/ukan_basic.yaml) |
| `wav_kan_unet` | Wav-KAN UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/wav_kan_unet_basic.yaml) |
| `unext` | UNeXt | MICCAI 2022 | [basic](../../configs/architectures/combinations/general/unext_basic.yaml) |
| `rolling_unet` | Rolling-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/rolling_unet_basic.yaml) |
| `rolling_unet_s` | Rolling-UNet (small) | arXiv 2024 | [basic](../../configs/architectures/combinations/general/rolling_unet_s_basic.yaml) |
| `rolling_unet_m` | Rolling-UNet (medium) | arXiv 2024 | [basic](../../configs/architectures/combinations/general/rolling_unet_m_basic.yaml) |
| `rolling_unet_l` | Rolling-UNet (large) | arXiv 2024 | [basic](../../configs/architectures/combinations/general/rolling_unet_l_basic.yaml) |

## RWKV (4)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `u_rwkv` | U-RWKV | arXiv 2024 | [unet](../../configs/architectures/combinations/general/rwkv_unet.yaml), [small](../../configs/architectures/combinations/general/rwkv_unet_small.yaml), [tiny](../../configs/architectures/combinations/general/rwkv_unet_tiny.yaml) |
| `rwkv_unet` | RWKV-UNet | arXiv 2024 | [emcad](../../configs/architectures/combinations/general/rwkv_emcad.yaml), [cascade_full](../../configs/architectures/combinations/general/rwkv_cascade_full.yaml), [cfm](../../configs/architectures/combinations/general/rwkv_cfm.yaml) |
| `md_rwkv_unet` | MD-RWKV-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/md_rwkv_unet_basic.yaml) |
| `rir_zigzag` | RIR-Zigzag | arXiv 2024 | [yaml](../../configs/architectures/combinations/general/rir_zigzag.yaml) |

## Linear Attention (3)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `ttt_unet` | TTT-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/ttt_unet_basic.yaml) |
| `xlstm_unet_bot` / `xlstm_unet_enc` | xLSTM-UNet | arXiv 2024 | [basic](../../configs/architectures/combinations/general/xlstm_unet_basic.yaml) |
| `u_vixlstm` | U-VixLSTM | arXiv 2024 | [basic](../../configs/architectures/combinations/general/u_vixlstm_basic.yaml) |

## 文本引导 (13)

文本引导分割模型，forward 签名为 `(image, text=None)`。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `tganet` | TGANet | MICCAI 2022 | [synapse_clip](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `lvit` | LViT | IEEE TMI 2023 | [mosmed_plus_lvit](../../configs/training_paradigms/text_guided/mosmed_plus_lvit.yaml), [qata_covid19_lvit](../../configs/training_paradigms/text_guided/qata_covid19_lvit.yaml) |
| `languide` | LanGuideMedSeg | MICCAI 2023 | [mosmed_plus_languide](../../configs/training_paradigms/text_guided/mosmed_plus_languide.yaml), [qata_covid19_languide](../../configs/training_paradigms/text_guided/qata_covid19_languide.yaml) |
| `clip_universal` | CLIP-Driven Universal Model | ICCV 2023 | [synapse_clip_large](../../configs/training_paradigms/text_guided/synapse_clip_large.yaml) |
| `cris` | CRIS | CVPR 2022 | [synapse_clip](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `biomedparse` | BiomedParse | Nature Methods 2024 | - |
| `tpro` | TPRO | ECCV 2024 | - |
| `salip` | SaLIP | arXiv 2024 | - |
| `causal_clipseg` | Causal CLIPSeg | arXiv 2024 | - |
| `medclip_sam` | MedCLIP-SAM | arXiv 2024 | [synapse_grounding_dino_medsam](../../configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml) |
| `tp_drseg` | TP-DRSeg | arXiv 2024 | - |
| `cxrclipseg` | CXR-CLIPSeg | arXiv 2024 | - |
| `medisee` | MediSee (MLLM) | arXiv 2024 | [mosmed_plus_medisee](../../configs/training_paradigms/text_guided/mosmed_plus_medisee.yaml), [qata_covid19_medisee](../../configs/training_paradigms/text_guided/qata_covid19_medisee.yaml) |

## YAML 使用示例

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  encoder:
    in_channels: 3
  arch_params: {}

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 16
  num_workers: 4
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01
  scheduler:
    name: cosine
    min_lr: 0.000001
```
