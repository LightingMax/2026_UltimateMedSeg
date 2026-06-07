# Decoders

[中文文档](decoders_CN.md)

This project provides 40 decoder modules, grouped by category as follows.

## Basic (4)

Basic upsampling decoders.

| Name | Description | YAML |
|---|---|---|
| `unet` | Standard UNet decoder with conv + upsample | [unet_basic](../../configs/architectures/combinations/general/unet_basic.yaml) |
| `bilinear` | Bilinear interpolation upsampling | [bilinear](../../configs/architectures/combinations/general/bilinear_basic.yaml) |
| `deconv` | Transposed convolution upsampling | [deconv](../../configs/architectures/combinations/general/deconv_resnet34.yaml) |
| `dw_sep` | Depthwise separable convolution decoder | [dw_sep](../../configs/architectures/combinations/general/dwsep_resnet34.yaml) |

## Dense (2)

Dense connection decoders.

| Name | Description | YAML |
|---|---|---|
| `unetpp` | UNet++ dense nested decoder | [unetpp](../../configs/architectures/combinations/general/unetpp_emcad.yaml) |
| `unet3plus` | UNet 3+ full-scale skip connection decoder | [unet3plus](../../configs/architectures/combinations/general/unet3plus_basic.yaml) |

## Cascade (10)

Cascade decoders that progressively refine segmentation.

| Name | Description | YAML |
|---|---|---|
| `cascade` | CASCADE decoder | [cascade_resnet34](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `cascade_full` | CASCADE full decoder | [transunet_cascade_full](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `cascade_emcad` | CASCADE + EMCAD hybrid | [mednext_cascade_emcad](../../configs/architectures/combinations/general/mednext_cascade_emcad.yaml) |
| `cfm` | Cascaded Feature Merging | [mednext_cfm](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `emcad` | Efficient Multi-scale Cascaded Attention Decoder | [mednext_emcad](../../configs/architectures/combinations/general/mednext_emcad.yaml) |
| `edldnet` | EDLDNet decoder | [pvtv2_edldnet](../../configs/architectures/combinations/general/pvtv2_edldnet.yaml) |
| `gcascade` | G-CASCADE with add fusion | [pvtv2_gcascade](../../configs/architectures/combinations/general/pvtv2_gcascade.yaml) |
| `gcascade_cat` | G-CASCADE with concat fusion | [gcascade_cat](../../configs/architectures/combinations/general/gcascade_cat_basic.yaml) |
| `merit_add` | MERIT decoder (add fusion) | [merit_add](../../configs/architectures/combinations/general/merit_add_basic.yaml) |
| `merit_cat` | MERIT decoder (concat fusion) | [merit_cat](../../configs/architectures/combinations/general/merit_cat_basic.yaml) |

## Pyramid (1)

Pyramid aggregation decoder.

| Name | Description | YAML |
|---|---|---|
| `upernet` | UPerNet Unified Perceptual Parsing | [upernet](../../configs/architectures/combinations/general/upernet_basic.yaml) |

## MLP (2)

MLP-based decoders.

| Name | Description | YAML |
|---|---|---|
| `mlp` | Generic MLP decoder | [mlp_resnet34](../../configs/architectures/combinations/general/mlp_resnet34.yaml) |
| `segformer` | SegFormer-style MLP decoder | [swinunet_segformer](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |

## Specific (12)

Architecture-specific decoders.

| Name | Associated Network | YAML |
|---|---|---|
| `cfanet` | CFA-Net | [cfanet](../../configs/architectures/combinations/general/cfanet_basic.yaml) |
| `dcsaunet` | DCSAU-Net | [dcsaunet](../../configs/architectures/combinations/general/dcsaunet_basic.yaml) |
| `rwkv_unet` | RWKV-UNet | [rwkv_unet](../../configs/architectures/combinations/general/rwkv_unet.yaml) |
| `kiunet` | KiU-Net | [kiunet](../../configs/architectures/combinations/general/kiunet_basic.yaml) |
| `transunet` | TransUNet (CUP) | [transunet](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `fatnet` | FAT-Net | [fatnet](../../configs/architectures/combinations/general/fatnet_basic.yaml) |
| `h2former` | H2Former | [h2former](../../configs/architectures/combinations/general/h2former_basic.yaml) |
| `hiformer` | HiFormer | [hiformer](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `missformer` | MISSFormer | [missformer](../../configs/architectures/combinations/general/missformer_basic.yaml) |
| `scaleformer` | ScaleFormer | [scaleformer](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `malunet` | MALUNet | [malunet](../../configs/architectures/combinations/general/malunet_basic.yaml) |
| `ege_unet` | EGE-UNet | [ege_unet](../../configs/architectures/combinations/general/ege_unet_basic.yaml) |

## Transformer (5)

Transformer-based decoders.

| Name | Description | YAML |
|---|---|---|
| `daeformer` | DAEFormer decoder | [daeformer](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `mtunet` | MT-UNet decoder | [mtunet](../../configs/architectures/combinations/general/mtunet_basic.yaml) |
| `nnformer` | nnFormer decoder | [mednext_nnformer](../../configs/architectures/combinations/general/mednext_nnformer.yaml) |
| `swinunet` | Swin-UNet decoder | [swinunet](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `uctransnet` | UCTransNet decoder | [uctransnet](../../configs/architectures/combinations/general/uctransnet.yaml) |

## Attention (3)

Attention-based decoders.

| Name | Description | YAML |
|---|---|---|
| `attention` | Attention gate decoder | [attention_unet](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `ham` | Hybrid Attention Module | [ham_resnet34](../../configs/architectures/combinations/general/ham_resnet34.yaml) |
| `lawin` | Large Window Attention decoder | [lawin_resnet50](../../configs/architectures/combinations/general/lawin_resnet50.yaml) |

## Mamba (1)

| Name | Description | YAML |
|---|---|---|
| `vmunet` | VM-UNet Mamba decoder | [vmunet](../../configs/architectures/combinations/general/vm_unet_basic.yaml) |

---

## YAML Usage Example

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: emcad          # choose any decoder
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 24
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
    lr: 0.01
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```
