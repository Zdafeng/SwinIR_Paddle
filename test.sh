# 001 Classical Image Super-Resolution (middle size)
# Note that --training_patch_size is just used to differentiate two different settings in Table 2 of the paper. Images are NOT tested patch by patch.
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pdparams --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pdparams --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pdparams --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pdparams --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pdparams --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pdparams --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pdparams --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pdparams --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR


# 002 Lightweight Image Super-Resolution (small size)
python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pdparams --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task lightweight_sr --scale 3 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pdparams --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task lightweight_sr --scale 4 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pdparams --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR


# 003 Real-World Image Super-Resolution (use --tile 400 if you run out-of-memory)
# (middle size)
python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pdparams --folder_lq testsets/RealSRSet+5images --tile 400

# (larger size + trained on more datasets)
python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pdparams --folder_lq testsets/RealSRSet+5images


# 004 Grayscale Image Deoising (middle size)
python main_test_swinir.py --task gray_dn --noise 15 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pdparams --folder_gt testsets/Set12
python main_test_swinir.py --task gray_dn --noise 25 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pdparams --folder_gt testsets/Set12
python main_test_swinir.py --task gray_dn --noise 50 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pdparams --folder_gt testsets/Set12


# 005 Color Image Deoising (middle size)
python main_test_swinir.py --task color_dn --noise 15 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pdparams --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 25 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pdparams --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 50 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pdparams --folder_gt testsets/McMaster


# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
python main_test_swinir.py --task jpeg_car --jpeg 10 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pdparams --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 20 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pdparams --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 30 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pdparams --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 40 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pdparams --folder_gt testsets/classic5

