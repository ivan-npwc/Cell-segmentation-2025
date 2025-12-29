library(tcltk)
library(keras)

#tcltk::tk_choose.files()


pth=  "/home/ivan/TRAIN/CELL/checkpoints_deeplabv3/CellSegmentation_DeepLabV3_Val_0.900_epoch_17.h5"



 dice_coef <<- custom_metric("dice_coef", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})
 dice_coef_loss <- function(y_true, y_pred) - dice_coef(y_true, y_pred)
 
unet1 <- load_model_hdf5(pth, custom_objects = c(dice_coef = dice_coef,
                                                        dice_coef_loss=dice_coef_loss))

a=get_weights(unet1)
saveRDS(a,"/home/ivan/CELL/CellSegmentation_DeepLabV3_Vl_0.9_epoch_17")

##########################################################

#library(keras)
#pth= "/home/ivan/TRAIN/VGG16CheckPoints/NFS_PUP_FS_SPLIT_2024-12-16_loss_0.08_epoch_20Best.h5"
#mdl <- load_model_hdf5(pth)
#a=get_weights(mdl)
#saveRDS(a,"/media/ivan/SSLbrand_2025-03-06_val_0.29_epoch_26_256")

