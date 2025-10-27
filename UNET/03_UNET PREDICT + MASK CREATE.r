library(reticulate)
  library("tensorflow")
# Настройка Python
py_pth <- "C:\\Users\\usato\\AppData\\Local\\r-miniconda\\envs\\tf_2_10_env/python.exe"
use_python(py_pth, required = TRUE)
use_condaenv("tf_2_10_env", required = TRUE)
tf$config$list_physical_devices('GPU')








  library("abind")
  library("parallel")
  library("doParallel")
  library("foreach")
  library("tensorflow")
  library("tfdatasets")
  library("purrr")
  library(EBImage)
  library(keras)
                     predict_dir = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA predict 20251010\\IN_split"
				     savePredMskDir = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA predict 20251010\\IN_predicted Mask"
                     batch_size_global=  400
                     batch_size =40		
                    vision_dimensions = 384L					 
##############################################################################  
# Функции потерь и метрик
dice_coef <- custom_metric("dice_coef", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

iou_metric <- custom_metric("iou", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  union <- k_sum(y_true_f) + k_sum(y_pred_f) - intersection
  (intersection + smooth) / (union + smooth)
})
pth = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA_Train\\TRAIN_cell\\checkpoints_resnet\\CellSegmentation20251016_Val_0.91_epoch_96_256.h5"

model = load_model_hdf5(pth,
custom_objects = list(
                      dice_coef = dice_coef,
                      iou = iou_metric
      ),compile = FALSE)
	  model
 ############################################################################  
 unlink(savePredMskDir,recursive=T)
 dir.create(savePredMskDir)
 ###################################################################
  listImage_glob <<-list.files(predict_dir, full.names = T,  recursive = T, include.dirs = F,pattern="png|JPG|jpg|jpeg|JPEG")
  global_steps <<- round(length(listImage_glob)/batch_size_global)
  if(length(listImage_glob) > (global_steps*batch_size_global)) {global_steps=global_steps+1}
 ###################################################### 
 create_dataset <- function(data1, batch_size = batch_size, vision_dimensions) {  
  dataset <- data1 %>% 
    tensor_slices_dataset() %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$decode_image(tf$io$read_file(.x$img), channels = 3, expand_animations = FALSE)
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$resize(.x$img, size = c(vision_dimensions, vision_dimensions))
    )) %>%
	 dataset_map(~.x %>% list_modify(
    image <-tf$keras$applications$resnet$preprocess_input(.x$img) 	# Предобработка для VGG16 (нормализация ImageNet)
	 ))
  dataset <- dataset %>% 
    dataset_batch(batch_size)
  dataset %>% 
    dataset_map(unname) # Keras needs an unnamed output.
}
  #################################################################################
  for (e in 1:global_steps) { 
  ###############################
  if (e==1) { cl <- makePSOCKcluster(detectCores (logical=F)-1) 
               clusterEvalQ(cl, {library(EBImage)})
               registerDoParallel(cl)}
  ########################
  
  
  
  batch_ind_global <- c(1:length(listImage_glob))[1:batch_size_global]
  listImage <- listImage_glob[batch_ind_global]
  listImage=listImage[is.na(listImage)==F]
  if (length(listImage_glob) > length(listImage)) {
 # batch_ind_global=batch_ind_global[is.na(batch_ind_global)==F]
    listImage_glob <<- listImage_glob[-batch_ind_global] 
  }
  data1 <<- tibble::tibble(img = listImage)
  #######################################################################################
  pred_dataset <- create_dataset(data1, batch_size=batch_size, vision_dimensions=vision_dimensions)
  preds=keras:::predict.keras.engine.training.Model(object=model,
                                                    x=pred_dataset)
 print(paste0("Done  ", e, "  pred from  " ,global_steps)) 	
 
    #######################################################################################

 foreach(i = 1:length(listImage)) %dopar% {	
#   for (i in 1: length (listImage)) {

     name=basename(listImage[i])
     img_pth=listImage[i]
	 name=basename(img_pth)
	 MskName=gsub("jpg","png",name)
     mask0=preds[i, , , ]
	 
     img0 <- mask0#t(mask0)
     dim(img0) <- c(384, 384, 1)
     img = getFrame(img0, 1)
	 img=resize(img, w = 380, h = 380)
	 PthSave=paste0(savePredMskDir, "\\", MskName)
	 writeImage(img,PthSave)
	 
	 
    	
}
} 
	
	
	



stopCluster(cl)


