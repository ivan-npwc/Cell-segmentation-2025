
library(reticulate)
  library("tensorflow")
# Настройка Python
py_pth <- "C:\\Users\\usato\\AppData\\Local\\r-miniconda\\envs\\tf_2_10_env/python.exe"
use_python(py_pth, required = TRUE)
use_condaenv("tf_2_10_env", required = TRUE)
tf$config$list_physical_devices('GPU')











library(keras)
library(tfdatasets)
library(tidyverse)
library(tensorflow)
library(reticulate)



 # Параметры
 #trainDir <-  "C:\\Users\\usato\\SSL_DB\\TRAIN\\TRAIN_pv\\No zero"
 trainDir <- "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA_Train\\TRAIN_cell"
 images_dir <- file.path(trainDir, "Image")
 masks_dir <- file.path(trainDir, "Mask")

epochs <- 100
batch_size <- 4L
img_height <- 384L
img_width <- 384L
validation_split <- 0.2

# Проверка данных
cat("Images found:", length(list.files(images_dir)), "\n")
cat("Masks found:", length(list.files(masks_dir)), "\n")

# Создание датафрейма с путями
image_files <- list.files(images_dir, full.names = TRUE, pattern = "\\.(jpg|jpeg|png)$", ignore.case = TRUE)
mask_files <- list.files(masks_dir, full.names = TRUE, pattern = "\\.(png|jpg|jpeg)$", ignore.case = TRUE)

# Проверка соответствия имен
get_base_name <- function(path) {
  tools::file_path_sans_ext(basename(path))
}

image_names <- sapply(image_files, get_base_name)
mask_names <- sapply(mask_files, get_base_name)

# Находим общие имена
common_names <- intersect(image_names, mask_names)
cat("Common image-mask pairs:", length(common_names), "\n")

if (length(common_names) == 0) {
  stop("No matching image-mask pairs found!")
}

# Фильтруем файлы по общим именам
image_files <- image_files[image_names %in% common_names]
mask_files <- mask_files[mask_names %in% common_names]

# Упорядочиваем по именам
image_files <- image_files[order(image_names[image_names %in% common_names])]
mask_files <- mask_files[order(mask_names[mask_names %in% common_names])]

data_df <- data.frame(
  image = image_files,
  mask = mask_files,
  stringsAsFactors = FALSE
)

# Обновленная предобработка для ResNet50
preprocess_image <- function(image_path) {
  image <- tf$io$read_file(image_path)
  image <- tf$image$decode_image(image, channels = 3, expand_animations = FALSE)
  image <- tf$image$convert_image_dtype(image, dtype = tf$float32)
  image <- tf$image$resize(image, size = c(img_height, img_width))
  # ResNet50 предобработка
  image <- tf$keras$applications$resnet$preprocess_input(image)
  return(image)
}

preprocess_mask <- function(mask_path) {
  mask <- tf$io$read_file(mask_path)
  mask <- tf$image$decode_image(mask, channels = 1, expand_animations = FALSE)
  mask <- tf$image$convert_image_dtype(mask, dtype = tf$float32)
  mask <- tf$image$resize(mask, size = c(img_height, img_width))
  mask <- tf$round(mask)  # Бинаризация масок
  return(mask)
}

# Создание tf.data.Dataset
create_dataset <- function(df, batch_size, shuffle = FALSE, augment = FALSE) {
  
  dataset <- tensor_slices_dataset(list(df$image, df$mask)) %>%
    dataset_map(function(image_path, mask_path) {
      image <- preprocess_image(image_path)
      mask <- preprocess_mask(mask_path)
      list(image, mask)
    }) %>%
    dataset_map(function(image, mask) {
      # Гарантируем правильную форму
      image <- tf$ensure_shape(image, list(img_height, img_width, 3L))  # 3 канала для VGG
      mask <- tf$ensure_shape(mask, list(img_height, img_width, 1L))
      list(image, mask)
    })
  
  if (shuffle) {
    dataset <- dataset %>% dataset_shuffle(buffer_size = nrow(df))
  }
  
  if (augment) {
    dataset <- dataset %>% 
      dataset_map(function(image, mask) {
        # Случайное отражение по горизонтали
        result <- tf$cond(
          tf$random$uniform(shape = shape(), minval = 0, maxval = 1) > 0.5,
          true_fn = function() {
            list(tf$image$flip_left_right(image), tf$image$flip_left_right(mask))
          },
          false_fn = function() list(image, mask)
        )
        image <- result[[1]]
        mask <- result[[2]]
        ###############################################
        # Flip up-down
        result <- tf$cond(
          tf$random$uniform(shape = shape(), minval = 0, maxval = 1) > 0.5,
          true_fn = function() {
            list(tf$image$flip_up_down(image), tf$image$flip_up_down(mask))
          },
          false_fn = function() list(image, mask)
        )
        
        image <- result[[1]]
        mask <- result[[2]]
     ###################################################################
	    result <- tf$cond(
          tf$random$uniform(shape = shape(), minval = 0, maxval = 1) > 0.5,
     true_fn = function() {

	scale_factor <- runif(1, 0.3, 0.7)
    new_height <- as.integer(img_height * scale_factor)
    new_width <- as.integer(img_width * scale_factor)
    downscaled <- tf$image$resize(image, size = c(new_height, new_width))
	
	
   list(tf$image$resize(downscaled, size = c(img_height, img_width)),mask)

	  },
	    false_fn = function() list(image, mask)
        )

	    image <- result[[1]]
        mask <- result[[2]]
########################################################	 
        # Яркость (только для изображения)
        image <- tf$cond(
          tf$random$uniform(shape = shape(), minval = 0, maxval = 1) > 0.5,
          true_fn = function() tf$image$random_brightness(image, max_delta = 0.1),
          false_fn = function() image
        )
        list(image, mask)
      })
  }
  
  dataset <- dataset %>%
    dataset_batch(batch_size) %>%
    dataset_prefetch(buffer_size = tf$data$AUTOTUNE)
  
  return(dataset)
}

# Разделение на train/validation
set.seed(123)
train_indices <- sample(1:nrow(data_df), size = round((1 - validation_split) * nrow(data_df)))
train_df <- data_df[train_indices, ]
val_df <- data_df[-train_indices, ]

cat("Training samples:", nrow(train_df), "\n")
cat("Validation samples:", nrow(val_df), "\n")

# Создание датасетов
train_dataset <- create_dataset(train_df, batch_size, shuffle = FALSE, augment = TRUE)
val_dataset <- create_dataset(val_df, batch_size, shuffle = FALSE, augment = FALSE)

# Проверка одного батча
check_batch <- function(dataset) {
  iterator <- as_iterator(dataset)
  batch <- iter_next(iterator)
  cat("Batch image shape:", batch[[1]]$shape$as_list(), "\n")
  cat("Batch mask shape:", batch[[2]]$shape$as_list(), "\n")
  cat("Image range:", as.numeric(tf$reduce_min(batch[[1]])), "to", as.numeric(tf$reduce_max(batch[[1]])), "\n")
  cat("Mask range:", as.numeric(tf$reduce_min(batch[[2]])), "to", as.numeric(tf$reduce_max(batch[[2]])), "\n")
}

cat("=== Training batch check ===\n")
check_batch(train_dataset)
cat("=== Validation batch check ===\n")
check_batch(val_dataset)
################################################################
# Альтернативная версия с более простой архитектурой (рекомендуется)
create_resnet50_unet_simple <- function(input_shape = c(384, 384, 3), dropout_rate = 0.3) {
  
  # Используем 384x384 для идеального соответствия размеров
  base_model <- application_resnet50(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = input_shape
  )
  
  base_model$trainable <- FALSE
  
  cat("ResNet50 base loaded for", paste(input_shape, collapse = "x"), "\n")
  
  # Skip connections для 384x384:
  # conv2_block3_out: 96x96
  # conv3_block4_out: 48x48  
  # conv4_block6_out: 24x24
  # conv5_block3_out: 12x12
  
  skip2 <- base_model$get_layer("conv2_block3_out")$output  # 96x96
  skip3 <- base_model$get_layer("conv3_block4_out")$output  # 48x48
  skip4 <- base_model$get_layer("conv4_block6_out")$output  # 24x24
  bottleneck <- base_model$output  # 12x12
  
  # Декодер
  
  # Up 1: 12x12 -> 24x24
  up1 <- bottleneck %>%
    layer_conv_2d_transpose(512, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  concat1 <- layer_concatenate(list(up1, skip4))
  conv1 <- concat1 %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up 2: 24x24 -> 48x48
  up2 <- conv1 %>%
    layer_conv_2d_transpose(256, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  concat2 <- layer_concatenate(list(up2, skip3))
  conv2 <- concat2 %>%
    layer_conv_2d(256, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(256, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up 3: 48x48 -> 96x96
  up3 <- conv2 %>%
    layer_conv_2d_transpose(128, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  concat3 <- layer_concatenate(list(up3, skip2))
  conv3 <- concat3 %>%
    layer_conv_2d(128, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(128, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up 4: 96x96 -> 192x192
  up4 <- conv3 %>%
    layer_conv_2d_transpose(64, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  conv4 <- up4 %>%
    layer_conv_2d(64, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(64, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up 5: 192x192 -> 384x384
  up5 <- conv4 %>%
    layer_conv_2d_transpose(32, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Output
  outputs <- up5 %>%
    layer_conv_2d(1, 1, activation = "sigmoid")
  
  model <- keras_model(inputs = base_model$input, outputs = outputs)
  
  return(model)
}

####################################################
model = create_resnet50_unet_simple()

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
###################################################
 dice_coef_loss <- function(y_true, y_pred) - dice_coef(y_true, y_pred)
 #########################################################
# Компиляция с более низким learning rate (предобученные веса)
#model %>% compile(
#  optimizer = optimizer_adam(learning_rate = 1e-5),  # Меньше LR из-за предобученных весов
#  loss = "binary_crossentropy",
#  metrics = list("accuracy", dice_coef, iou_metric)
#)

  model <- model %>%
       compile(
           optimizer = optimizer_adam(learning_rate= 0.00001 , decay = 1e-6 ),
           loss =     dice_coef_loss,#"binary_crossentropy", 
           metrics = dice_coef #, metric_binary_accuracy
              )





# Колбэки
checkpoint_dir <- file.path(trainDir, "checkpoints_resnet")
dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
BaseName <- basename(file.path(checkpoint_dir, "Val_{val_dice_coef:.2f}_epoch_{epoch:02d}_256.h5"))
filepath <- paste0(checkpoint_dir,"\\CellSegmentation20251016_",BaseName)


callbacks <- list(
  callback_model_checkpoint(
   filepath = filepath,
   period = 1,
   verbose = 1
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 8,
    verbose = 1,
    min_lr = 1e-7
  ),
  callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    verbose = 1,
    restore_best_weights = TRUE
  ),
  callback_tensorboard(
    log_dir = file.path(checkpoint_dir, "logs")
  )
)

# Обучение
cat("Starting training with resnet U-Net (ImageNet weights)...\n")
history <- model %>% fit(
  train_dataset,
  epochs = epochs,
  validation_data = val_dataset,
  callbacks = callbacks,
  verbose = 1
)
#####################################################################
#pth="C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA_Train\\TRAIN_cell\\checkpoints_vgg16\\best_model.h5"
#
#model = load_model_hdf5(pth,
#custom_objects = list(
#                      dice_coef = dice_coef,
#                      iou = iou_metric
#      ))
#
# Функция для разморозки весов и дообучения (опционально)
#unfreeze_and_finetune <- function(model, learning_rate = 1e-6) {
#  # Размораживаем последние блоки VGG
#  for (layer in model$layers) {
#    if (grepl("block5", layer$name) || grepl("block4", layer$name)) {
#      layer$trainable <- TRUE
#    }
#  }
#  
#  # Перекомпилируем с меньшим learning rate
#  model %>% compile(
#    optimizer = optimizer_adam(learning_rate = learning_rate),
#    loss = "binary_crossentropy",
#    metrics = list("accuracy", dice_coef, iou_metric)
#  )
#  
#  return(model)
#}

# Опционально: дообучение с размороженными слоями
#cat("Starting fine-tuning with unfrozen layers...\n")
#model <- unfreeze_and_finetune(model, learning_rate = 1e-6)

#history_finetune <- model %>% fit(
#  train_dataset,
#  epochs = 30,  # Короткое дообучение
#  validation_data = val_dataset,
#  callbacks = callbacks,
#  verbose = 1
#)

# Сохранение финальной модели
#model %>% save_model_hdf5(file.path(checkpoint_dir, "final_model_vgg16.h5"))
#cat("Training completed!\n")

#cat("VGG16 U-Net model saved to:", checkpoint_dir, "\n")
#cat("Key features:\n")
#cat("- VGG16 encoder with ImageNet weights\n")
#cat"- Skip connections from VGG16 intermediate layers\n")
#cat("- Batch Normalization and Dropout in decoder\n")
#cat"- Two-stage training: frozen → fine-tuned\n")