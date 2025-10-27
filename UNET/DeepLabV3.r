
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
library(keras)
library(tensorflow)

#' Полная исправленная реализация DeepLabV3+
#' 
create_deeplabv3_plus_fixed <- function(input_shape = c(384, 384, 3), 
                                       num_classes = 1, 
                                       backbone = "resnet50") {
  
  inputs <- layer_input(shape = input_shape)
  
  # Загрузка предобученной ResNet50
  base_model <- application_resnet50(
    weights = "imagenet",
    include_top = FALSE,
    input_tensor = inputs
  )
  
  # Получаем нужные слои для ASPP и декодера
  # ASPP input - выход из последнего блока ResNet
  aspp_input <- base_model$output
  # Low-level features из более раннего слоя (conv2)
  low_level_feat <- base_model$get_layer("conv2_block3_out")$output
  
  cat("ASPP input shape:", dim(aspp_input), "\n")
  cat("Low-level features shape:", dim(low_level_feat), "\n")
  
  # ASPP модуль
  aspp_output <- aspp_module_fixed(aspp_input)
  
  # Декодер с правильным выравниванием размеров
  decoder_output <- deeplab_decoder_fixed(aspp_output, low_level_feat, num_classes)
  
  # Финальный апсэмплинг до исходного размера
  final_upsample_factor <- input_shape[1] / dim(decoder_output)[2]
  final_output <- decoder_output %>%
    layer_upsampling_2d(size = c(final_upsample_factor, final_upsample_factor), 
                       interpolation = "bilinear")
  
  model <- keras_model(inputs = inputs, outputs = final_output)
  return(model)
}

#' Исправленный ASPP модуль
#' 
aspp_module_fixed <- function(input_tensor, filters = 256) {
  
  # Branch 1: 1x1 convolution
  branch1 <- input_tensor %>%
    layer_conv_2d(filters, 1, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Branch 2: 3x3 convolution with rate = 6
  branch2 <- input_tensor %>%
    layer_conv_2d(filters, 3, dilation_rate = 6, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Branch 3: 3x3 convolution with rate = 12
  branch3 <- input_tensor %>%
    layer_conv_2d(filters, 3, dilation_rate = 12, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Branch 4: 3x3 convolution with rate = 18
  branch4 <- input_tensor %>%
    layer_conv_2d(filters, 3, dilation_rate = 18, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Branch 5: Image Pooling
  branch5 <- input_tensor %>%
    layer_global_average_pooling_2d() %>%
    layer_reshape(c(1, 1, dim(input_tensor)[4])) %>%
    layer_conv_2d(filters, 1, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Вычисляем размер для апсэмплинга branch5
  target_size <- dim(input_tensor)[2:3]
  branch5_upsampled <- branch5 %>%
    layer_upsampling_2d(size = target_size, interpolation = "bilinear")
  
  # Concatenate all branches
  concatenated <- layer_concatenate(list(branch1, branch2, branch3, branch4, branch5_upsampled))
  
  # Final 1x1 convolution
  output <- concatenated %>%
    layer_conv_2d(filters, 1, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(0.1)
  
  return(output)
}

#' Исправленный декодер с выравниванием размеров
#' 
deeplab_decoder_fixed <- function(aspp_output, low_level_feat, num_classes) {
  
  # Обработка low-level features
  low_level_feat_processed <- low_level_feat %>%
    layer_conv_2d(48, 1, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Вычисляем коэффициент апсэмплинга
  # Для ResNet50 с input 512x512:
  # - aspp_output: 16x16 (после 32x downsampling)  
  # - low_level_feat: 128x128 (после 4x downsampling)
  # Нужно апсэмплить aspp_output в 8 раз
  
  aspp_shape <- dim(aspp_output)
  low_level_shape <- dim(low_level_feat_processed)
  
  upsample_factor <- low_level_shape[2] / aspp_shape[2]  # 128 / 16 = 8
  
  cat("Upsampling ASPP by factor:", upsample_factor, "\n")
  
  # Апсэмплинг ASPP output
  aspp_upsampled <- aspp_output %>%
    layer_upsampling_2d(size = c(upsample_factor, upsample_factor), 
                       interpolation = "bilinear")
  
  # Проверяем размеры перед конкатенацией
  cat("After upsampling:\n")
  cat("  ASPP shape:", dim(aspp_upsampled), "\n")
  cat("  Low-level shape:", dim(low_level_feat_processed), "\n")
  
  # Теперь конкатенация
  concatenated <- layer_concatenate(list(aspp_upsampled, low_level_feat_processed))
  
  # Декодерные свертки
  x <- concatenated %>%
    layer_conv_2d(256, 3, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(0.1) %>%
    
    layer_conv_2d(256, 3, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(0.1)
  
  # Финальная классификация
  output <- x %>%
    layer_conv_2d(num_classes, 1, padding = "same")
  
  return(output)
}

# Создаем исправленную модель
deeplab_model <- create_deeplabv3_plus_fixed(
  input_shape = c(384, 384, 3),
  num_classes = 1
)

# Компиляция
deeplab_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

summary(deeplab_model)