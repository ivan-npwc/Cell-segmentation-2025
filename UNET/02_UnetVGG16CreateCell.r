
library(keras)
library(tensorflow)


# Настройка Python
py_pth <- "C:\\Users\\usato\\AppData\\Local\\r-miniconda\\envs\\tf_2_10_env/python.exe"
use_python(py_pth, required = TRUE)
use_condaenv("tf_2_10_env", required = TRUE)

# Проверка GPU
tf$config$list_physical_devices('GPU')



# Создание U-Net с VGG16 энкодером
create_vgg16_unet <- function(input_shape = c(256, 256, 3), dropout_rate = 0.3) {
  
  # Загружаем VGG16 с предобученными весами (ImageNet)
  vgg_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_shape = input_shape
  )
  
  # Замораживаем веса VGG (опционально - можно разморозить позже)
  vgg_base$trainable <- FALSE
  
  cat("VGG16 base loaded with ImageNet weights\n")
  
  # Получаем выходы промежуточных слоев VGG для skip connections
  # Block 1
  block1_conv2 <- vgg_base$get_layer("block1_conv2")$output
  block1_pool <- vgg_base$get_layer("block1_pool")$output
  
  # Block 2
  block2_conv2 <- vgg_base$get_layer("block2_conv2")$output
  block2_pool <- vgg_base$get_layer("block2_pool")$output
  
  # Block 3
  block3_conv3 <- vgg_base$get_layer("block3_conv3")$output
  block3_pool <- vgg_base$get_layer("block3_pool")$output
  
  # Block 4
  block4_conv3 <- vgg_base$get_layer("block4_conv3")$output
  block4_pool <- vgg_base$get_layer("block4_pool")$output
  
  # Block 5 (bottleneck)
  block5_conv3 <- vgg_base$get_layer("block5_conv3")$output
  block5_pool <- vgg_base$get_layer("block5_pool")$output
  
  # Декодер U-Net
  
  # Up sampling 1
  up6 <- block5_pool %>%
    layer_conv_2d_transpose(512, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  up6 <- up6 %>% layer_dropout(dropout_rate)
  
  # Concatenate with block5_conv3
  concat6 <- layer_concatenate(list(up6, block5_conv3))
  
  conv6 <- concat6 %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up sampling 2
  up7 <- conv6 %>%
    layer_conv_2d_transpose(512, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  up7 <- up7 %>% layer_dropout(dropout_rate)
  
  # Concatenate with block4_conv3
  concat7 <- layer_concatenate(list(up7, block4_conv3))
  
  conv7 <- concat7 %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(512, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up sampling 3
  up8 <- conv7 %>%
    layer_conv_2d_transpose(256, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  up8 <- up8 %>% layer_dropout(dropout_rate)
  
  # Concatenate with block3_conv3
  concat8 <- layer_concatenate(list(up8, block3_conv3))
  
  conv8 <- concat8 %>%
    layer_conv_2d(256, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(256, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up sampling 4
  up9 <- conv8 %>%
    layer_conv_2d_transpose(128, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  up9 <- up9 %>% layer_dropout(dropout_rate)
  
  # Concatenate with block2_conv2
  concat9 <- layer_concatenate(list(up9, block2_conv2))
  
  conv9 <- concat9 %>%
    layer_conv_2d(128, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(128, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Up sampling 5
  up10 <- conv9 %>%
    layer_conv_2d_transpose(64, 2, strides = 2, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  up10 <- up10 %>% layer_dropout(dropout_rate)
  
  # Concatenate with block1_conv2
  concat10 <- layer_concatenate(list(up10, block1_conv2))
  
  conv10 <- concat10 %>%
    layer_conv_2d(64, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(64, 3, padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  
  # Output layer
  outputs <- conv10 %>%
    layer_conv_2d(1, 1, activation = "sigmoid")
  
  model <- keras_model(inputs = vgg_base$input, outputs = outputs)
  
  return(model)
}

# Создание модели с VGG16 энкодером
model <- create_vgg16_unet()

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

# Компиляция с более низким learning rate (предобученные веса)
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-5),  # Меньше LR из-за предобученных весов
  loss = "binary_crossentropy",
  metrics = list("accuracy", dice_coef, iou_metric)
)

# Вывод информации о модели
cat("=== VGG16 U-Net Model Summary ===\n")
summary(model)