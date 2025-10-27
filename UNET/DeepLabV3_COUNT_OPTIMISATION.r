library(reticulate)
library(tensorflow)
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
library(EBImage)  # Основная библиотека для обработки изображений

# Параметры
trainDir <- "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA_Train\\TRAIN_cell"
images_dir <- file.path(trainDir, "Image")
masks_dir <- file.path(trainDir, "Mask")

epochs <- 100
batch_size <- 4L
img_height <- 512L
img_width <- 512L
validation_split <- 0.2

# Проверка данных
cat("Images found:", length(list.files(images_dir)), "\n")
cat("Masks found:", length(list.files(masks_dir)), "\n")

# Создание датафрейма с путями и количеством объектов
image_files <- list.files(images_dir, full.names = TRUE, pattern = "\\.(jpg|jpeg|png)$", ignore.case = TRUE)
mask_files <- list.files(masks_dir, full.names = TRUE, pattern = "\\.(png|jpg|jpeg)$", ignore.case = TRUE)

# Функция для извлечения количества объектов из имени файла
extract_object_count <- function(filename) {
  # Пример: "1#1_A4_vlf01#23.png" -> 23
  base_name <- tools::file_path_sans_ext(basename(filename))
  
  # Ищем число после последнего #
  matches <- regmatches(base_name, regexpr("#[0-9]+$", base_name))
  if (length(matches) > 0) {
    count <- as.numeric(substring(matches, 2))
    return(count)
  }
  
  # Альтернативный паттерн: ищем числа в конце имени
  matches <- regmatches(base_name, regexpr("[0-9]+$", base_name))
  if (length(matches) > 0) {
    return(as.numeric(matches))
  }
  
  # Если не нашли, возвращаем NA
  warning("Could not extract object count from: ", filename)
  return(NA)
}

# Создаем датафрейм с количеством объектов
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

# Извлекаем количество объектов из масок
object_counts <- sapply(mask_files, extract_object_count)

# Проверяем, что все количества извлечены
if (any(is.na(object_counts))) {
  cat("Warning: Could not extract object counts for", sum(is.na(object_counts)), "files\n")
  # Удаляем файлы с NA
  valid_indices <- !is.na(object_counts)
  image_files <- image_files[valid_indices]
  mask_files <- mask_files[valid_indices]
  object_counts <- object_counts[valid_indices]
}

data_df <- data.frame(
  image = image_files,
  mask = mask_files,
  object_count = object_counts,
  stringsAsFactors = FALSE
)

cat("Object count statistics:\n")
print(summary(data_df$object_count))

# Предобработка для DeepLabV3+ (ResNet50 backbone)
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

# Функция для подсчета объектов в предсказанной маске с использованием EBImage
count_objects_ebimage <- function(mask_array, min_size = 50, connectivity = 8) {
  # Конвертируем маску в бинарное изображение EBImage
  binary_mask <- mask_array > 0.5
  
  # Если маска пустая, возвращаем 0
  if (sum(binary_mask) == 0) {
    return(0)
  }
  
  # Создаем объект EBImage
  ebimage_mask <- EBImage::Image(binary_mask, dim = dim(binary_mask))
  
  # Находим связанные компоненты
  labeled_mask <- EBImage::bwlabel(ebimage_mask, connectivity = connectivity)
  
  # Вычисляем свойства каждого объекта
  object_features <- EBImage::computeFeatures.shape(labeled_mask)
  
  if (is.null(object_features) || nrow(object_features) == 0) {
    return(0)
  }
  
  # Фильтруем объекты по размеру (исключаем слишком маленькие)
  object_areas <- object_features[, "s.area"]
  valid_objects <- object_areas >= min_size
  
  return(sum(valid_objects))
}

# Улучшенная функция подсчета с морфологическими операциями
count_objects_advanced <- function(mask_array, min_size = 50, max_size = 10000) {
  # Конвертируем маску в бинарное изображение EBImage
  binary_mask <- mask_array > 0.5
  
  if (sum(binary_mask) == 0) {
    return(0)
  }
  
  # Создаем объект EBImage
  ebimage_mask <- EBImage::Image(binary_mask, dim = dim(binary_mask))
  
  # Применяем морфологические операции для улучшения сегментации
  # Закрытие для заполнения маленьких отверстий
  kernel <- EBImage::makeBrush(3, shape = "disc")
  cleaned_mask <- EBImage::closing(ebimage_mask, kernel)
  
  # Открытие для разделения касающихся объектов
  separated_mask <- EBImage::opening(cleaned_mask, kernel)
  
  # Находим связанные компоненты
  labeled_mask <- EBImage::bwlabel(separated_mask, connectivity = 8)
  
  # Вычисляем свойства объектов
  object_features <- EBImage::computeFeatures.shape(labeled_mask)
  
  if (is.null(object_features) || nrow(object_features) == 0) {
    return(0)
  }
  
  # Фильтруем объекты по размеру
  object_areas <- object_features[, "s.area"]
  valid_objects <- (object_areas >= min_size) & (object_areas <= max_size)
  
  return(sum(valid_objects))
}

# Кастомный callback для оценки количества объектов с EBImage
ObjectCountEBImageCallback <- R6::R6Class(
  "ObjectCountEBImageCallback",
  inherit = KerasCallback,
  
  public = list(
    train_counts = NULL,
    val_counts = NULL,
    train_pred_counts = NULL,
    val_pred_counts = NULL,
    min_object_size = 50,
    
    on_epoch_end = function(epoch, logs = NULL) {
      # Вычисляем точность подсчета объектов для тренировочных данных
      train_mae <- self$evaluate_object_count(self$train_counts, self$train_pred_counts)
      # Вычисляем точность подсчета объектов для валидационных данных
      val_mae <- self$evaluate_object_count(self$val_counts, self$val_pred_counts)
      
      logs$object_count_mae <- train_mae
      logs$val_object_count_mae <- val_mae
      
      cat(sprintf("Epoch %d - Object Count MAE: %.2f - Val Object Count MAE: %.2f\n", 
                  epoch, train_mae, val_mae))
    },
    
    evaluate_object_count = function(true_counts, pred_counts) {
      if (length(true_counts) == 0 || length(pred_counts) == 0) return(0)
      mae <- mean(abs(true_counts - pred_counts))
      return(mae)
    },
    
    # Функция для сбора предсказаний с EBImage
    collect_predictions = function(model, dataset, true_counts) {
      pred_counts <- c()
      iterator <- as_iterator(dataset)
      batch_num <- 1
      
      while(TRUE) {
        batch <- iter_next(iterator)
        if (is.null(batch)) break
        
        images <- batch[[1]]
        predictions <- predict(model, images)
        
        # Подсчитываем объекты в каждом предсказании с помощью EBImage
        for(i in 1:dim(predictions)[1]) {
          pred_mask <- as.array(predictions[i,,,1])
          count <- count_objects_advanced(pred_mask, min_size = self$min_object_size)
          pred_counts <- c(pred_counts, count)
        }
        
        cat(sprintf("Processed batch %d\n", batch_num))
        batch_num <- batch_num + 1
      }
      
      return(pred_counts)
    },
    
    # Функция для визуализации результатов подсчета
    visualize_counting = function(model, dataset, n_samples = 3) {
      iterator <- as_iterator(dataset)
      batch <- iter_next(iterator)
      
      if (!is.null(batch)) {
        images <- batch[[1]]
        true_masks <- batch[[2]]
        predictions <- predict(model, images)
        
        par(mfrow = c(n_samples, 3))
        
        for(i in 1:min(n_samples, dim(images)[1])) {
          # Оригинальное изображение
          img_array <- as.array(images[i,,,])
          # Денормализация для отображения
          img_array <- (img_array - min(img_array)) / (max(img_array) - min(img_array))
          plot(EBImage::Image(img_array, dim = c(dim(img_array)[1], dim(img_array)[2], 3)))
          title("Original Image")
          
          # Истинная маска с подсчетом
          true_mask <- as.array(true_masks[i,,,1])
          true_count <- count_objects_advanced(true_mask, self$min_object_size)
          plot(EBImage::Image(true_mask, dim = dim(true_mask)))
          title(paste("True Mask - Count:", true_count))
          
          # Предсказанная маска с подсчетом
          pred_mask <- as.array(predictions[i,,,1])
          pred_count <- count_objects_advanced(pred_mask, self$min_object_size)
          plot(EBImage::Image(pred_mask, dim = dim(pred_mask)))
          title(paste("Pred Mask - Count:", pred_count))
        }
      }
    }
  )
)

# [Остальной код создания датасетов остается таким же...]
# Создание tf.data.Dataset с информацией о количестве объектов
create_dataset <- function(df, batch_size, shuffle = FALSE, augment = FALSE) {
  
  dataset <- tensor_slices_dataset(list(df$image, df$mask, df$object_count)) %>%
    dataset_map(function(image_path, mask_path, object_count) {
      image <- preprocess_image(image_path)
      mask <- preprocess_mask(mask_path)
      list(image, mask, object_count)
    }) %>%
    dataset_map(function(image, mask, object_count) {
      # Гарантируем правильную форму
      image <- tf$ensure_shape(image, list(img_height, img_width, 3L))
      mask <- tf$ensure_shape(mask, list(img_height, img_width, 1L))
      list(image, mask, object_count)
    })
  
  if (shuffle) {
    dataset <- dataset %>% dataset_shuffle(buffer_size = nrow(df))
  }
  
  if (augment) {
    dataset <- dataset %>% 
      dataset_map(function(image, mask, object_count) {
        # Аугментации (как в предыдущем коде)
        # ... [аугментации остаются такими же] ...
        list(image, mask, object_count)
      })
  }
  
  # Для тренировки нам нужны только image и mask
  dataset <- dataset %>% 
    dataset_map(function(image, mask, object_count) {
      list(image, mask)
    }) %>%
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
train_dataset <- create_dataset(train_df, batch_size, shuffle = TRUE, augment = TRUE)
val_dataset <- create_dataset(val_df, batch_size, shuffle = FALSE, augment = FALSE)

# [Код создания модели DeepLabV3+ остается таким же...]
# Создание модели DeepLabV3+
create_deeplabv3_plus_cell <- function(input_shape = c(512, 512, 3), num_classes = 1) {
  # ... [код создания модели] ...
}

model <- create_deeplabv3_plus_cell(input_shape = c(img_height, img_width, 3))

# Функции потерь и метрик
dice_coef <- custom_metric("dice_coef", function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
})

dice_coef_loss <- function(y_true, y_pred) -dice_coef(y_true, y_pred)

# Компиляция модели
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),
  loss = dice_coef_loss,
  metrics = list(dice_coef, "binary_accuracy")
)

# Создаем callback для оценки количества объектов с EBImage
object_count_callback <- ObjectCountEBImageCallback$new()
object_count_callback$train_counts <- train_df$object_count
object_count_callback$val_counts <- val_df$object_count
object_count_callback$min_object_size <- 30  # Настройка под ваши клетки

# Колбэки
checkpoint_dir <- file.path(trainDir, "checkpoints_deeplabv3_ebimage")
dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
BaseName <- basename(file.path(checkpoint_dir, "ValCountMAE_{val_object_count_mae:.1f}_epoch_{epoch:02d}.h5"))
filepath <- paste0(checkpoint_dir, "\\CellSegmentation_EBImage_", BaseName)

callbacks <- list(
  callback_model_checkpoint(
    filepath = filepath,
    monitor = "val_object_count_mae",
    save_best_only = TRUE,
    mode = "min",
    verbose = 1
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_object_count_mae",
    factor = 0.5,
    patience = 8,
    verbose = 1,
    mode = "min",
    min_lr = 1e-7
  ),
  callback_early_stopping(
    monitor = "val_object_count_mae",
    patience = 15,
    verbose = 1,
    mode = "min",
    restore_best_weights = TRUE
  ),
  object_count_callback
)

# Функция для обновления предсказаний в callback
update_predictions <- function() {
  cat("Collecting predictions for object count evaluation with EBImage...\n")
  object_count_callback$train_pred_counts <- object_count_callback$collect_predictions(model, train_dataset, train_df$object_count)
  object_count_callback$val_pred_counts <- object_count_callback$collect_predictions(model, val_dataset, val_df$object_count)
}

# Обучение с визуализацией
cat("Starting training with EBImage object counting...\n")

# Визуализация до обучения
cat("Visualizing before training:\n")
object_count_callback$visualize_counting(model, val_dataset, n_samples = 2)

for(epoch in 1:epochs) {
  cat(sprintf("Epoch %d/%d\n", epoch, epochs))
  
  # Обновляем предсказания каждые 5 эпох
  if (epoch %% 5 == 1 || epoch == 1) {
    update_predictions()
  }
  
  history_epoch <- model %>% fit(
    train_dataset,
    epochs = 1,
    validation_data = val_dataset,
    callbacks = callbacks,
    verbose = 1,
    initial_epoch = epoch - 1
  )
  
  # Визуализация прогресса каждые 10 эпох
  if (epoch %% 10 == 0) {
    cat("Progress visualization:\n")
    object_count_callback$visualize_counting(model, val_dataset, n_samples = 2)
  }
}

# Финальная оценка и визуализация
cat("Final evaluation with EBImage...\n")
update_predictions()

final_train_mae <- object_count_callback$evaluate_object_count(
  object_count_callback$train_counts, 
  object_count_callback$train_pred_counts
)
final_val_mae <- object_count_callback$evaluate_object_count(
  object_count_callback$val_counts, 
  object_count_callback$val_pred_counts
)

cat(sprintf("Final Training Object Count MAE: %.2f\n", final_train_mae))
cat(sprintf("Final Validation Object Count MAE: %.2f\n", final_val_mae))

# Финальная визуализация
cat("Final results visualization:\n")
object_count_callback$visualize_counting(model, val_dataset, n_samples = 4)

# Сохранение модели
final_model_path <- file.path(checkpoint_dir, "final_deeplabv3_ebimage_counting.h5")
model %>% save_model_hdf5(final_model_path)
cat("Training completed! Model saved to:", final_model_path, "\n")