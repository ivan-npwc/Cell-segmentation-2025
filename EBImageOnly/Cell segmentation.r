
  library(EBImage)
  dir= "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\cell 2025\\data\\kamb N2--TL23-L_counting_selected_frames\\kamb N2--TL23-L_counting_selected_frames\\VLF_23\\E13"
  
  lstfls= list.files(dir, full.names=T)
  
  imgCount=length(lstfls)
  
  for (i in 1:length(lstfls)){
  if (i ==1){
    pth=lstfls[i]
    img = readImage(pth)
	img0=NaigFunction(img)
	img0=img0/imgCount

    } else{
	 pth=lstfls[i]
    img1 = readImage(pth)
	img1=NaigFunction(img1)
	  img1=  img1/imgCount
	img0 = img0 + img1
	
	}
  }
  
######################################################################
NaigFunction=function(img){

 img_norm <- normalize(img)
# Увеличение контраста (можно регулировать параметры)
img_contrast <- 2 * (img_norm - 0.2)
img_contrast[img_contrast < 0] <- 0
img_contrast[img_contrast > 1] <- 1

img_blur <- gblur(img_contrast, sigma = 1)


# Бинаризация (метод Оцу)
img_th <- thresh(img_blur, w=50, h=50, offset=0.1)
img_bw <- img_th > 0.5

 kern <- makeBrush(13, shape='disc')
 img_clean <- closing(opening(img_bw, kern), kern)  # закрытие мелких разрывов
 

# Заполнение небольших отверстий внутри клеток
img_fill <- fillHull(img_clean)

# 4. Подсчет и анализ клеток
# Подсчет меток (каждая метка - одна клетка)
labels <- bwlabel(img_fill)
cell_count <- max(labels)

# Визуализация результатов
#par(mfrow=c(2,2))
#plot(img)
#plot(img_eq)
#plot(img_bw)
#plot(colorLabels(labels))
return(img_fill)
}

##########################################################################

 for (i in 1:length(lstfls)){
  if (i ==1){
    pth=lstfls[i]
    img = readImage(pth)

	img=img/imgCount

    } else{
	 pth=lstfls[i]
    img1 = readImage(pth)
	  img1=  img1/imgCount
	img = img + img1
	
	}
  }
  





 kern <- makeBrush(3, shape='disc')
 
 img_bw <- img0 > 0.25
 display(img_bw)

 
 img0_clean <- closing(opening(img0, kern), kern)  # закрытие мелких разрывов
 display(img0_clean)









# 4. Подсчет и анализ клеток
# Подсчет меток (каждая метка - одна клетка)
labels <- bwlabel(img0)
cell_count <- max(labels)
par(mfrow=c(2,2))
plot(img)
plot(img_eq)
plot(img0)
plot(colorLabels(labels))

# В