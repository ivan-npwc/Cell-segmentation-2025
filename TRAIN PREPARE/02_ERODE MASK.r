# erode

 library(EBImage)
 dirMsk ="C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\Mask_randomCrop_380"
 
 listMsk=list.files(dirMsk, full.names=T)
 
 for (i in 1:length(listMsk)) {
 
  pth = listMsk[i]
  Msk = readImage(pth)
  Msk = erode(Msk, makeBrush(7, shape='disc'))
  writeImage(Msk,pth)
 
 }
 

 