 # RANDOM CROP 381 ##dim256+25% over= 381 
 
 
 library(EBImage)
 
 dirImgFrom  =  "/home/ivan/TRAIN/CELL/Naig20251226/Image"
 dirMskFrom  =  "/home/ivan/TRAIN/CELL/Naig20251226/Mask"
 
 dirImgTo =  "/home/ivan/TRAIN/CELL/Naig20251226/Image_random_crop"
 dirMskTo =   "/home/ivan/TRAIN/CELL/Naig20251226/Mask_random_crop"
 
 unlink(dirImgTo,recursive=T)
 unlink(dirMskTo,recursive=T)
 dir.create(dirImgTo)
 dir.create(dirMskTo)
 
 tilesize=380
 
 lstImgs = list.files(dirImgFrom, full.names=T)
 
 for (y in 1:length(lstImgs)){
  pthImg = lstImgs[y]
  NameImg = basename(pthImg)
  NameMsk = gsub("jpg","png",NameImg)
  pthMsk = paste0(dirMskFrom,"\\",NameMsk)
  img = readImage (pthImg)
  msk= readImage(pthMsk)
  
   for (i in 1:100){
   CentrX = sample(200:830)[1]
   CentrY = sample(200:830)[1]
    
   StartX = CentrX-190
   StopX = CentrX+190
  
   StartY = CentrY-190 
   StopY = CentrY+190
   
   IntervalX = StartX:StopX
   IntervalY = StartY:StopY
 
  Img_Random_Crop = img[IntervalX,IntervalY,]
  Msk_Random_Crop = msk[IntervalX,IntervalY,]
  
  ImgNewName = paste0(i,"_",NameImg)
  MskNewName = paste0(i,"_",NameMsk)
  
  ImgPthSave = paste0(dirImgTo,"\\",ImgNewName)
  MskPthSave = paste0(dirMskTo,"\\",MskNewName)
  
  writeImage(Img_Random_Crop,ImgPthSave)
  writeImage(Msk_Random_Crop,MskPthSave)
  
 }
 }