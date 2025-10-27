#RANDOM DEFORMATION 
 library(magick)
 library(parallel)
 library(doParallel)
 library(foreach)
 
 dirImg = "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\Image_randomCrop_380"
 dirMsk = "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\Mask_randomCrop_380"
 
 
 DirSaveImg = "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\Image_randomDeformation"
 DirSaveMsk = "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\Mask_randomDeformation"
 
 
 lstImgs=list.files(dirImg, full.names=T)
 
 
 cl <- makePSOCKcluster(detectCores (logical = FALSE)-1)
clusterEvalQ(cl, {
  library(magick)     
}
 
 registerDoParallel(cl)
 
  foreach(i = 1:length(lstImgs)) %dopar% {	
 
 #for (i in 1:length(lstImgs)) {
 
 defIndex= sample(c(-0.4,-0.3,-0.25,-0.2,-0.15,-0.1,0.1,0.15,0.2,0.25,0.3,0.4))[1]
 
 pthImg = lstImgs[i]
 Name=basename(pthImg)
 NameMsk=gsub("jpg", "png", Name)
 pthMsk=paste0(dirMsk,"\\", NameMsk)
 
   Img=image_read(pthImg)
   Msk=image_read(pthMsk)

            Img_implode = image_implode(Img, factor = defIndex)
			Msk_implode = image_implode(Msk, factor = defIndex)
			
  ImgNewName=paste0(i,"#", Name)
  MskNewName=paste0(i,"#", NameMsk)
  
  ImgNewPth=paste0(DirSaveImg,"\\",ImgNewName)
  MskNewPth=paste0(DirSaveMsk,"\\",MskNewName)
			
        image_write(Img_implode,ImgNewPth)
		image_write(Msk_implode,MskNewPth)
		
		
		}