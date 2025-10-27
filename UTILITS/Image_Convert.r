
library(EBImage)
library (magick)




ImagePath= "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELLS SEGMENTATION\\TRAIN\\TRAIN\\Image"
MaskPath= "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELLS SEGMENTATION\\TRAIN\\TRAIN\\Mask"
MaskDir_jpg = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELLS SEGMENTATION\\TRAIN\\TRAIN\\Image_jpg"


List=list.files(ImagePath, full.names=T)



 for (i in 1: length(MskList)){

    pth     = List[i]
	name    =basename(pth)
	NEWNAME = gsub("tif","JPEG",pth))
    SAVEPTH= paste0(MaskDir_jpg,"\\",NEWNAME)
	img=readImage(ImgP)
  
   writeImage(img,SAVEPTH)
   

}
