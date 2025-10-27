#MASK MERGE AFTER PREDICT
#OVERLAP= 62

library(EBImage)




 DirImg = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA predict 20251010\\IN_predicted Mask"
 DirSave=  "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA predict 20251010\\Merg_PredictedMask"
 lstTls=list.files(DirImg, full.names=T)
 unlink(DirSave,recursive=T); dir.create(DirSave)
 
 lstTls=data.frame(pth=lstTls,name=basename(lstTls))
  for (e in 1:length(lstTls[,1])){
      lstTls$tiles[e]=  strsplit(as.character(lstTls$name[e]) ,split = "#" )[[1]][1]
	  lstTls$img[e]=  strsplit(as.character(lstTls$name[e]) ,split = "#" )[[1]][2]
 }
 
 lstImgs=unique(lstTls$img)
 
 for (i in 1:length(lstImgs)){
 
    img=lstImgs[i]
	pthSaveImg=paste0(DirSave,"\\",img)
	subtbl=lstTls[lstTls$img==img,]
	subtbl$tiles=as.numeric(subtbl$tiles)
	subtbl=subtbl[order(subtbl$tiles),]
	pths= as.character(subtbl$pth)
	tiles=readImage(pths)
	tiles=transpose(tiles)
	tilesCrop= tiles[63:319, 63:319,]
    tilesCrop = tile(tilesCrop, 4,lwd=0)       # combine
	tilesCrop=resize(tilesCrop, w = 1024, h = 1024)
   # display(tilesCrop, title='Tiles')
	
	writeImage(tilesCrop,pthSaveImg)
 
 
 
 }
 }