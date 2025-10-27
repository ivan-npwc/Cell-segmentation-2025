
library(EBImage)
library (magick)




ImagePath= "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\ORIG\\IGOR'\\Image"
MaskPath= "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\ORIG\\IGOR'\\Mask"
PathCheck=  "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\ORIG\\IGOR'\\Image_Mask"
Mask_erodeDir = "C:\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA TRAIN\\ORIG\\IGOR'\\Mask_erode"

dir.create(PathCheck)
MskList=list.files(MaskPath)


#
#library(parallel)
#library(doParallel)
#library(foreach)

#cl <- makePSOCKcluster(4) 
#clusterEvalQ(cl, {
#library(EBImage)
#})
	
#registerDoParallel(cl)


#foreach(i = 1:length(MskList)) %dopar% {
Morf=NULL
for (i in 1:length(MskList)){

    pth=MskList[i]
	PathCheckImg=paste0(PathCheck,"/",pth)
    mskP=paste0(MaskPath,"/",pth) 
    ImgP= paste0(ImagePath,"/",gsub("png","jpg",pth))
	
	MskErodeSavePth=paste0(Mask_erodeDir,"\\", pth)
  if (file.exists(ImgP)==F){ImgP= paste0(ImagePath,"/",gsub("png","JPEG",pth))} 
  

	img=readImage(ImgP)
    msk=readImage(mskP)
	   colorMode(msk) = Grayscale
	   msk=msk[,,1]
######################blob analis	
	 #  nmask = thresh(msk, 25, 25, 0.01)  
     #  nmask1 <- fillHull(nmask)
     #  nmask2 = opening(nmask1, makeBrush(7,shape='disc') ) # shape='Gaussian', sigma=50 
     #  nmask3 = fillHull(nmask2)
	   nmask3=erode(msk, makeBrush(9, shape='disc'))  # nmask3
	   nmask3.1 <- fillHull(nmask3)
       nmask4 = bwlabel( nmask3.1)
           writeImage(nmask4, MskErodeSavePth )

	    shapeFeatures <- computeFeatures.shape(nmask4)
	#	dtshapeFeatures=data.frame(shapeFeatures)
	   # numtoExl=row.names(dtshapeFeatures[dtshapeFeatures$s.area < 1100,])
	    #nmask5= rmObjects(nmask4, numtoExl, reenumerate=FALSE)
		
		#nmask5=nmask4
		
	#	shapeFeatures1 =  computeFeatures.shape(nmask5)
	#	dtshapeFeatures1=data.frame(shapeFeatures1)
	#	numtoExl1=row.names(dtshapeFeatures1[dtshapeFeatures1$s.radius.max > 150 ,])
		#nmask6= rmObjects(nmask5, numtoExl1, reenumerate=FALSE)
		
  #  dtshapeFeatures$img=mskP
  #  Morf=rbind(dtshapeFeatures,Morf)
	
	#colorMode(img) = Grayscale
	#colorMode(msk) = Grayscale
	#img=img[,,1]
	#msk=msk[,,1]
	
#	nmask6=nmask4
  #  nmask = bwlabel(nmask6)
   
           img1 = channel(img, 'rgb')
           imgMsk=paintObjects(nmask4, img1, thick=TRUE)
   
            future=data.frame(computeFeatures.moment(nmask4))  # coordinat
			xy=data.frame(x=future$m.cx,  y = future$m.cy)
            labels=c(1:length(future$m.cx))
   
   png(file=PathCheckImg,width = 1024, height = 1024)
   plot(getFrame(imgMsk, 1, type = "render"))
   text(x=future$m.cx, y = future$m.cy, labels=labels, col="green",cex = 1.5)
  
   dev.off()

}
head(Morf)
