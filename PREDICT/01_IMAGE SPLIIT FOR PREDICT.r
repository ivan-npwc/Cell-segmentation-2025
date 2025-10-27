#IMAGE SPLIIT FOR PREDICT with overlap

library(EBImage)




 DirImg = "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA 20251010\\IN"
 DirSave= "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA 20251010\\IN_split"
 dir.create(DirSave,showWarnings=F)
 
 lstImgs=list.files(DirImg, full.names=T,pattern=".JPEG")
 
 
 for (i in 1:length(lstImgs)){
 
 pth = lstImgs[i]
 Name = basename(pth)
 img = readImage(pth)
 img <- resize(img, w = 1024, h = 1024)
 #dim(img)=c(1024,1024,1)
 ####add border

                     UpOver =   img[0:1024,0:62,]
                     flip_UpOver=flip(UpOver)
                     DownOver = img[0:1024,963:1024,]
                     flip_DownOver=flip(DownOver)
                     UpDownOver = abind(flip_UpOver,img,flip_DownOver, along = 2)

                     LeftOver =UpDownOver[0:62,0:1148,]
                     LeftOverFlop=flop(LeftOver)
 
                     RightOver = UpDownOver[963:1024,0:1148,]
                     RightOverFlop=flop(RightOver)
                     OverImg=abind(LeftOverFlop,UpDownOver,RightOverFlop,along=1)
#################################   GET COORDS CENTR EACH TILES
 seqXY= seq(190,1024,by=256) # TILES SIZE 256 WITHOUT OVERLUP
 
 line1=rep(seqXY[1],4)
 line2=rep(seqXY[2],4)
 line3=rep(seqXY[3],4)
 line4=rep(seqXY[4],4)
 
 Coords_line_1= cbind(seqXY, line1)
 Coords_line_2= cbind(seqXY, line2)
 Coords_line_3= cbind(seqXY, line3)
 Coords_line_4= cbind(seqXY, line4)
 
 coordsCentr=data.frame(rbind(Coords_line_1,Coords_line_2,Coords_line_3,Coords_line_4))
 names(coordsCentr)=c("x","y")
 ###################################
 #  labels=c(1:16)
#   #png(file="test.png",width = 1148, height = 1148)
#   plot(getFrame(OverImg2, 1, type = "render"))
#   text(x=coordsCentr[,1], y = coordsCentr[,2], labels=labels, col="green",cex = 2)
  # dev.off()
 #########################
  ############################## DISPLAY TILES and overlup
 # OverImg1 = untile(OverImg, c(4, 4)) # split without overlup
 # display(OverImg1, title='Blocks')
#  OverImg2 = tile(OverImg1, 4)       # combine
#  display(OverImg2, title='Tiles')
  
 #    png(file="OverlapExample.png",width = 1145, height = 1145)
	 
 #    plot(getFrame(OverImg2, 1, type = "render"))
#	 rect(xleft=60, ybottom=1086, xright=1086, ytop=60, lwd=2,lty=2)
#	 text(x=coordsCentr[,1], y = coordsCentr[,2], labels=labels, col="green",cex = 4)
 #    rect(xleft=236, ybottom=616, xright=616, ytop=236, lwd=1,lty=1,border=2)
#	 rect(xleft=0, ybottom=340, xright=340, ytop=0, lwd=1,lty=1,border=2)
	 
#   dev.off()
  

 #############################
 OverLapFromCentr = 190 # 25% OF 256 TILES SIZE 
 #tile size=256, over=64, tottal 384*384
  for (x in 1: length(coordsCentr[,1])){
	 
	centr=as.numeric(coordsCentr[x,])
	x_start = centr[1] - OverLapFromCentr
	x_stop = centr[1] + OverLapFromCentr
	
	y_start = centr[2] - OverLapFromCentr
	y_stop = centr[2] + OverLapFromCentr

	xInt=c(x_start:x_stop)
	yInt=c(y_start:y_stop)
	 
	crop=OverImg[xInt,yInt,]
	crop=resize(crop, w = 380, h = 380)
	NameCrop=paste0(x,"#",Name) 
	SavePth=paste0(DirSave,"\\",NameCrop)
    writeImage(crop,SavePth)
  
  
  
  }
  }
 
 
 
 
 
 
 
 
 
 
 
 
 