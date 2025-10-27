    #IMG RENAME AND COPY
	
	dirfrom= "C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA 20251010\\ORIG"
	dirto ="C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA 20251010\\IN"
	
	lstdir =list.files(dirfrom,full.names=T)
	
	for (i in 1:length(lstdir)){
	dir= lstdir[i]
	bsname=basename(dir)
	lstimgs=list.files(dir, full.names=T)
	copytto=paste0(dirto,"\\",bsname,"_",basename(lstimgs))
	file.copy(lstimgs,copytto)
	}



library(EBImage)
List =list.files(dirto, full.names=T)


 for (i in 1: length(List)){

		pth     = List[i]
		name    =basename(pth)
		NEWNAME = gsub("tif","JPEG",name)
		SAVEPTH= paste0(dirto,"\\",NEWNAME)
		img=readImage(pth)
		writeImage(img,SAVEPTH)
   

}