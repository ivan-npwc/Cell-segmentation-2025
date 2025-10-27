
library(EBImage)


pth="C:\\Users\\usato\\Documents\\YandexDisk\\CURRENT WORK\\CELL SEGMENTATION 20241007\\DATA\\ORIG\\NAGY\\20241118\\Image\\VLF_07_a1.JPEG"



 # Read and preprocess image
   msk = readImage(pth)
   
   # Create initial mask
   nmask = thresh(msk, 25, 25, 0.001)
   nmask1 = fillHull(nmask)
   
   # Morphological operations
   nmask2 = opening(nmask1, makeBrush(7, shape='disc'))
   nmask3 = fillHull(nmask2)
   nmask3 = erode(nmask3, makeBrush(9, shape='disc'))
   nmask4 = bwlabel(nmask3)
   
   # Process labeled mask
   nmask7 = bwlabel(nmask4)[,,1]
   colorMode(nmask7) = Grayscale
   
   # Create visualization
   img1 = channel(msk, 'rgb')
   imgMsk = paintObjects(nmask7, img1, thick=TRUE)
   
   # Extract coordinates
   future = data.frame(computeFeatures.moment(nmask7))
   xy = data.frame(x=future$m.cx, y=future$m.cy)
   labels = c(1:length(future$m.cx))
 
   plot(getFrame(imgMsk, 1, type = "render"))
   text(x=future$m.cx, y = future$m.cy, labels=labels, col="green",cex = 1)
  


