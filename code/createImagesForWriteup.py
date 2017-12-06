import numpy as np
import cv2

import createBasisWeights as cbw

n = 15
cols = 2 * n + 1
rows = cols 
basis = np.reshape(cbw.createBasisWeightsGaussian(15,1)[10,:,:],(rows,cols))
angle = 45
M = cv2.getRotationMatrix2D((cols/2.0-0.5,rows/2.0-0.5),angle,1)
Mid = cv2.getRotationMatrix2D((cols/2.0-0.5,rows/2.0-0.5),0,1)

rotimgcub = cv2.warpAffine(basis,M,(rows,cols),flags=cv2.INTER_CUBIC)
rotimg = cv2.warpAffine(basis,M,(rows,cols))
print(np.mean(np.abs(basis)))
print(np.mean(np.abs(basis-rotimg)))
print(np.mean(np.abs(basis-rotimgcub)))
#bbasis= cv2.warpAffine(basis,Mid,(rows,cols))#,flags=cv2.INTER_NEAREST)
basis = cv2.resize(basis, (210,210),interpolation=cv2.INTER_NEAREST)
rotimg = cv2.resize(rotimg, (210,210),interpolation=cv2.INTER_NEAREST)
#roundtrip = cv2.resize(roundtrip, (210,210),interpolation=cv2.INTER_NEAREST)
#cv2.imshow("basis", basis)
#cv2.imshow("rotbasis",rotimg) 
#cv2.imshow("rotdiff", (rotimg - basis))
#cv2.imshow("rounddiff", (roundtrip - basis))
cv2.imwrite("../writeup/basis.png", basis*255)
cv2.imwrite("../writeup/rotbasis45.png", rotimg*255)
diff = np.array((basis-rotimg) * 1000, dtype=np.uint8)
imc = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
cv2.imwrite("../writeup/diffbasisrot45.png", imc)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
