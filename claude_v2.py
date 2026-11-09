import numpy as np, ilmt_streak_sf as S
from scipy import ndimage as ndi
img, hdr = S.load_fits("./ILMT_Contest_Materials/test_files/20221026-0057.fits")
pre = S.preprocess(img, S.Config())
rms, w, mask, res = pre['rms'], pre['w'], pre['mask'], pre['res']
print("rms pctl 0,1,50,99,100:", np.percentile(rms,[0,1,50,99,100]))
print("pixels rms<0.3*median:", (rms<0.3*np.median(rms)).sum())
y,x = np.unravel_index(np.argmax(w), w.shape); print("max-weight pixel x,y:", x, y)
v,c = np.unique(img, return_counts=True); print("most common value:", v[c.argmax()], c.max()/img.size)
print("PSF, n_src, n_compact, n_bleed:", pre['psf'], pre['n_src'], pre['n_compact'], pre['n_bleed'])
print("mask %:", mask.mean()*100)
print("mask over vertical line x~550:", mask[1000:2500,540:560].mean())
print("mask over horizontal line y~400:", mask[390:410,1000:3500].mean())
u = np.where(mask, 0, res); print("unmasked px >200 ADU:", (u>200).sum())
lab,n = ndi.label(mask); sz = ndi.sum(mask,lab,range(1,n+1))
print("largest masked blobs:", [(int(s), ndi.find_objects(lab)[i]) for i,s in sorted(enumerate(sz),key=lambda t:-t[1])[:4]])