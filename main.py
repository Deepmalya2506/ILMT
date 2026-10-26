import ilmt_streak_sf as S

img, hdr = S.load_fits("./ILMT_Contest_Materials/test_files/20221026-0057.fits")
res, pre = S.detect_frame(img, S.Config())
S.annotate(img, res, pre, "frame", save="out.png", show_mask=True)   