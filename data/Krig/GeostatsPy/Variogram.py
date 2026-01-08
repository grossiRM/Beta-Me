import numpy as np


def GAM(array, tmin, tmax, xsiz, ysiz, ixd, iyd, nlag, isill):              # _______________________________ Variogram _______________________________
    if array.ndim == 2:         ny, nx = array.shape
    elif array.ndim == 1:       ny, nx = len(array),1  ; array = array.reshape((ny,1))

    nvarg = 1   ;mxdlv = nlag      ;lag = np.zeros(mxdlv)      ;vario = np.zeros(mxdlv)        ;hm = np.zeros(mxdlv)       ;tm = np.zeros(mxdlv)  ; npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)   ;ivtail[0] = 0              ;ivhead = np.zeros(nvarg + 2)   ;ivhead[0] = 0              ; ivtype = np.zeros(nvarg + 2)   ; ivtype[0] = 0
    inside = (array > tmin) & (array < tmax)    ; stdev = array[(array > tmin) & (array < tmax)].std()      ; var = stdev ** 2.0

    for iy in range(0, ny):
        for ix in range(0, nx):
            if inside[iy, ix]:
                vrt = array[iy, ix]    ;ixinc = ixd   ;iyinc = iyd   ;ix1 = ix  ;iy1 = iy
                for il in range(0, nlag):
                    ix1 = ix1 + ixinc
                    if 0 <= ix1 < nx:   
                        iy1 = iy1 + iyinc
                        if 1 <= iy1 < ny:
                            if inside[iy1, ix1]:
                                vrh = array[iy1, ix1]  ; npp[il] = npp[il] + 1  ; tm[il] = tm[il] + vrt ; hm[il] = hm[il] + vrh  ; vario[il] = vario[il] + ((vrh - vrt) ** 2.0)
    for il in range(0, nlag):
        if npp[il] > 0:
            rnum = npp[il]  ;lag[il] = np.sqrt((ixd * xsiz * il) ** 2 + (iyd * ysiz * il) ** 2) ;vario[il] = vario[il] / float(rnum) ;hm[il] = hm[il] / float(rnum) ; tm[il] = tm[il] / float(rnum)
            if isill == 1:   vario[il] = vario[il] / var
            vario[il] = 0.5 * vario[il]
    return lag, vario, npp