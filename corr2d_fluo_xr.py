#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 ALBA Synchrotron
#
# Authors: Josué Gomez  & Joaquín Otón
#
# This file is part of Mistral beamline software.
# (see https://www.albasynchrotron.es/en/beamlines/bl09-mistral)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import sys

import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation
from skimage import transform as sktransform, morphology
from skimage.filters.thresholding import threshold_minimum, threshold_otsu
from scipy import ndimage, interpolate


parse = argparse.ArgumentParser(description='Find inplane alignment between '
                                'mosaic image comming from xray '
                                'microscopy and cryosim z-projection image.')
parse.add_argument('-j', '--mosaic',
                   help='x-ray mosaic image',
                   required=True)
parse.add_argument('-r', '--mosaicPxSize',
                   help='Pixel size for x-ray mosaic image',
                   required=True)
parse.add_argument('-p', '--cryoimgPxSize',
                   help='z-projection fluorecence image pixel size',
                   required=True)
args = parse.parse_args()


def rotZ2mat(gamma):
    out = np.matrix([[np.cos(gamma), -np.sin(gamma), 0 ],
                     [np.sin(gamma),  np.cos(gamma), 0 ],
                     [   0,              0,          1 ]])
    return out

def extendRadSym(linear_profiles, outNdim=2):
    iDims = linear_profiles.shape
    nDims = len(iDims)
    xSize = iDims[0]

    if nDims == 1:
        linp = np.expand_dims(linear_profiles, axis=1)
        nProfiles = 1
    elif nDims == 2:
        linp = linear_profiles
        nProfiles = iDims[1]
    else:
        raise Exception('extendRadSym: Wrong number of dimensions for input \
                        profiles')

    oSize = 2*xSize - 1
    xC = int(oSize/2)
    xV = np.arange(oSize) - xC

    # Meshgrid consider the same xV for all the dimensions. xx is a list with
    # the arrays for all the dimension
    xx = np.array(np.meshgrid(*[xV]*outNdim))
    ro = np.sqrt((xx**2).sum(axis=0))
    radSym2D = np.empty((oSize,)*outNdim + (nProfiles,))
    for k in range(nProfiles):
        f = interpolate.interp1d(xV[xC:], linp[:, k], bounds_error=False,
                                 fill_value=0., kind='cubic')
        radSym2D[..., k] = f(ro)

    return np.squeeze(radSym2D)

def maskRaisedCosineRadial(shape, radius, dx=1, pad=20):
    nDim = len(shape)
    xDim2 = round(radius/dx)
    profile1D = np.ones(xDim2)
    profile1D[xDim2 - pad:] = raisedCos(pad)[::-1]
    arrayNdim = extendRadSym(profile1D, nDim)
    return padArrayCentered(arrayNdim, shape, 'constant')[0]

def getPhaseShiftFourier(shape, shifts):
    ndim = len(shape)
    size = shape[0]
    sizeh = np.asarray(size)//2
    xV = [np.arange(size) - sizeh]
    XX = np.meshgrid(*(xV*ndim), copy=False, indexing='ij')
    phaseShift = np.zeros(shape)
    for k in range(ndim):
        phaseShift += shifts[k]/size*XX[k]

    phaseShift = np.exp(-1j*2*np.pi*phaseShift)
    return phaseShift

def griddingCorrect(array, gridCoords=None):
    if gridCoords is None:
        size = np.array(array.shape)
        sizeh = size//2
        XX = list()
        for k in range(array.ndim):
            XX.append(np.arange(size[k]) - sizeh[k])

        gridCoordsArray = np.array(np.meshgrid(*XX, copy=False))
    else:
        gridCoordsArray = np.array(gridCoords)

    ro = np.sqrt((gridCoordsArray**2).sum(axis=0))
    ro /= min(size)
    sinc2 = (np.sin(np.pi * ro) / (np.pi * ro))**2
    eps = 1e-2

    maski = np.logical_or(sinc2 < eps, ro > 1.0)
    mask = np.invert(maski)
    mask[tuple(sizeh)] = False

    arrayOut = array.copy()

    arrayOut[maski] = arrayOut[maski] / eps
    arrayOut[mask] = arrayOut[mask] / sinc2[mask]

    return arrayOut

def cropArrayCentered(arrayIn, croppedSize):
    iSize = np.array(arrayIn.shape)
    iDim = np.size(iSize)

    cSize = np.array(croppedSize)  # array croppedSize
    cDim = np.size(cSize)

    iC = np.floor(iSize/2).astype(int)
    cC = np.floor(cSize/2).astype(int)

    padpre = iC - cC
    padpost = (iSize - cSize) - padpre

    slInd = [None]*iDim
    for k in range(cDim):
        slInd[k] = slice(padpre[k],
                         None if padpost[k] == 0 else -padpost[k])

    for k in range(cDim, iDim):
        slInd[k] = slice(None)

    return arrayIn[tuple(slInd)]

def transform(array, R, padfactor=2, edge=-1, end_values=0, dimOrder=None):
    ndim = array.ndim
    iSize = np.array(array.shape)
    iSizeh = iSize//2  # Rotation center position

    sizePad = int(np.max(iSize)*padfactor)
    sizePadND = (sizePad,)*ndim
    sizePadh = sizePad//2

    if dimOrder is None:
        dimOrder = np.arange(ndim)[::-1]  # [2, 1, 0]

    if end_values == 'mean':
        end_values = array.mean()

    rcpad = sizePad//32
    arrayPad = padArrayCentered(array, sizePadND,
                                   end_values=end_values, rcpad=rcpad)[0]

    aFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayPad)))

    if edge > 0:
        mask = maskRaisedCosineRadial(sizePadND, sizePadh, pad=edge)
        aFT *= mask

    phaseShift = getPhaseShiftFourier(sizePadND, R[dimOrder, -1].A)

    R = R.copy()
    R[:-1, -1] = 0  # Only apply rotations to Fourier pattern

    aFTrot = transformRS(aFT, R, dimOrder=dimOrder)*phaseShift

    aRot = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aFTrot))).real
    aRot = griddingCorrect(aRot)
    aRot = cropArrayCentered(aRot, iSize)
    return aRot

def transformRS(array: np.ndarray, R: np.ndarray,
                Fourier=False, dimOrder=None):
    ndim = array.ndim
    iSize = np.array(array.shape)
    vC = iSize//2  # Rotation center position

    if ndim != (R.shape[0] - 1):
        raise ValueError(f"Transformation matrix dimension {R.shape[0]} does"
                         f" not match array dimensions {ndim} + 1.")

    if dimOrder is None:
        dimOrder = np.arange(ndim)[::-1]  # [2, 1, 0]

    if Fourier:
        vC[dimOrder[0]] = 0

    idimOrd = np.argsort(dimOrder)  # inverse order

    # Input coordinates for interpolation
    XXref = [None]*ndim

    for k in range(ndim):
        XXref[k] = np.arange(iSize[k]) - vC[k]  # original dimensions
    XXmesh = np.meshgrid(*XXref, copy=False, indexing='ij')
    XXoutF = [None]*(ndim+1)
    for k in range(ndim):
        XXoutF[k] = XXmesh[idimOrd[k]].ravel()
    XXoutF[-1] = np.ones(XXoutF[0].size)  # Augmented vector

    Rinv = R.I
    rotCoord = (Rinv@np.row_stack(XXoutF)).A.T

    if Fourier:
        xneg = rotCoord[:, 0] < 0
        rotCoord[xneg, :] *= -1

    rotArray = interpolate.interpn(XXref, array,
                                   rotCoord[:, dimOrder],
                                   bounds_error=False,
                                   fill_value=array[0, 0],
                                   method='linear')
    return rotArray.reshape(iSize)

def transformRS2D(array: np.ndarray, rot: float, shifts=[0, 0],
                  Fourier=False, dimOrder=None):
    A = rotZ2mat(rot)
    A[:2, 2] = np.atleast_2d(shifts).T
    return transformRS(array, A, Fourier, dimOrder)

def transform2D(array: np.ndarray, rot=0, shifts=[0, 0],
                padfactor=2, edge=-1, end_values=0, dimOrder=None):
    A = rotZ2mat(rot)
    A[:2, 2] = np.atleast_2d(shifts).T

    return transform(array, A, padfactor, edge, end_values, dimOrder)

def raisedCos(Ni):
    rc = (np.sin(np.linspace(-np.pi/2, np.pi/2, Ni)) + 1)/2
    return rc

def padwithrc(vector, pad_width, iaxis, kwargs):
    value = kwargs.get('end_values', 0)
    rcpad = kwargs.get('rcpad', -1)

    if pad_width[0] > 0:
        if rcpad < 0:
            vector[:pad_width[0]] = (raisedCos(pad_width[0]) *
                                     (vector[pad_width[0]+1]-value) + value)
        else:
            rcprof = raisedCos(rcpad)
            if rcpad < pad_width[0]:
                vector[:pad_width[0]-rcpad] = value
                vector[pad_width[0]-rcpad:pad_width[0]] = \
                    (rcprof *
                     (vector[pad_width[0]+1]-value) + value)
            else:
                vector[:pad_width[0]] = \
                    (rcprof[-pad_width[0]:] *
                     (vector[pad_width[0]+1]-value) + value)

    if pad_width[1] > 0:
        if rcpad < 0 or rcpad == pad_width[1]:
            vector[-pad_width[1]:] = \
                ((raisedCos(pad_width[1])[::-1]) *
                 (vector[-pad_width[1]-1]-value) + value)
        else:
            rcprof = raisedCos(rcpad)
            if rcpad < pad_width[1]:
                vector[-pad_width[1]:-pad_width[1]+rcpad] = \
                    ((rcprof[::-1]) *
                     (vector[-pad_width[1]-1]-value) + value)
                vector[-pad_width[1]+rcpad:] = value
            else:
                vector[-pad_width[1]:] = \
                    ((raisedCos(rcpad)[:-pad_width[1]-1:-1]) *
                     (vector[-pad_width[1]-1]-value) + value)

    return vector

def padArrayCentered(arrayIn, oSize, mode=padwithrc, **kwargs):
    if mode == padwithrc:
        unsupported_kwargs = set(kwargs) - set(['end_values', 'rcpad'])
        if unsupported_kwargs:
            raise ValueError(f"unsupported keyword arguments for mode "
                             f"'padwithrc': {unsupported_kwargs}")

    kwargs['mode'] = mode

    iSize = np.array(arrayIn.shape)
    iDim = arrayIn.ndim

    aoSize = np.array(oSize)  # array oSize
    pDim = np.size(aoSize)
    pSize = np.zeros(iDim).astype(int)  # Paddedsize autoexpanded
    pSize[:pDim] = aoSize

    if pDim < iDim:
        pSize[pDim:iDim] = iSize[pDim:iDim]

    iC = np.floor(iSize/2).astype(int)
    pC = np.floor(pSize/2).astype(int)

    padpre = pC-iC
    padpost = (pSize - iSize) - padpre

    arrayOut = np.pad(arrayIn, list(zip(padpre, padpost)), **kwargs)

    return arrayOut, padpre, padpost

def getMaskCenter(mask):
    iSize = mask.shape
    nDim = len(iSize)
    xV = [np.arange(x) for x in iSize]
    xx = np.meshgrid(*xV, indexing='ij')
    gc = np.zeros(nDim)  # Coordinates of the geometrical center
    maskSum = mask.sum()
    for k in range(nDim):
        gPos = xx[k][mask]
        gc[k] = (gPos.max() + gPos.min())/2

    return gc

def getGridMask(array, area_opening=512):
    thr = threshold_otsu(array)
    aThr = array < thr
    aMor = morphology.area_opening(aThr, area_opening)
    aLab, nLabels = morphology.label(aMor, return_num=True)
    label = 0
    areaMax = 0
    for k in range(1, nLabels+1):
        area = (aLab == k).sum()
        if area > areaMax:
            areaMax = area
            label = k
    return aLab == label

def imnorm(array):
    amin = array.min()
    amax = array.max()
    return (array - amin)/(amax -amin)

def readMosaic(fname, fnamepos=None):
    imtmp = plt.imread(fname)[:5800, :, :]
    if imtmp.ndim == 3:
        imtmp = imtmp[:, :, 0].astype(float)
    else:
        imtmp = imtmp.astype(float)
    
    thrmin = 0.5*threshold_minimum(imtmp)
    print("Selected Threshold:", thrmin)
    imtmp[0,:] = 0
    imtmp[-1,:] = 0
    imtmp[:,0] = 0
    imtmp[:,-1] = 0
    mask = skimage.segmentation.flood(imtmp,(0,0),  tolerance=thrmin)
    mask1 = skimage.morphology.dilation(mask, skimage.morphology.square(40))
    mask1 = np.logical_not(mask1)
    
    boundLabels = skimage.measure.label(mask1)
    mask2 = boundLabels == np.argmax(np.bincount(boundLabels.flat)[1:]) + 1

    imtmp[imtmp < 1] = 1
    imlog =imnorm(-np.log(imtmp))
    mean = imlog[mask2].mean()
    print("im log mean", mean)
    mosaic = imlog*mask2 + mean*np.logical_not(mask2)

    if fnamepos is not None:
        imtmp = plt.imread(fnamepos)[:5800, :, :]
        gc = getMaskCenter(imtmp[:, :, 0] - imtmp[:, :, 1] > 0)

        return mosaic, gc
    else:
        return mosaic, imtmp, mask2

def importImages(imagesfn):
    images = {}
    if 'xrpos' in imagesfn:
        images['xr'], images['coord'] = readMosaic(imagesfn['xr'],
                                                   imagesfn['xrpos'])
    else:
        images['xr'], mosaic, mask2 = readMosaic(imagesfn['xr'])
    
    images['fluo'] = plt.imread(imagesfn['fluo'])[:, :, 0]
    return images, mosaic, mask2

def cropImages(images, K):
    padK = 1.2  # padding to xr mosaic if red channel is given
    fluoSize = max(images['fluo'].shape)
    xrSizeEnd = int(fluoSize * K * padK)
    xrSizeEndh2 = xrSizeEnd // 2

    if 'coord' in images:
        gc = images['coord']
        sl = [None] * 2
        for k in [0, 1]:
            xi = 0 if gc[k] < xrSizeEndh2 else int(gc[k] - xrSizeEndh2)
            xf = xi + xrSizeEnd
            sl[k] = slice(xi, xf)

        return images['xr'][tuple(sl)]
    else:
        return images['xr']

def scalingImgs(images):
    # %%% Cropping XR image and scaling
    dxfluo = float(args.cryoimgPxSize) # 17,9 nm
    dxr = float(args.mosaicPxSize) # 10 nm
    imxrcrop = cropImages(images, dxfluo/dxr)
    
    # As we will operate in fourier space, if rescale introduce artifacts, we
    # must implement the rescaling to avoid those artifacts.
    imxrs =sktransform.rescale(imxrcrop, dxr/dxfluo, anti_aliasing=True)
    xrSizeSc = imxrs.shape
    
    imfluo = padArrayCentered(images['fluo'], xrSizeSc, rcpad=24)[0]

    imgFluoRescaled = sktransform.rescale(imxrcrop, dxfluo/dxr)
    imgFluoRescaled = cropArrayCentered(imgFluoRescaled, images['xr'].shape)[0]
    return imxrs, imfluo, imgFluoRescaled

def featuresFinding(imgxr, imgfluo):
    imxrth1 = ndimage.white_tophat(imgxr, size=28)
    imxrth2 = ndimage.white_tophat(imgxr, size=10)

    imxrLD2 = imxrth1 - imxrth2

    return imxrLD2, imgfluo

def alignment(imgXr, imgFluo):
    refFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgXr)))
    ccmax = 0
    anglemax = 0
    shiftmax = []
    
    for angle in np.linspace(0, -359, 360):
        fluorot = transform2D(imgFluo, np.deg2rad(angle), [0., 0.])
        imFTRot = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fluorot)))
        
        ccft = refFT.conj() * imFTRot
        cc = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ccft))).real
        mpos = np.unravel_index(cc.argmax(), cc.shape)
        shift = mpos - np.array(cc.shape)//2
        print(f'angle={angle} - ccmax={cc.max()}')
        if cc.max() > ccmax:
            ccmax = cc.max()
            anglemax = angle
            shiftmax = -shift
    
    fluorotfix = transformRS2D(imgFluo, np.deg2rad(anglemax), shiftmax[::-1])
    return fluorotfix, anglemax, shiftmax[::-1]

def main():
    
    listImgxrfn = glob.glob(args.mosaic)
    
    for xrFn in listImgxrfn:
        basename = xrFn[:-4]
        fluoFn = basename+'fluo.tif'
        imagesfn = {'xr': xrFn,
                    'fluo': fluoFn}
        print('filenames: ', xrFn, fluoFn)
        images, mosaic, mask2 = importImages(imagesfn)
        print("finishing import images")
        imgxr, imgfluo, imgFluoRescaled = scalingImgs(images)
        print("finishing scaling")
        imxrLD, imgfluoLD = featuresFinding(imgxr, imgfluo)
        print("finishing featuresFinding")
        fluorotfix, anglemax, shiftmax = alignment(imxrLD, imgfluoLD)
        print("MAX CC ANGLE: ", anglemax)

        saveFluoImgfn = 'fluo_aligned'+basename.split('\\')[1]+'.png'
        saveXrImgfn = 'xray_masked'+basename.split('\\')[1]+'.png'
        #
        plt.imsave(saveFluoImgfn, fluorotfix, cmap = 'hot')
        plt.imsave(saveXrImgfn, imxrLD, cmap = 'gray')

        fig, ax = plt.subplots(1, 2, num=1, clear=True)
        ax[0].imshow(imxrLD)
        ax[1].imshow(fluorotfix)
        fig.show()

if __name__ == "__main__":
    main()
    input('Press ENTER to exit')
