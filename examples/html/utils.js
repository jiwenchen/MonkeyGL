function decodeVR(buffer) {
    jdata = buffer.data;
    img_b64 = jdata.image;
    let base64Buffer = _loadBase64(img_b64, false);
    let png = new PNG(base64Buffer);
    let pixelArray = png.decodePixels();
    let isColor = true;
    let width = png.width;
    let height = png.height;
    let window_center = 128;
    let window_width = 256;
    let pixel_spacing_x = 1;
    let pixel_spacing_y = 1;

    let rescale_intercept = 0;
    let rescale_slope = 1;
    let invert = false;
    let signed = false;

    let buf = (pixelArray.length / 3) * 4; // RGB32
    let pixels;
    if (signed) {
        pixels = new Int8Array(buf);
    } else {
        pixels = new Uint8Array(buf);
    }
    pixelArray = _convertPixel(pixels, pixelArray);

    let pixelValues = _getPixelValues(pixelArray);
    let minPixelValue = pixelValues.minPixelValue;
    let maxPixelValue = pixelValues.maxPixelValue;
    if (!window_center || !window_width) {
        if (!isColor) {
        window_center = (maxPixelValue - minPixelValue) / 2 + minPixelValue;
        window_width = maxPixelValue - minPixelValue;
        } else {
        window_center = 128;
        window_width = 256;
        }
    }

    let cornerstoneMetaData = {
        color: isColor,
        columnPixelSpacing: pixel_spacing_y,
        rowPixelSpacing: pixel_spacing_x,
        columns: width,
        rows: height,
        originalWidth: width,
        originalHeight: height,
        width,
        height,
        intercept: rescale_intercept,
        invert: !!invert,
        isSigned: !!signed,
        maxPixelValue: maxPixelValue,
        minPixelValue: minPixelValue,
        sizeInBytes: pixelArray.byteLength,
        slope: rescale_slope,
        windowCenter: window_center,
        windowWidth: window_width,
        getPixelData: () => pixelArray,
    };
    return cornerstoneMetaData;
}

function decodeMPR(buffer) {
    jdata = buffer.data;
    img_b64 = jdata.image;

    let base64Buffer = _loadBase64(img_b64, false);
    let png = new PNG(base64Buffer);
    let pixelArray = png.decodePixels();
    let width = png.width * 2;
    let height = png.height;

    isColor = false;
    window_center = 40;
    window_width = 400;
    pixel_spacing_x = 1;
    pixel_spacing_y = 1;

    rescale_intercept = 0;
    rescale_slope = 1;
    invert = false;
    signed = true;

    pixelArray = new Int16Array(pixelArray.buffer);

    let pixelValues = _getPixelValues(pixelArray);
    let minPixelValue = pixelValues.minPixelValue;
    let maxPixelValue = pixelValues.maxPixelValue;
    if (!window_center || !window_width) {
        if (!isColor) {
        window_center = (maxPixelValue - minPixelValue) / 2 + minPixelValue;
        window_width = maxPixelValue - minPixelValue;
        } else {
        window_center = 128;
        window_width = 256;
        }
    }

    let cornerstoneMetaData = {
        color: isColor,
        columnPixelSpacing: pixel_spacing_y,
        rowPixelSpacing: pixel_spacing_x,
        columns: width,
        rows: height,
        originalWidth: width,
        originalHeight: height,
        width,
        height,
        intercept: rescale_intercept,
        invert: !!invert,
        isSigned: !!signed,
        maxPixelValue: maxPixelValue,
        minPixelValue: minPixelValue,
        sizeInBytes: pixelArray.byteLength,
        slope: rescale_slope,
        windowCenter: window_center,
        windowWidth: window_width,
        getPixelData: () => pixelArray,
    };
    return cornerstoneMetaData;
}

function _loadBase64(base64Str, isBuffer=true) {
    let binary_string = window.atob(base64Str);
    let len = binary_string.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binary_string.charCodeAt(i);
    }

    return isBuffer ? bytes.buffer : bytes;
}

function _convertPixel(targetPixel, pixelArray) {
    let index = 0;
    for (let i = 0; i < pixelArray.length; i += 3) {
        targetPixel[index++] = pixelArray[i];
        targetPixel[index++] = pixelArray[i + 1];
        targetPixel[index++] = pixelArray[i + 2];
        targetPixel[index++] = 255; // Alpha channel
    }
    return targetPixel;
    }

    function _getPixelValues(pixelData) {
    let minPixelValue = Number.MAX_VALUE;
    let maxPixelValue = Number.MIN_VALUE;
    let len = pixelData.length;
    let pixel = void 0;

    for (let i = 0; i < len; i++) {
        pixel = pixelData[i];
        minPixelValue = minPixelValue < pixel ? minPixelValue : pixel;
        maxPixelValue = maxPixelValue > pixel ? maxPixelValue : pixel;
    }

    return {
        minPixelValue: minPixelValue,
        maxPixelValue: maxPixelValue,
    };
    }