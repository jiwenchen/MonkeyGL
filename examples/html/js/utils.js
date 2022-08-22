function decodeVR(buffer) {
    const img_b64 = buffer.data.image;
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
    const img_b64 = buffer.data.image || buffer.data;

    let base64Buffer = _loadBase64(img_b64, false);
    let png = new PNG(base64Buffer);
    let pixelArray = png.decodePixels();
    let width = png.width * 2;
    let height = png.height;

    let isColor = false;
    let window_center = 40;
    let window_width = 400;
    let pixel_spacing_x = 1;
    let pixel_spacing_y = 1;

    let rescale_intercept = 0;
    let rescale_slope = 1;
    let invert = false;
    let signed = true;

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

function _loadBase64(base64Str, isBuffer = true) {
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

function getTwoPointsDistance(sourcePt, targetPt) {
    // This function is for finding distance between two points(sourcePt, targetPt)
    let distance = Math.sqrt(
        (sourcePt.x - targetPt.x) * (sourcePt.x - targetPt.x) +
        (sourcePt.y - targetPt.y) * (sourcePt.y - targetPt.y)
    );
    return distance;
}

function findIntersectionPointsCircleLine(circleCenter, radius, lineAtrr, standardPt) {
    //  Find the Points of Intersection of a Circle with a Line
    //y=k*x+b        (𝑥−x0)2+(𝑦−y0)2=𝑟2
    //lineAtrr:slope->k,yAxis->b
    let points = [];
    let pt1, pt2;
    let x0 = circleCenter.x;
    let y0 = circleCenter.y;
    let k = lineAtrr.slope;
    let b = lineAtrr.yAxis;
    //substitute  y=k*x+b  into (𝑥−x0)2+(𝑦−y0)2=𝑟2
    //𝐴𝑥2+𝐵𝑥+𝐶=0
    let sqrtNum;
    if (k !== Infinity) {
        sqrtNum = Math.sqrt(
            4 * Math.pow(k * b - k * y0 - x0, 2) -
            4 * (1 + k * k) * (x0 * x0 + Math.pow(b - y0, 2) - radius * radius)
        );
    }
    if (k === Infinity) {
        //the vertical lines 𝑥=𝑘
        pt1 = {
            x: x0,
            y: y0 - radius
        };
        pt2 = {
            x: x0,
            y: y0 + radius
        };
        points = [pt1, pt2];
    } else if (sqrtNum === 0) {
        // one intersection point
        let x = (2 * x0 + 2 * k * y0 - 2 * k * b + sqrtNum) / (2 * (1 + k * k));
        let y = k * x + b;
        pt1 = {
            x: x,
            y: y
        };
        points = [pt1];
    } else if (sqrtNum > 0) {
        //two intersection points
        let x1 = (2 * x0 + 2 * k * y0 - 2 * k * b - sqrtNum) / (2 * (1 + k * k));
        let x2 = (2 * x0 + 2 * k * y0 - 2 * k * b + sqrtNum) / (2 * (1 + k * k));
        let y1 = k * x1 + b;
        let y2 = k * x2 + b;
        pt1 = {
            x: x1,
            y: y1
        };
        pt2 = {
            x: x2,
            y: y2
        };
        points = [pt1, pt2];
        if (standardPt) {
            let d1 = getTwoPointsDistance(pt1, standardPt);
            let d2 = getTwoPointsDistance(pt2, standardPt);
            if (d1 > d2) {
                points = [pt2, pt1];
            }
        }
    }

    return points;
}
function keepLinePointInImage(image, eq) {
    // This function is for finding two intersection points on image(or something like rectangle)
    // ref) https://blog.csdn.net/qq_43046501/article/details/105518929
    let w = image.width;
    let h = image.height;
    let res = [];
    let x1 = 0;
    let y1 = eq.yAxis;
    let x2 = w;
    let y2 = eq.slope * x2 + eq.yAxis;
    let y3 = h;
    let x3 = (h - eq.yAxis) / eq.slope;
    let y4 = 0;
    let x4 = -eq.yAxis / eq.slope;
    if (eq.slope === Infinity && eq.yAxis >= 0) {
        res = [
            {
                x: eq.yAxis,
                y: 0
            },
            {
                x: eq.yAxis,
                y: h
            }
        ];
    } else if (eq.slope === 0 && eq.yAxis >= 0) {
        res = [
            {
                x: 0,
                y: eq.yAxis
            },
            {
                x: w,
                y: eq.yAxis
            }
        ];
    } else {
        if (y1 <= h && y1 >= 0) {
            res.push({
                x: x1,
                y: y1
            });
        }
        if (y2 <= h && y2 >= 0) {
            res.push({
                x: x2,
                y: y2
            });
        }
        if (x3 < w && x3 > 0) {
            res.push({
                x: x3,
                y: y3
            });
        }
        if (x4 < w && x4 > 0) {
            res.push({
                x: x4,
                y: y4
            });
        }
    }

    if (res.length !== 2) {
        console.warn("keepLinePointInImage waring! cannot find interacition points. Used default points");
        res = [
            {
                x: x1,
                y: y1
            },
            {
                x: x2,
                y: y2
            }
        ];
    }

    // check if return data is 2 points
    return res;
}

function drawEmptyCircle(context, curPt, color, lineWidth, radius) {
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.strokeStyle = color;
    context.beginPath();
    context.lineWidth = lineWidth;
    context.arc(curPt.x, curPt.y, radius, 0, 2 * Math.PI);
    context.stroke();
    context.restore();
}
