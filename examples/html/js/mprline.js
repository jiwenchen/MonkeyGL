const TYPES = ['axial', 'sagittal', 'coronal'];
let MPR_ELEMENTS = [];
let initStatus = {};
function initCrosshairs(type) {
    if (!initStatus[type]) {
        initStatus[type] = true;
        showCrosshairs(type)
    }
}
function showCrosshairs(type) {
    MPR_ELEMENTS = TYPES.map(id =>
        document.getElementById(id)
    )
    let list = [...MPR_ELEMENTS];
    if(type){
        list = [document.getElementById(type)]
    }
    list.forEach(item => {
        cornerstoneTools.mouseInput.enable(item);
        const mprOperateLine = new MprOperateLine();
        let { image } = cornerstone.getEnabledElement(item) || { width : 200, height: 200};
        let params = { crosshair: { x: image.width / 2, y: image.height / 2 }, planeType: item.id };
        mprOperateLine.initMprLines(item, params);
        mprOperateLines.push(mprOperateLine);
        $(item).on(
            "CornerstoneToolsMprOperatePositionModified",
            mprPositionModified
        );
    });


}

function setOperaterLinePosByElement(vaItem, enabledElement, evt) {
    if (enabledElement) {
        let operaterLine = getMprOperateLine(enabledElement);
        let eeImage = cornerstone.getEnabledElement(enabledElement).image;
        operaterLine.setLinePos(vaItem, eeImage, evt);
    } else {
        console.log("Error: there is no enabled element for setting operater line pos by element");
    }
}



let mprOperateLines = [];
let thicknessDistance = 0;



function MprOperateLine() {
    var horizontalLine = {};
    var verticalLine = {};
    var centerCircle = {};
    var hThicknessLine = {};
    var vThicknessLine = {};

    var thickAlpha = 50;
    var centerDiameter = 6;
    let centerWidthParam = 26;
    let isEditing = false;
    ///////// BEGIN ACTIVE TOOL ///////
    function initMprLines(element, params) {
        this.element = element;
        let enabledElement = cornerstone.getEnabledElement(element);
        let image = enabledElement.image;
        let {crosshair, planeType} = params;
        centerDiameter = Math.ceil(image.width * 0.02);
        if (centerDiameter % 2 !== 0) centerDiameter += 1;
        centerDiameter = centerDiameter + isEditing * centerWidthParam;
        let radius = centerDiameter / 2;
        let controlRadius = thicknessDistance + thickAlpha;

        horizontalLine = {
            firstStartPoint: {
                x: 0,
                y: crosshair.y
            },
            firstEndPoint: {
                x: crosshair.x - radius,
                y: crosshair.y
            },
            lastStartPoint: {
                x: crosshair.x + radius,
                y: crosshair.y
            },
            lastEndPoint: {
                x: image.width,
                y: crosshair.y
            },
            active: false,
            startPt: {},
            lastPt: {}
        };
        verticalLine = {
            firstStartPoint: {
                x: crosshair.x,
                y: 0
            },
            firstEndPoint: {
                x: crosshair.x,
                y: crosshair.y - radius
            },
            lastStartPoint: {
                x: crosshair.x,
                y: crosshair.y + radius
            },
            lastEndPoint: {
                x: crosshair.x,
                y: image.height
            },
            active: false,
            startPt: {},
            lastPt: {}
        };
        hThicknessLine = {
            firstStartPoint: {
                x: 0,
                y: crosshair.y + thicknessDistance
            },
            firstEndPoint: {
                x: image.width,
                y: crosshair.y + thicknessDistance
            },
            lastStartPoint: {
                x: 0,
                y: crosshair.y - thicknessDistance
            },
            lastEndPoint: {
                x: image.width,
                y: crosshair.y - thicknessDistance
            },
            controlPosTopLeft: {
                x: crosshair.x - controlRadius,
                y: crosshair.y
            },
            controlPosTopRight: {
                x: crosshair.x - controlRadius,
                y: crosshair.y
            },
            controlPosBtmLeft: {
                x: crosshair.x + controlRadius,
                y: crosshair.y
            },
            controlPosBtmRight: {
                x: crosshair.x + controlRadius,
                y: crosshair.y
            },
            active: false
        };
        vThicknessLine = {
            firstStartPoint: {
                x: crosshair.x + thicknessDistance,
                y: 0
            },
            firstEndPoint: {
                x: crosshair.x + thicknessDistance,
                y: image.height
            },
            lastStartPoint: {
                x: crosshair.x - thicknessDistance,
                y: 0
            },
            lastEndPoint: {
                x: crosshair.x - thicknessDistance,
                y: image.height
            },
            controlPosTopLeft: {
                x: crosshair.x,
                y: crosshair.y - controlRadius
            },
            controlPosBtmLeft: {
                x: crosshair.x,
                y: crosshair.y - controlRadius
            },
            controlPosTopRight: {
                x: crosshair.x,
                y: crosshair.y + controlRadius
            },
            controlPosBtmRight: {
                x: crosshair.x,
                y: crosshair.y + controlRadius
            },
            active: false
        };
        centerCircle = {
            centerPoint: {
                x: crosshair.x,
                y: crosshair.y
            },
            active: false,
            color: "yellow"
        };

        if (planeType === "axial") {
            horizontalLine.color = "lawngreen";
            verticalLine.color = "red";
            hThicknessLine.color = "lawngreen";
            vThicknessLine.color = "red";
        } else if (planeType === "coronal") {
            horizontalLine.color = "deepskyblue";
            verticalLine.color = "red";
            hThicknessLine.color = "deepskyblue";
            vThicknessLine.color = "red";
        } else if (planeType === "sagittal") {
            horizontalLine.color = "deepskyblue";
            verticalLine.color = "lawngreen";
            hThicknessLine.color = "deepskyblue";
            vThicknessLine.color = "lawngreen";
        }
        reActiveLines(element);
        cornerstone.updateImage(element);
    }

    function mouseMoveCallback(e, eventData) {
        cornerstoneTools.toolCoordinates.setCoords(eventData);
        let imageNeedsUpdate = false;
        let coords = eventData.currentPoints.canvas;

        let nearCenterCircle = pointNearcenterCircle(eventData.element, coords);
        let nearVerticalLine = pointNearVerticalLine(eventData.element, coords);
        let nearHorizontalLine = pointNearHorizontalLine(eventData.element, coords);
        let nearVThicknessLine = pointNearControlPoint(eventData.element, coords, vThicknessLine);
        let nearHThicknessLine = pointNearControlPoint(eventData.element, coords, hThicknessLine);

        if ((nearCenterCircle && !centerCircle.active) || (!nearCenterCircle && centerCircle.active)) {
            centerCircle.active = !centerCircle.active;
            imageNeedsUpdate = true;
        } else if ((nearVerticalLine && !verticalLine.active) || (!nearVerticalLine && verticalLine.active)) {
            // cannot rotate on editor mode
            if (!isEditing) {
                verticalLine.active = !verticalLine.active;
                imageNeedsUpdate = true;
            }
        } else if (
            (nearHorizontalLine && !horizontalLine.active) ||
            (!nearHorizontalLine && horizontalLine.active)
        ) {
            // cannot rotate on editor mode
            if (!isEditing) {
                horizontalLine.active = !horizontalLine.active;
                imageNeedsUpdate = true;
            }
        } else if (
            (nearVThicknessLine && !vThicknessLine.active) ||
            (!nearVThicknessLine && vThicknessLine.active)
        ) {
            vThicknessLine.active = !vThicknessLine.active;
            imageNeedsUpdate = true;
        } else if (
            (nearHThicknessLine && !hThicknessLine.active) ||
            (!nearHThicknessLine && hThicknessLine.active)
        ) {
            hThicknessLine.active = !hThicknessLine.active;
            imageNeedsUpdate = true;
        }

        if (vThicknessLine.active === true || hThicknessLine.active === true) {
            verticalLine.active = false;
            horizontalLine.active = false;
        }
        if (centerCircle.active === true) {
            verticalLine.active = false;
            horizontalLine.active = false;
            vThicknessLine.active = false;
            hThicknessLine.active = false;
        }


        if (imageNeedsUpdate === true) {
            cornerstone.updateImage(eventData.element);
        }
    }

    function mouseDownCallback(e, eventData) {
        let targetItem;
        if (centerCircle.active === true) {
            targetItem = "centerCircle";
        } else if (verticalLine.active === true) {
            targetItem = "verticalLine";
        } else if (horizontalLine.active === true) {
            targetItem = "horizontalLine";
        } else if (vThicknessLine.active === true) {
            targetItem = "vThicknessLine";
        } else if (hThicknessLine.active === true) {
            targetItem = "hThicknessLine";
        }

        if (targetItem) {
            $(eventData.element).off("CornerstoneToolsMouseMove", mouseMoveCallback);
            moveLines(e, targetItem);
            e.stopImmediatePropagation();
        }
        return false;
    }

    function updateSourceTargetLine(
        sourceLine,
        targetLine,
        sourceThicknessLine,
        targetThicknessLine,
        sEq,
        tEq,
        image
    ) {
        let radius = centerDiameter / 2;
        if (sourceLine && targetLine) {
            /*
                There are 4 points ( External 1 - Internal 1- Internal 2 -External 2 ) in this order
            */
            let sourceExternalPoints = keepLinePointInImage(image, sEq);
            let targetExternalPoints = keepLinePointInImage(image, tEq);
            let sourceInternalPoints = findIntersectionPointsCircleLine(
                centerCircle.centerPoint,
                radius,
                sEq,
                sourceExternalPoints[0]
            );
            let targetInternalPoints = findIntersectionPointsCircleLine(
                centerCircle.centerPoint,
                radius,
                tEq,
                targetExternalPoints[0]
            );

            // pt-crosshairs
            sourceLine.firstStartPoint.x = sourceExternalPoints[0].x; // External
            sourceLine.firstStartPoint.y = sourceExternalPoints[0].y;
            sourceLine.firstEndPoint.x = sourceInternalPoints[0].x; // Internal
            sourceLine.firstEndPoint.y = sourceInternalPoints[0].y;
            sourceLine.lastStartPoint.x = sourceInternalPoints[1].x; // Internal
            sourceLine.lastStartPoint.y = sourceInternalPoints[1].y;
            sourceLine.lastEndPoint.x = sourceExternalPoints[1].x; // External
            sourceLine.lastEndPoint.y = sourceExternalPoints[1].y;

            targetLine.firstStartPoint.x = targetExternalPoints[0].x;
            targetLine.firstStartPoint.y = targetExternalPoints[0].y;
            targetLine.firstEndPoint.x = targetInternalPoints[0].x;
            targetLine.firstEndPoint.y = targetInternalPoints[0].y;
            targetLine.lastStartPoint.x = targetInternalPoints[1].x;
            targetLine.lastStartPoint.y = targetInternalPoints[1].y;
            targetLine.lastEndPoint.x = targetExternalPoints[1].x;
            targetLine.lastEndPoint.y = targetExternalPoints[1].y;
        }
    }

    function parallelMoveLine(eventData, image) {
        if (!keepCenterPointInSp(eventData)) return;
        centerCircle.centerPoint.x += eventData.deltaPoints.image.x;
        centerCircle.centerPoint.y += eventData.deltaPoints.image.y;
        let indexVertical = centerCircle.centerPoint.x;
        let indexHorizontal = centerCircle.centerPoint.y;
        let centerPt = {x: indexVertical, y: indexHorizontal};

        // Find the line-equation that has horizontalLine's slope which new center point (i.e the updated crosshair coordinates)
        let hLineEqSlope = getTwoPointsEquation(horizontalLine.firstStartPoint, horizontalLine.lastEndPoint)
            .slope;
        let hLineEqYAxis = hLineEqSlope !== Infinity ? centerPt.y - hLineEqSlope * centerPt.x : centerPt.x;
        let hEq = {slope: hLineEqSlope, yAxis: hLineEqYAxis};

        // Find the line-equation that has verticalLine's slope which new center point (i.e the updated crosshair coordinates)
        let vLineEqSlope = getTwoPointsEquation(verticalLine.firstStartPoint, verticalLine.lastEndPoint).slope;
        let vLineEqYAxis = vLineEqSlope !== Infinity ? centerPt.y - vLineEqSlope * centerPt.x : centerPt.x;
        let vEq = {slope: vLineEqSlope, yAxis: vLineEqYAxis};
        updateSourceTargetLine(
            horizontalLine,
            verticalLine,
            hThicknessLine,
            vThicknessLine,
            hEq,
            vEq,
            image
        );
    }

    function getTwoPointsEquation(centerPt, currentPt) {
        /*
        @param(centerPt): {x:Number,y:Number} // image point
        @param(currentPt): {x:Number,y:Number} // image point
        return (Object):{slope:slope,yAxis:yAxis} //
        */
        let slope = 0;
        let yAxis;
        if (centerPt.x - currentPt.x) {
            //y = slope * x + yAxisVal
            slope = (centerPt.y - currentPt.y) / (centerPt.x - currentPt.x);
            yAxis = centerPt.y - slope * centerPt.x;
        } else {
            //the line vertical
            //x=yAxis
            slope = Infinity;
            yAxis = centerPt.x;
        }

        return {slope: +slope.toFixed(2), yAxis: yAxis};
    }

    function getRotatedPos(crosshair, point, degree) {
        // ref) https://www.jb51.net/article/175674.htm
        // crosshair : {x:0, y:0}
        // point: {x:0, y:0}
        let dx = crosshair.x;
        let dy = crosshair.y;
        let x = point.x;
        let y = point.y;
        let theta = (Math.PI / 180) * degree;
        let xx = (x - dx) * Math.cos(theta) - (y - dy) * Math.sin(theta) + dx;
        let yy = (x - dx) * Math.sin(theta) + (y - dy) * Math.cos(theta) + dy;
        return {x: xx, y: yy};
    }

    function moveLines(mouseEventData, lineType) {
        var element = mouseEventData.element;
        var enabledElement = cornerstone.getEnabledElement(element);
        var image = enabledElement.image;

        // getTwoPointsEquation(centerPt, currentPt);
        function mouseDragCallback(e, eventData) {
            let indexVertical = centerCircle.centerPoint.x;
            let indexHorizontal = centerCircle.centerPoint.y;
            let centerPt = {x: indexVertical, y: indexHorizontal};
            let currentPt = {
                x: eventData.currentPoints.image.x,
                y: eventData.currentPoints.image.y
            };

            let sourceEq = getTwoPointsEquation(centerPt, currentPt);

            // source line and target line are orthogonal
            let targetSlope, targetAxis;
            if (sourceEq.slope === 0) targetSlope = Infinity;
            else if (sourceEq.slope === Infinity) targetSlope = 0;
            else targetSlope = -1 / sourceEq.slope;

            if (targetSlope === Infinity) targetAxis = centerPt.x;
            else targetAxis = centerPt.y - targetSlope * centerPt.x;

            let targetEq = {slope: targetSlope, yAxis: targetAxis};
            if (lineType === "verticalLine") {
                verticalLine.active = true;

                // These startPt and lastPt keys are used for populating payload
                verticalLine.startPt = Object.assign({}, eventData.currentPoints.image);
                verticalLine.lastPt = Object.assign({}, eventData.currentPoints.image);

                updateSourceTargetLine(
                    verticalLine,
                    horizontalLine,
                    vThicknessLine,
                    hThicknessLine,
                    sourceEq,
                    targetEq,
                    image
                );
            } else if (lineType === "horizontalLine") {
                horizontalLine.active = true;
                // These startPt and lastPt keys are used for populating payload
                horizontalLine.startPt = Object.assign({}, eventData.currentPoints.image);
                horizontalLine.lastPt = Object.assign({}, eventData.currentPoints.image);

                updateSourceTargetLine(
                    horizontalLine,
                    verticalLine,
                    hThicknessLine,
                    vThicknessLine,
                    sourceEq,
                    targetEq,
                    image
                );
            } else if (lineType.includes("ThicknessLine")) {
                let l;
                let vEq = getTwoPointsEquation(verticalLine.firstStartPoint, verticalLine.firstEndPoint);
                let hEq = getTwoPointsEquation(horizontalLine.firstStartPoint, horizontalLine.firstEndPoint);

                let activeStandardLine = lineType === "hThicknessLine" ? horizontalLine : verticalLine;
                let activeStandardLineEq = lineType === "hThicknessLine" ? hEq : vEq;

                let k = activeStandardLineEq.slope;
                let c = activeStandardLineEq.yAxis;
                let t;
                if (k !== Infinity) t = currentPt.y - k * currentPt.x;

                let startImgPt = eventData.startPoints.image;

                if (k === Infinity) {
                    let symbol = startImgPt.x > activeStandardLine.firstStartPoint.x ? 1 : -1;
                    l = thicknessDistance + symbol * eventData.deltaPoints.image.x;
                } else {
                    l = Math.abs(t - c) / Math.sqrt(k * k + 1);
                }

                thicknessDistance = l < 1 ? 0 : Number(l.toFixed(2));

                /////////// check limitation not to over the line ///////////
                let startStandardY = startImgPt.x * k + c;
                let currentStandardY = currentPt.x * k + c;
                let isUpperStart = startStandardY > startImgPt.y;
                let isUpperCurrent = currentStandardY > currentPt.y;
                if (isUpperStart !== isUpperCurrent) {
                    thicknessDistance = 0;
                }
                /////////////////////////////////////////

                updateSourceTargetLine("", "", vThicknessLine, hThicknessLine, vEq, hEq, image);
            } else if (lineType === "centerCircle") {
                centerCircle.active = true;
                parallelMoveLine(eventData, image);
            }
            keepCenterPointInImage(image);
            cornerstone.updateImage(element);

            var eventType = "CornerstoneToolsMprOperatePositionModified";
            var modifiedEventData = {
                changeType: lineType,
                element: element,
                horizontalLine: horizontalLine,
                verticalLine: verticalLine,
                centerCircle: centerCircle,
                slope: targetSlope,
                thicknessDistance: thicknessDistance
            };
            $(element).trigger(eventType, modifiedEventData);

            return false;
        }

        $(element).on("CornerstoneToolsMouseDrag", mouseDragCallback);

        function mouseUpCallback(e, eventData) {
            horizontalLine.active = false;
            verticalLine.active = false;
            hThicknessLine.active = false;
            vThicknessLine.active = false;
            centerCircle.active = false;
            $(element).off("CornerstoneToolsMouseDrag", mouseDragCallback);
            $(element).off("CornerstoneToolsMouseUp", mouseUpCallback);
            let evt = {};
            $(element).on("CornerstoneToolsMouseMove", evt, mouseMoveCallback);
            cornerstone.updateImage(element);
        }

        $(element).on("CornerstoneToolsMouseUp", mouseUpCallback);
        return true;
    }

    function keepCenterPointInImage(image) {
        if (centerCircle.centerPoint.x < 0) centerCircle.centerPoint.x = 0;
        if (centerCircle.centerPoint.x > image.width - 1) centerCircle.centerPoint.x = image.width - 1;
        if (centerCircle.centerPoint.y < 0) centerCircle.centerPoint.y = 0;
        if (centerCircle.centerPoint.y > image.height - 1) centerCircle.centerPoint.y = image.height - 1;
    }

    function keepCenterPointInSp(eventData) {
        let x = eventData.currentPoints.image.x;
        let y = eventData.currentPoints.image.y;
        let storedPixels = cornerstone.getStoredPixels(eventData.element, x, y, 1, 1);
        let sp = storedPixels[0];
        if (sp === -30000 || sp === undefined) {
            console.warn("sp === -30000 || sp === undefined");
            return false;
        }
        return true;
    }

    function pointNearHorizontalLine(element, coords) {
        var leftLineSegment = {
            start: cornerstone.pixelToCanvas(element, horizontalLine.firstStartPoint),
            end: cornerstone.pixelToCanvas(element, horizontalLine.firstEndPoint)
        };
        var distanceToLeftLine = cornerstoneMath.lineSegment.distanceToPoint(leftLineSegment, coords);
        var rightLineSegment = {
            start: cornerstone.pixelToCanvas(element, horizontalLine.lastStartPoint),
            end: cornerstone.pixelToCanvas(element, horizontalLine.lastEndPoint)
        };
        var distanceToRightLine = cornerstoneMath.lineSegment.distanceToPoint(rightLineSegment, coords);
        return distanceToLeftLine < 3 || distanceToRightLine < 3;
    }

    function pointNearVerticalLine(element, coords) {
        var topLineSegment = {
            start: cornerstone.pixelToCanvas(element, verticalLine.firstStartPoint),
            end: cornerstone.pixelToCanvas(element, verticalLine.firstEndPoint)
        };
        var distanceToTopLine = cornerstoneMath.lineSegment.distanceToPoint(topLineSegment, coords);
        var bottomLineSegment = {
            start: cornerstone.pixelToCanvas(element, verticalLine.lastStartPoint),
            end: cornerstone.pixelToCanvas(element, verticalLine.lastEndPoint)
        };
        var distanceToBottomLine = cornerstoneMath.lineSegment.distanceToPoint(bottomLineSegment, coords);
        return distanceToTopLine < 3 || distanceToBottomLine < 3;
    }

    function pointNearcenterCircle(element, coords) {
        var startCanvas = cornerstone.pixelToCanvas(element, centerCircle.centerPoint);
        var bottomRightPoint = {
            x: centerCircle.centerPoint.x,
            y: centerCircle.centerPoint.y
        };
        var endCanvas = cornerstone.pixelToCanvas(element, bottomRightPoint);

        var rect = {
            left: Math.min(startCanvas.x, endCanvas.x),
            top: Math.min(startCanvas.y, endCanvas.y),
            width: Math.abs(startCanvas.x - endCanvas.x),
            height: Math.abs(startCanvas.y - endCanvas.y)
        };
        var distanceToPoint = cornerstoneMath.rect.distanceToPoint(rect, coords);

        var bInRect = false;
        if (
            coords.x > rect.left &&
            coords.x < rect.left + rect.width &&
            coords.y > rect.top &&
            coords.y < rect.top + rect.height
        )
            bInRect = true;

        return distanceToPoint < 20 || bInRect;
    }

    function pointNearControlPoint(element, coords, selectedLine) {
        //coords ==> current canvas point({x,y})
        // need to check for 8points
        let topLeftControl = cornerstone.pixelToCanvas(element, selectedLine.controlPosTopLeft);
        let topRightControl = cornerstone.pixelToCanvas(element, selectedLine.controlPosTopRight);
        let bmtLeftControl = cornerstone.pixelToCanvas(element, selectedLine.controlPosBtmLeft);
        let bmtRightControl = cornerstone.pixelToCanvas(element, selectedLine.controlPosBtmRight);
        let topleft = getTwoPointsDistance(coords, topLeftControl) < 3 ? "topleft" : "";
        let topright = getTwoPointsDistance(coords, topRightControl) < 3 ? "topright" : "";
        let btmleft = getTwoPointsDistance(coords, bmtLeftControl) < 3 ? "btmleft" : "";
        let btmright = getTwoPointsDistance(coords, bmtRightControl) < 3 ? "btmright" : "";
        return topleft || topright || btmleft || btmright;
    }

    ///////// BEGIN IMAGE RENDERING ///////
    function onImageRendered(e, eventData) {
        // we have tool data for this element - iterate over each one and draw it
        var context = eventData.canvasContext.canvas.getContext("2d");
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.save();
        drawMprHorizontalLine(context, eventData); // horizontal pt-crosshair
        drawMprVerticalLine(context, eventData); // vertical pt-crosshair
        drawMprCenter(context, eventData); // center pt-crosshair point
        context.restore();
    }

    function pointOnOutLine(point, width, height) {
        /*
        Check if the point is on the outline of the image
        point(Object): {x: Number(), y: Number()}
        width, height(Number)
        */
        return Boolean(point.x < 0 || point > width - 1 || point.y < 0 || point.y > height - 1);
    }

    function drawMprHorizontalLine(context, eventData) {
        let color = horizontalLine.color;
        let lineWidth = 1;
        if (horizontalLine.active === true) lineWidth = 2;
        let width = eventData.image.width;
        let height = eventData.image.height;

        let hFirstStartPt = horizontalLine.firstStartPoint;
        let hFirstEndPt = horizontalLine.firstEndPoint;
        let hLastStartPt = horizontalLine.lastStartPoint;
        let hLastEndPt = horizontalLine.lastEndPoint;

        let check1 = pointOnOutLine(hFirstStartPt, width, height);
        let check2 = pointOnOutLine(hFirstEndPt, width, height);
        let check3 = pointOnOutLine(hLastStartPt, width, height);
        let check4 = pointOnOutLine(hLastEndPt, width, height);

        // Get the handle positions in canvas coordinates
        let leftLineStartCanvas = cornerstone.pixelToCanvas(eventData.element, hFirstStartPt);
        let leftLineEndCanvas = cornerstone.pixelToCanvas(eventData.element, hFirstEndPt);

        let rightLineStartCanvas = cornerstone.pixelToCanvas(eventData.element, hLastStartPt);
        let rightLineEndCanvas = cornerstone.pixelToCanvas(eventData.element, hLastEndPt);

        if (!check1 || !check2) {
            // Draw the measurement line
            context.beginPath();
            context.strokeStyle = color;
            context.lineWidth = lineWidth;
            context.moveTo(leftLineStartCanvas.x, leftLineStartCanvas.y);
            context.lineTo(leftLineEndCanvas.x, leftLineEndCanvas.y);
            context.stroke();
        }

        if (!check3 || !check4) {
            // Draw the measurement line
            context.beginPath();
            context.strokeStyle = color;
            context.lineWidth = lineWidth;
            context.moveTo(rightLineStartCanvas.x, rightLineStartCanvas.y);
            context.lineTo(rightLineEndCanvas.x, rightLineEndCanvas.y);
            context.stroke();
        }
    }

    function drawMprVerticalLine(context, eventData) {
        var color = verticalLine.color;
        var lineWidth = 1;
        if (verticalLine.active === true) lineWidth = 2;
        let width = eventData.image.width;
        let height = eventData.image.height;

        let vFirstStartPt = verticalLine.firstStartPoint;
        let vFirstEndPt = verticalLine.firstEndPoint;
        let vLastStartPt = verticalLine.lastStartPoint;
        let vLastEndPt = verticalLine.lastEndPoint;

        let check1 = pointOnOutLine(vFirstStartPt, width, height);
        let check2 = pointOnOutLine(vFirstEndPt, width, height);
        let check3 = pointOnOutLine(vLastStartPt, width, height);
        let check4 = pointOnOutLine(vLastEndPt, width, height);

        // Get the handle positions in canvas coordinates
        var topLineStartCanvas = cornerstone.pixelToCanvas(eventData.element, vFirstStartPt);
        var topLineEndCanvas = cornerstone.pixelToCanvas(eventData.element, vFirstEndPt);

        var bottomLineStartCanvas = cornerstone.pixelToCanvas(eventData.element, vLastStartPt);
        var bottomLineEndCanvas = cornerstone.pixelToCanvas(eventData.element, vLastEndPt);

        if (!check1 || !check2) {
            // Draw the measurement line
            context.beginPath();
            context.strokeStyle = color;
            context.lineWidth = lineWidth;
            context.moveTo(topLineStartCanvas.x, topLineStartCanvas.y);
            context.lineTo(topLineEndCanvas.x, topLineEndCanvas.y);
            context.stroke();
        }
        if (!check3 || !check4) {
            // Draw the measurement line
            context.beginPath();
            context.strokeStyle = color;
            context.lineWidth = lineWidth;
            context.moveTo(bottomLineStartCanvas.x, bottomLineStartCanvas.y);
            context.lineTo(bottomLineEndCanvas.x, bottomLineEndCanvas.y);
            context.stroke();
        }
    }

    function drawMprCenter(context, eventData) {
        if (centerCircle.active) {
            let color = centerCircle.color;
            let lineWidth = 2; // centerCircle.active ? 2 : 1;
            let curPt = cornerstone.pixelToCanvas(eventData.element, centerCircle.centerPoint);
            drawEmptyCircle(
                context,
                curPt,
                color,
                lineWidth,
                (centerDiameter - isEditing * centerWidthParam) / 2
            );
        }
    }

    function updateMoveLine(itm, image, evt) {
        let indexVertical = centerCircle.centerPoint.x;
        let indexHorizontal = centerCircle.centerPoint.y;
        let centerPt = {x: indexVertical, y: indexHorizontal};

        let degree = itm.rotate_angle;

        //Fix) standard point as 1, when it item.crosshair x and y is on 0, will found same point on hStartRotatePos,vStartRotatePos
        let hStartRotatePos = getRotatedPos(itm.crosshair, {x: 1, y: itm.crosshair.y}, degree);
        let vStartRotatePos = getRotatedPos(itm.crosshair, {x: itm.crosshair.x, y: 1}, degree);

        let hEq = getTwoPointsEquation(itm.crosshair, hStartRotatePos);
        let vEq = getTwoPointsEquation(itm.crosshair, vStartRotatePos);
        let thickness = thicknessDistance;
        //todo - this condition is added, otherwise whenever force-updated thickness behaviour is like setting previous one
        if (evt !== "MPR_THICKNESS") {
            thickness = itm.axes_thickness / 2;
            thicknessDistance = thickness;
        }
        updateSourceTargetLine(
            horizontalLine,
            verticalLine,
            hThicknessLine,
            vThicknessLine,
            hEq,
            vEq,
            image
        );
    }

    function setLinePos(itm, eeImage, evt) {
        // 1. set centerCircle pos as crosshair
        centerCircle.centerPoint.x = itm.crosshair.x;
        centerCircle.centerPoint.y = itm.crosshair.y;
        // 2. find slope and yaxis
        updateMoveLine(itm, eeImage, evt);
        cornerstone.updateImage(this.element);
    }

    function reActiveLines(ele) {
        let list = ele? [ele]: [...MPR_ELEMENTS]
        list.forEach(element => {
            let eventData = {};
            $(element).off("CornerstoneImageRendered", onImageRendered);
            $(element).off("CornerstoneToolsMouseMove", mouseMoveCallback);
            $(element).off("CornerstoneToolsMouseDown", mouseDownCallback);
            $(element).on("CornerstoneImageRendered", onImageRendered);
            $(element).on("CornerstoneToolsMouseMove", eventData, mouseMoveCallback);
            $(element).on("CornerstoneToolsMouseDown", eventData, mouseDownCallback);
        })
    }

    function inactiveLines() {
        MPR_ELEMENTS.forEach(element => {
            $(element).off("CornerstoneImageRendered", onImageRendered);
            $(element).off("CornerstoneToolsMouseMove", mouseMoveCallback);
            $(element).off("CornerstoneToolsMouseDown", mouseDownCallback);
            cornerstone.updateImage(element);
        })
    }

    return {
        initMprLines: initMprLines,
        setLinePos: setLinePos,
        inactiveLines: inactiveLines
    };
}


function getMprOperateLine(element) {
    return mprOperateLines.find(item => item.element === element)
}

// 节流100sm内最多请求一次
let _throttlePosition = _.throttle(updatePosition, 100);

function mprPositionModified(evt, eventData) {
    _throttlePosition(eventData);
}


// firstStartPoint, firstEndPoint
// lastStartPoint, lastEndPoint
function getSlopeAngle(s1, s2) {
    return Math.atan((s2.y - s1.y) / (s2.x - s1.x)) * 180 / Math.PI;
}

// get other pane image and update line
function updatePosition(eventData) {
    // console.log("todo: updatePosition: update other pane image and line", eventData)
    let mprItem = {};
    let {changeType, element, thicknessDistance, centerCircle, horizontalLine} = eventData;

    console.log(`points: [${horizontalLine.firstStartPoint.x}, ${horizontalLine.firstStartPoint.y}], [${horizontalLine.firstEndPoint.x}, ${horizontalLine.firstEndPoint.y}], `);

    let angle = getSlopeAngle(horizontalLine.firstStartPoint, horizontalLine.firstEndPoint)
    mprItem["plane_name"] = element.id;
    mprItem["angle"] = angle;
    if (changeType.includes("ThicknessLine")) {
        let thickness = thicknessDistance;

        if (thickness === Infinity) {
            console.warn("Infinity thickness");
            return;
        }
        mprItem['thickness'] = thickness * 2
    } else {
        if (changeType === "centerCircle") {
            // mpr crosshair move
            let indexVertical = centerCircle.centerPoint.x;
            let indexHorizontal = centerCircle.centerPoint.y;
            mprItem["crosshair"] = [indexVertical, indexHorizontal]; //[x,y]
            //update crosshair of imageData
        } else {
            // mpr crosshair rotate
            let startPt = Object.assign({}, eventData[changeType].startPt); // currentPoints
            let lastPt = Object.assign({}, eventData[changeType].lastPt);
            mprItem["start_pt"] = [startPt.x, startPt.y]; // [x, y]
            mprItem["end_pt"] = [lastPt.x, lastPt.y]; // [x, y]
            mprItem["angle"] = angle
        }
    }
    mprItem["change_type"] = changeType;
    handleCenterLine(mprItem)
}

function handleCenterLine(mprItem) {
    const plane_type = TYPES.indexOf(mprItem.plane_name);
    if (plane_type === -1) {
        console.log("plane_name error... plane_name: ", mprItem.plane_name)
    }
    let url = '';
    if ("centerCircle" == mprItem.change_type){
        const x = mprItem.crosshair[0];
        const y = mprItem.crosshair[1];
        url = `/panmpr?uid=${uid}&plane_type=${plane_type}&x=${x}&y=${y}`;
    }
    else if ("verticalLine" == mprItem.change_type || "horizontalLine" == mprItem.change_type){
        start_pt = mprItem.start_pt;
        end_pt = mprItem.end_pt;
        angle = mprItem.angle
        console.log(`angle: ${angle}`)
        url = `/rotatech?uid=${uid}&plane_type=${plane_type}&angle=${angle}`;
    }
    // 1. request after position change
    fetch(apiPrefix + url, {credentials: "same-origin"})
        .then((response) => response.json())
        .then((buffer) => {
            const list = [0,1,2].filter(item => item !=plane_type);
            list.forEach(item => {
                updateMprImage(item);    
            })
        });
}

/**getEnable
 *
 * @param result is [imageObj] list.
 * imageObj {"plane_type": 1,"crosshair": {x: 300.0, y: 100.0},"rotate_angle": -45.0,"data": ""}
 * @param imageObj.plane_type pane type
 * @param imageObj.crosshair center point coords
 * @param imageObj.rotate_angle line angle
 * @param imageObj.data line image base64
 *
 *
 */
function handleCenterLineResult(result) {
    if (!Array.isArray(result)) {
        console.log("handleCenterLineResult return struct")
        return;
    }
    result.forEach(item => {
            // this.updateMprImage(item);
            // this.updateLine()
            // 2. async line
            setOperaterLinePosByElement(item, MPR_ELEMENTS[item.plane_type], 'centerCircle')
        // }
    });
}


$('#crosshairs').change(function(){
    let isCrosshairsChecked = $(this).is(':checked');
    if(isCrosshairsChecked){
        showCrosshairs();
    }else{
        mprOperateLines.forEach(line => {
            line.inactiveLines();
        })
        mprOperateLines = [];
    }
})