
const MAX_CANVAS_SIZE = 300;

// let imageContainerElem = document.getElementById('image-container');
let canvasElem = document.getElementById('canvas');
let gifDisplayerElem = document.getElementById('gif-displayer');
let refImgElem = document.getElementById('ref-img');
let trgImgElem = document.getElementById('trg-img');
let imgTableRowElem = document.getElementById('images-table-row')
let progressbarElem = document.getElementById('progressbar');
let stepsSlideElem = document.getElementById('steps-slide');
let heightSlideElem = document.getElementById('height-slide');
let widthSlideElem = document.getElementById('width-slide');
let noiseScaleSlideElem = document.getElementById('noise-scale-slide');
let lineWidthSlideElem = document.getElementById('line-width-slide');
let refGuideSlideElem = document.getElementById('ref-guide-slide');
let fontSizeSlideElem = document.getElementById('font-size-slide');
let gifDurationSlideElem = document.getElementById('gif-duration-slide');
let fpsSlideElem = document.getElementById('fps-slide');

let radioCpuElem = document.getElementById('radio-cpu');
let radioGpuElem = document.getElementById('radio-gpu');
let radioPenElem = document.getElementById('radio-pen');
let radioEraserElem = document.getElementById('radio-eraser');
let generateBnElem = document.getElementById('generate-bn');
let downloadBnElem = document.getElementById('download-bn');
let reloadModelBnElem = document.getElementById('reload-model-bn');
let resetRefImgBnElem = document.getElementById('reset-ref-img-bn');
let uploadRefImgFileElem = document.getElementById('upload-ref-img');

let stepsNumberElem = document.getElementById('steps-slide-display');
let heightNumberElem = document.getElementById('height-slide-display');
let widthNumberElem = document.getElementById('width-slide-display');
let noiseScaleNumberElem = document.getElementById('noise-scale-slide-display');
let lineWidthNumberElem = document.getElementById('line-width-slide-display');
let refGuideNumberElem = document.getElementById('ref-guide-slide-display');
let fontSizeNumberElem = document.getElementById('font-size-slide-display');
let gifDurationNumberElem = document.getElementById('gif-duration-slide-display');
let fpsNumberElem = document.getElementById('fps-slide-display');

let srcInputElem = document.getElementById('src-input');
let trgInputElem = document.getElementById('trg-input');

let uncondTabElem = document.getElementById('uncond-tab');
let img2imgTabElem = document.getElementById('img2img-tab');
let transitionTabElem = document.getElementById('transition-tab');
let tabElems = [uncondTabElem, img2imgTabElem, transitionTabElem];
let canvasElemList = Array.from(imgTableRowElem.children).map(e => e.children[0]);
let refImgDraw = new DrawableCanvas(refImgElem);
let trgImgDraw = new DrawableCanvas(trgImgElem);


function initializeDOM(){
    generateBnElem.disabled = true;
    adjustCanvasShape(canvasElemList, 48, 48);
    refImgDraw.enable();
    trgImgDraw.enable();
}

function advancedImageShow(image){
    image = tf.tensor(image);
    let height = image.shape[1];
    let width = image.shape[2];
    
    if (height < width){
        height = Math.round(height / width * MAX_CANVAS_SIZE);
        width = MAX_CANVAS_SIZE;
    } else {
        width = Math.round(width / height * MAX_CANVAS_SIZE);
        height = MAX_CANVAS_SIZE;
    }

    image = tf.image.resizeBilinear(image, [height, width]);
    if (globalObj.diffusionMode == 'transition'){
        image = image.squeeze(-1);
        imshowGif(gifDisplayerElem, image.neg().add(1), 1000 * gifDurationSlideElem.value / image.shape[0], 4, p => {
            progressbarElem.value = image.shape[0] + p * 5;
        });
    } else {
        image = image.squeeze(0).squeeze(-1);
        imshow(canvasElem, image.neg().add(1));
    }
    image.dispose();
}
function adjustCanvasShape(canvasElemList, height, width){
    if(!(canvasElemList instanceof Array)){
        canvasElemList = [canvasElemList];
    }
    
    if (height < width){
        height = height / width * MAX_CANVAS_SIZE;
        width = MAX_CANVAS_SIZE;
    } else {
        width = width / height * MAX_CANVAS_SIZE;
        height = MAX_CANVAS_SIZE;
    }
    for(let canvasElem of canvasElemList){    
        canvasElem.width = width;
        canvasElem.height = height;
        let yMargin = Math.max(0, (MAX_CANVAS_SIZE - height) / 2);
        let xMargin = Math.max(0, (MAX_CANVAS_SIZE - width) / 2);

        canvasElem.style.marginTop = yMargin.toString() + "px";
        canvasElem.style.marginBottom = yMargin.toString() + "px";
        canvasElem.style.marginLeft = xMargin.toString() + "px";
        canvasElem.style.marginRight = xMargin.toString() + "px";
        if (canvasElem.tagName.toUpperCase() === 'CANVAS'){
            ctx = canvasElem.getContext("2d");
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, width, height);
        }
    }
}

function bindEventListenerOfElements(elemList, name, funcList){
    if (funcList == undefined) funcList = [];
    else if (typeof(funcList) === 'function') funcList = [funcList];
    elemList.forEach(elem => {
        elem.addEventListener(name, evt => {
            elemList.forEach(elem1 => {
                elem1.value = elem.value;
            });
            funcList.forEach(func => {
                func(evt);
            });
        });
    });
}

generateBnElem.addEventListener('click', evt => {
    if (!globalObj.unetPrepared){
        alert("The model is not prepared.");
        return;
    }
    if (globalObj.isGenerating){
        return;
    }
    globalObj.isGenerating = true;
    progressbarElem.value = 0;

    if (globalObj.diffusionMode == "img2img"){
        progressbarElem.max = stepsSlideElem.value;

        tf.browser.fromPixelsAsync(refImgElem).then(img => {
            img = tf.tidy(() => {
                img = tf.image.rgbToGrayscale(img);
                img = img.cast('float32').div(255).neg().add(1);
                return img;
            });
            return img;
        }).then(imgArr => {
            globalObj.worker.postMessage({
                action: 'generate image',
                mode: globalObj.diffusionMode,
                steps: Number.parseInt(stepsSlideElem.value),
                height: Number.parseInt(heightSlideElem.value),
                width: Number.parseInt(widthSlideElem.value),
                refImg: imgArr.arraySync(),
                guideRatio: Number.parseFloat(refGuideNumberElem.value),
                noiseScale: Math.max(Number.parseFloat(noiseScaleSlideElem.value), 
                                     Number.parseFloat(refGuideNumberElem.value)),
            });
        });  
    } else if (globalObj.diffusionMode == "transition") {
        // set progress bar
        let nFrames = Math.ceil((fpsSlideElem.value * gifDurationSlideElem.value) / 2);
        progressbarElem.max = 2 * nFrames - 1 + 5;

        Promise.all([tf.browser.fromPixelsAsync(refImgElem), tf.browser.fromPixelsAsync(trgImgElem)])
        .then(imgs => {
            let [srcImg, trgImg] = imgs;
            srcImg = tf.tidy(() => {
                srcImg = tf.image.rgbToGrayscale(srcImg);
                srcImg = srcImg.cast('float32').div(255).neg().add(1);
                return srcImg;
            });
            trgImg = tf.tidy(() => {
                trgImg = tf.image.rgbToGrayscale(trgImg);
                trgImg = trgImg.cast('float32').div(255).neg().add(1);
                return trgImg;
            });
            return [srcImg, trgImg];
        }).then(imgsArr => {
            let [srcArr, trgArr] = imgsArr;
            globalObj.worker.postMessage({
                action: 'generate image',
                mode: globalObj.diffusionMode,
                steps: Number.parseInt(stepsSlideElem.value),
                frames: nFrames,
                height: Number.parseInt(heightSlideElem.value),
                width: Number.parseInt(widthSlideElem.value),
                srcImg: srcArr.arraySync(),
                trgImg: trgArr.arraySync(),
                noiseScale: Number.parseFloat(noiseScaleSlideElem.value),
            });
        });  
    } else {
        progressbarElem.max = stepsSlideElem.value;
        globalObj.worker.postMessage({
            action: 'generate image',
            mode: globalObj.diffusionMode,
            steps: Number.parseInt(stepsSlideElem.value),
            height: Number.parseInt(heightSlideElem.value),
            width: Number.parseInt(widthSlideElem.value),
            noiseScale: Number.parseFloat(noiseScaleSlideElem.value),
        });
    }
    
});

reloadModelBnElem.addEventListener('click', evt=>{
    globalObj.unetPrepared = false;
    generateBnElem.disabled = true;
    if ('kaiu-diffusion-model-weights' in localStorage){
        globalObj.worker.postMessage({
            action: 'set weights',
            data: localStorage.getItem('kaiu-diffusion-model-weights')
        });
    } else {
        globalObj.worker.postMessage({
            action: 'load weights',
            path: 'model.json'
        });
    }
});

downloadBnElem.addEventListener('click', evt => {
    let link = document.createElement("a");
    
    if(globalObj.diffusionMode === 'transition'){
        link.download = 'image.gif';
        link.href = gifDisplayerElem.src;
    } else {
        link.download = 'image.png';
        link.href = canvasElem.toDataURL('image/png');
    }
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

radioCpuElem.addEventListener('change', () => {
    if (radioCpuElem.checked){
        globalObj.worker.postMessage({
            action: 'set backend',
            backend: 'cpu'
        });
    }
});
radioGpuElem.addEventListener('change', () => {
    if (radioGpuElem.checked){
        globalObj.worker.postMessage({
            action: 'set backend',
            backend: 'webgl'
        });
    } 
});

radioPenElem.addEventListener('change', () => {
    if (radioPenElem.checked){
        refImgDraw.lineColor = 'black';
        trgImgDraw.lineColor = 'black';
    }
});
radioEraserElem.addEventListener('change', () => {
    if (radioEraserElem.checked){
        refImgDraw.lineColor = 'white';
        trgImgDraw.lineColor = 'white';
    } 
});


uncondTabElem.addEventListener('click', evt => {
    tabElems.forEach(elem => elem.classList.remove('tab-bn-active'));
    uncondTabElem.classList.add('tab-bn-active');
    let enable = [true, false, false, false];
    for(let i=0; i<imgTableRowElem.childElementCount; i++){
        imgTableRowElem.children[i].style.display = enable[i] ? '' : 'none';
    }
    
    globalObj.diffusionMode = 'uncond';
});
img2imgTabElem.addEventListener('click', evt => {
    tabElems.forEach(elem => elem.classList.remove('tab-bn-active'));
    img2imgTabElem.classList.add('tab-bn-active');
    let enable = [true, false, true, false];
    for(let i=0; i<imgTableRowElem.childElementCount; i++){
        imgTableRowElem.children[i].style.display = enable[i] ? '' : 'none';
    }
    globalObj.diffusionMode = 'img2img';
});
transitionTabElem.addEventListener('click', evt => {
    tabElems.forEach(elem => elem.classList.remove('tab-bn-active'));
    transitionTabElem.classList.add('tab-bn-active');
    let enable = [false, true, true, true];
    for(let i=0; i<imgTableRowElem.childElementCount; i++){
        imgTableRowElem.children[i].style.display = enable[i] ? '' : 'none';
    }
    globalObj.diffusionMode = 'transition';
});

resetRefImgBnElem.addEventListener('click', evt => {
    adjustCanvasShape(canvasElemList, Number.parseInt(heightSlideElem.value), Number.parseInt(widthSlideElem.value));
});

uploadRefImgFileElem.addEventListener('input', evt => {
    var img = new Image();
    img.onload = evt2 => {
        let ratio = Math.sqrt((heightNumberElem.value * widthNumberElem.value) / (img.height * img.width));
        let height = ratio * img.height;
        let width = ratio * img.width;
        if(height > heightNumberElem.max || width > widthNumberElem.max){
            if(height > width){
                height = heightNumberElem.max;
                width = heightNumberElem.max * width / height;
            } else{
                width = widthNumberElem.max;
                height = widthNumberElem.max * height / width;
            }
        }
        height = Math.max(Math.round(height / 8) * 8, 8);
        width = Math.max(Math.round(width / 8) * 8, 8);

        adjustCanvasShape(canvasElemList, height, width);
        heightNumberElem.value = height;
        heightSlideElem.value = height;
        widthNumberElem.value = width;
        widthSlideElem.value = width;

        ratio = Math.sqrt((height * width) / (img.height * img.width));
        img.height = ratio *　img.height;
        img.width = ratio *　img.width;
        refImgElem.getContext('2d').drawImage(img, 0, 0);
    };
    img.onerror = evt2 => {
        alert('Error: Fails to upload image.');
    };
    img.src = URL.createObjectURL(uploadRefImgFileElem.files[0]);
});

srcInputElem.addEventListener('input', evt => {
    adjustCanvasShape([refImgElem], refImgElem.height, refImgElem.width);
    let fontSize = fontSizeSlideElem.value / heightSlideElem.value * refImgElem.height;
    drawTextOnCanvas(refImgElem, srcInputElem.value, fontSize, refImgElem.height, refImgElem.width);
});
trgInputElem.addEventListener('input', evt => {
    adjustCanvasShape([trgImgElem], trgImgElem.height, trgImgElem.width);
    let fontSize = fontSizeSlideElem.value / heightSlideElem.value * trgImgElem.height;
    drawTextOnCanvas(trgImgElem, trgInputElem.value, fontSize, trgImgElem.height, trgImgElem.width);
});

bindEventListenerOfElements([stepsNumberElem, stepsSlideElem], 'input');
bindEventListenerOfElements([heightNumberElem, heightSlideElem], 'input', evt => {
    adjustCanvasShape(canvasElemList, Number.parseInt(heightSlideElem.value), Number.parseInt(widthSlideElem.value));
});
bindEventListenerOfElements([widthNumberElem, widthSlideElem], 'input', evt => {
    adjustCanvasShape(canvasElemList, Number.parseInt(heightSlideElem.value), Number.parseInt(widthSlideElem.value));
});
bindEventListenerOfElements([lineWidthNumberElem, lineWidthSlideElem], 'input', evt => {
    refImgDraw.lineWidth = lineWidthSlideElem.value;
    trgImgDraw.lineWidth = lineWidthSlideElem.value;
});
bindEventListenerOfElements([refGuideNumberElem, refGuideSlideElem], 'input');
bindEventListenerOfElements([fontSizeNumberElem, fontSizeSlideElem], 'input');
bindEventListenerOfElements([gifDurationNumberElem, gifDurationSlideElem], 'input');
bindEventListenerOfElements([fpsNumberElem, fpsSlideElem], 'input');
bindEventListenerOfElements([noiseScaleNumberElem, noiseScaleSlideElem], 'input');

initializeDOM();