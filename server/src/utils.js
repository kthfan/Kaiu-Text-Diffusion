
function loadJS(src){
	let script = document.createElement("script");
	script.src = src;
	script.async = true;
	document.body.appendChild(script);
	let [ok, err] = [null, null]; 
	let promise = new Promise((resolve, reject) => {
		ok = resolve;
		err = reject;
	});
	script.onload = () => {
		ok();
	}
	return promise;
}

function createUNet(config){
	let unet = new UNet(config);
	let x = tf.randomNormal([1, 48, 48, 1]);
	let t = tf.tensor([999]);
	let eps = unet.apply([x, t]);
	return unet;
}

function drawTextOnCanvas(canvas, text, fontSize, height, width, antialias=true){
	const S = 2;
	let ctx, canvas2;
	if(antialias){ // draw on a larger canvas
		fontSize *= S;
		height *= S;
		width *= S;

		canvas2 = document.createElement('canvas');
		canvas2.width = width;
		canvas2.height = height;
		ctx = canvas2.getContext("2d", {alpha:true, colorSpace: 'srgb'});
	} else {
		ctx = canvas.getContext("2d");
	}
	let nCharsPerRow = Math.floor(width / fontSize);
	let nLines = Math.ceil(text.length / nCharsPerRow);
	let nCharsPerRowRemain = text.length % nCharsPerRow;
	let marginLeft = (width - nCharsPerRow * fontSize) / 2;
	let marginTop = (height - nLines * fontSize) / 2;
	let marginLeftRemain = (width - nCharsPerRowRemain * fontSize) / 2;
	ctx.font = `${fontSize}px 標楷體`;
	ctx.fillStyle = 'black';
	ctx.textAlign = "left";
	ctx.textBaseline = "top";

	for(let i=0; i < text.length; i++){
		let r = Math.floor(i / nCharsPerRow);
		let c = i - r * nCharsPerRow;
		if (nCharsPerRowRemain !== 0 && r === nLines - 1){
			ctx.fillText(text[i], marginLeftRemain + c * fontSize, marginTop + r * fontSize);
		} else {
			ctx.fillText(text[i], marginLeft + c * fontSize, marginTop + r * fontSize);
		}
	}
	if(antialias){ // past the image back to our canvas
		let ctx0 = canvas.getContext("2d", {alpha:true, colorSpace: 'srgb'});
		let img = document.createElement('img');
		img.style.display = 'none';
		img.src = canvas2.toDataURL('image/png');
		img.onload = () => {
			document.body.appendChild(img);
			ctx0.drawImage(img, 0, 0, width, height, 0, 0, width / S, height / S);
			document.body.removeChild(img);
		}
	}
}

function imshow(canvas, tensor) {
	if(canvas === undefined){
    	canvas =  document.createElement("canvas");
		document.body.appendChild(canvas);
	}
    tensor = tf.clone(tensor);

    canvas.width = tensor.shape[0];
    canvas.height = tensor.shape[1];
	tf.browser.draw(tensor, canvas);
}

function computeVar(x, axis, keepDims){
	let Ex = tf.mean(x, axis, keepDims);
	let Ex2 = tf.mean(x.pow(2), axis, keepDims);
	let V = Ex2.sub(Ex.pow(2));
	return V;
}

function applyGaussianBlur(images, kernelSize, sigma){
	if (!(kernelSize instanceof Array)){
		kernelSize = [kernelSize, kernelSize];
	}
	if (!(sigma instanceof Array)){
		sigma = [sigma, sigma];
	}
	let [sigmaY, sigmaX] = sigma;
	let kernelY, kernelX, kernel;
	let ksizeHalf = [Math.floor(kernelSize[0] / 2), Math.floor(kernelSize[1] / 2)];
	if(sigmaY > 0 && kernelSize[0] > 1){
		kernelY = tf.linspace(-ksizeHalf[0], ksizeHalf[0], kernelSize[0]);
		kernelY = kernelY.div(sigmaY).pow(2).mul(-0.5).exp();
		kernelY = kernelY.div(kernelY.sum());
	} else{
		kernelY = tf.ones([1]);
	}
	
	if(sigmaX > 0 && kernelSize[1] > 1){
		kernelX = tf.linspace(-ksizeHalf[1], ksizeHalf[1], kernelSize[1]);
		kernelX = kernelX.div(sigmaX).pow(2).mul(-0.5).exp();
		kernelX = kernelX.div(kernelX.sum());
	}else{
		kernelX = tf.ones([1]);
	}

	kernel = tf.expandDims(kernelX, 0).mul(tf.expandDims(kernelY, 1));
	kernel = kernel.expandDims(-1).expandDims(-1);
	if(images.shape.length === 3){
		images = tf.mirrorPad(images, [[ksizeHalf[0], ksizeHalf[0]], [ksizeHalf[1], ksizeHalf[1]], [0, 0]], 'symmetric');
		images = tf.depthwiseConv2d(images.expandDims(0), kernel, 1, 'valid').squeeze(0);
	} else if(images.shape.length === 4){
		images = tf.mirrorPad(images, [[0, 0], [ksizeHalf[0], ksizeHalf[0]], [ksizeHalf[1], ksizeHalf[1]], [0, 0]], 'symmetric');
		images = tf.depthwiseConv2d(images, kernel, 1, 'valid');
	} else throw Error('Wrong dim.');
	return images;
}

function resizeBilinear(images, size, alignCorners=false, halfPixelCenters=false, antialias=false){
	if(antialias){
		let h0, w0, h1, w1;
		if(images.shape.length === 3){
			h0 = images.shape[0];
			w0 = images.shape[1];
		} else if(images.shape.length === 4){
			h0 = images.shape[1];
			w0 = images.shape[2];
		} else throw Error('Wrong dim.');
		[h1, w1] = size;
		
		const S = 0.5;
		let sigmaY = S * (h0 / h1);
		let sigmaX = S * (w0 / w1);

		if(sigmaY > S || sigmaX > S){
			let kernelSize = [Math.ceil(2 * sigmaY), Math.ceil(2 * sigmaX)];
			if(sigmaY <= S){
				kernelSize[0] = 0;
			}
			if(sigmaX <= S){
				kernelSize[1] = 0;
			}
			images = applyGaussianBlur(images, kernelSize, [sigmaY, sigmaX]);
		}
	}
	return tf.image.resizeBilinear(images, size, alignCorners, halfPixelCenters);
}

function imshowGif(imgElem, tensor, delay=100, workers=4, callback=null){
    tensor = tf.clone(tensor);

    imgElem.width = tensor.shape[2];
    imgElem.height = tensor.shape[1];

	let gif = new GIF({
		quality: 10,
		workers: workers,
		// workerScript: 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js',
		width: tensor.shape[2],
		height: tensor.shape[1],
	});
	gif.on('finished', blob => {
		imgElem.src = URL.createObjectURL(blob);
	});
	if (callback){
		gif.on('progress', callback);
	}
	let canvas = document.createElement('canvas');
	let ctx = canvas.getContext('2d', {willReadFrequently: true});
	for(let i=0; i < tensor.shape[0]; i++){
		tf.browser.draw(tf.gather(tensor, i, 0), canvas);
		gif.addFrame(ctx, {copy: true, delay: delay});
	}
	gif.render();
}