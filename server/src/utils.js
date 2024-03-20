
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

function drawTextOnCanvas(canvas, text, fontSize, height, width){
	const ctx = canvas.getContext("2d");
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
}

function imshow(canvas, tensor) {
	if(canvas === undefined){
    	canvas =  document.createElement("canvas");
		document.body.appendChild(canvas);
	}
    tensor = tf.clone(tensor);
    let min = tf.min(tensor, [0, 1], true);
    let max = tf.max(tensor, [0, 1], true);
    tensor = tensor.sub(min).div(max.sub(min));

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

function applyGaussianBlur(x, kernelSize, sigma){
	let ksizeHalf = (kernelSize - 1) * 0.5;
    let kernel = tf.linspace(-ksizeHalf, ksizeHalf, kernelSize);
	kernel = kernel.div(sigma).pow(2).mul(-0.5).exp();
    kernel = kernel.div(kernel.sum());
	kernel = tf.expandDims(kernel, 0).mul(tf.expandDims(kernel, 1));
	kernel = kernel.expandDims(-1).expandDims(-1);
	x = tf.mirrorPad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'symmetric');
	x = tf.depthwiseConv2d(x, kernel, 1, 'valid');
	return x;
}

function imshowGif(imgElem, tensor, delay=100, workers=4){
    tensor = tf.clone(tensor);
    let min = tf.min(tensor, [1, 2], true);
    let max = tf.max(tensor, [1, 2], true);
    tensor = tensor.sub(min).div(max.sub(min));

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
	let canvas = document.createElement('canvas');
	let ctx = canvas.getContext('2d');
	for(let i=0; i < tensor.shape[0]; i++){
		tf.browser.draw(tf.gather(tensor, i, 0), canvas);
		gif.addFrame(ctx, {copy: true, delay: delay});
	}
	gif.render();
}