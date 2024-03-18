
function load_js(src){
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

function compute_var(x, axis, keepDims){
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