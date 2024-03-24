


function tToSigma(t, numTimesteps=1000, betaD=14.617, betaMin=0.0015){
	t = t.div(numTimesteps - 1);
	let sigma = tf.sqrt(tf.exp(t.pow(2).mul(betaD / 2).add(t.mul(betaMin))).sub(1));
	return sigma;
}
function sigmaToT(sigma, numTimesteps=1000, betaD=14.617, betaMin=0.0015){
	let t = sigma.pow(2).add(1).log().mul(2 * betaD).add(betaMin**2).sqrt().add(-betaMin).div(betaD);
    t = t.mul(numTimesteps - 1);
	return t;
}

function ddpmDenoise(model, x, t, sigma){
	let cOut = - sigma;
	let cIn = 1 / (sigma ** 2 + 1. ** 2) ** 0.5;
	let sIn = tf.ones([x.shape[0]]);
	let eps = model.apply([x.mul(cIn), sIn.mul(t)]);
	let denoised = x.add(eps.mul(cOut));
	return denoised;
}

function getAncestralStep(sigmaFrom, sigmaTo, eta=1.){
	if (!eta)
		return [sigmaTo, 0.];
	let sigmaUp = Math.min(sigmaTo, eta * (sigmaTo ** 2 * (sigmaFrom ** 2 - sigmaTo ** 2) / sigmaFrom ** 2) ** 0.5)
	let sigmaDown = (sigmaTo ** 2 - sigmaUp ** 2) ** 0.5
	return [sigmaDown, sigmaUp];
}

function sampleEulerAncestral(model, x, callback, steps=20, eta=1., numTimesteps=1000, betaD=14.617, betaMin=0.0015){
	let sigmaMax = tToSigma(tf.tensor(numTimesteps - 1), numTimesteps, betaD, betaMin).arraySync();
	return sampleEulerAncestralManual(model, x.mul(sigmaMax), sigmaMax, callback, steps, eta, numTimesteps, betaD, betaMin);
}

function sampleEulerAncestralManual(model, x, sigmaMax, callback, steps=20, eta=1., num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	if(callback === undefined)
		callback = () => {};
	let tMax = sigmaToT(tf.tensor(sigmaMax), num_timesteps, beta_d, beta_min);
	tMax = tMax.arraySync();
    steps = Math.ceil(steps / (num_timesteps - 1) * tMax);
	if (steps <= 0 || tMax <= 0) {
		return x;
	}
	
    let tList = tf.linspace(tMax, 1, steps);
	let sigmas = tToSigma(tList, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	tList = tList.arraySync();
	for(let i=0; i < steps; i++){
		callback(i);
		x = tf.tidy(() => {
			let t = tList[i];
			let sigma = sigmas[i];
			let sigma_next = sigmas[i + 1];
			
			let denoised = ddpmDenoise(model, x, t, sigma);
			let [sigma_down, sigma_up] = getAncestralStep(sigma, sigma_next, eta);
			let d =  tf.sub(x, denoised).div(sigma);
			let dt = sigma_down - sigma;
			x = x.add(d.mul(dt));
			if(sigma_next > 0)
				x = x.add(tf.randomNormal(x.shape).mul(sigma_up));
			return x;
		});
	}
	return x	
}

function lowPassFilter(x, D){
	let h = x.shape[1];
	let w = x.shape[2];
	x = resizeBilinear(x, [Number.parseInt(h / D), Number.parseInt(w / D)], false, false, true);
	x = resizeBilinear(x, [h, w]);
	return x;
}
function sampleEulerAncestralWithILVR(model, x, ref, callback, steps=20, guideRatio=0.5, eta=1., num_timesteps=1000, 
									  beta_d=14.617, beta_min=0.0015, D=2){
	let t_list = tf.linspace(num_timesteps - 1, 1, steps);
	let sigmas = tToSigma(t_list, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas, tf.zeros([1])], 0);
	sigmas = sigmas.arraySync();
	t_list = t_list.arraySync();

	x = x.mul(sigmas[0]);
	for(let i=0; i < steps; i++){
		callback(i);
		x = tf.tidy(() => {
			let t = t_list[i];
			let sigma = sigmas[i];
			let sigma_next = sigmas[i + 1];
			
			let denoised = ddpmDenoise(model, x, t, sigma);
			let [sigma_down, sigma_up] = getAncestralStep(sigma, sigma_next, eta);
			
			let d =  tf.sub(x, denoised).div(sigma);
			let dt = sigma_down - sigma;
			x = x.add(d.mul(dt));
			if(sigma_next > 0){
				x = x.add(tf.randomNormal(x.shape).mul((1 - guideRatio**2)**0.5 * sigma_up));

				// ILVER
				let ref_t = ref.add(tf.randomNormal(x.shape).mul(sigma_next));
				let guide = lowPassFilter(ref_t, D).sub(lowPassFilter(x, D));
				guide = guide.div(computeVar(guide).add(1e-7).sqrt()).mul(guideRatio * sigma_up);
				x = x.add(guide);
			}
			
			return x;
		});
	}
	return x
}


function getSamplePairSigma(n=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	let sigmaMax = tToSigma(tf.tensor(num_timesteps - 1), num_timesteps, beta_d, beta_min).arraySync();
    // let sigmas = tf.linspace(0, sigmaMax, n);
	
	let tList = tf.linspace(5e-3, 1, n);
	let sigmas = tf.sqrt(tf.exp(tList.mul(Math.log(sigmaMax**2 + 1))).sub(1));
	// let sigmas = tToSigma(tf.linspace(1, num_timesteps - 1, n - 1), num_timesteps, beta_d, beta_min);
	// sigmas = tf.concat([tf.zeros([1]), sigmas], 0);
	return sigmas;
}

function qSamplePair(x1, x2, n=20, num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
    let sigmaMax = tToSigma(tf.tensor(num_timesteps - 1), num_timesteps, beta_d, beta_min).arraySync();
	let sigmas = getSamplePairSigma(n, num_timesteps, beta_d, beta_min);
	sigmas = sigmas.arraySync();
    let noise = tf.randomNormal(x1.shape);
	let x1List = [];
	let x2List = [];
    for(let i=0; i < n - 1; i++){
        let sigma = sigmas[i];
        let alpha = 0.5 * sigma / sigmaMax;
        let _x1 = x1.mul(1 - alpha).add(x2.mul(alpha)).add(noise.mul(sigma));
        let _x2 = x1.mul(alpha).add(x2.mul(1 - alpha)).add(noise.mul(sigma));
        x1List.push(_x1);
        x2List.push(_x2);
	}
	let _xn = x1.mul(0.5).add(x2.mul(0.5)).add(noise.mul(sigmas[n - 1]));

	x1List.push(_xn);
    x2List.reverse();
    return x1List.concat(x2List);
}

function pSamplePair(model, xList, callback, steps=20, n=20, eta=1., num_timesteps=1000, beta_d=14.617, beta_min=0.0015){
	let yList = [];
	let sigmas = getSamplePairSigma(n, num_timesteps, beta_d, beta_min);
	sigmas = tf.concat([sigmas.slice([0], [n - 1]), sigmas.reverse(0)], 0);
	sigmas = sigmas.arraySync();
	
    for (let i=0; i < xList.length; i++){
		callback(i);
		let y = sampleEulerAncestralManual(model, xList[i], sigmas[i], undefined, steps, eta, num_timesteps, beta_d, beta_min);
		yList.push(y);
	}
	
	return yList;
}
