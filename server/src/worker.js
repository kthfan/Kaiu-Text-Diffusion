
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js");
// importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tfjs-vis.umd.min.js");
importScripts('utils.js');
importScripts('modules.js');
importScripts('sampling.js');

let globalObj = {};

function generateImageUncond(height, width, steps, callback){
    let genImg = tf.tidy(() => {
        let genImg = sampleEulerAncestral(globalObj.unet, 
                        tf.randomNormal([1, height, width, 1]),
                        callback,
                        steps);
        genImg = genImg.squeeze(0).squeeze(-1);
        genImg = genImg.mul(0.1850).add(0.0438);
        genImg = tf.clipByValue(genImg, 0, 1);
        return genImg;
    });
    return genImg;
}

function generateImageImg2Img(refImg, height, width, steps, guideRatio, callback){
    refImg = tf.tensor(refImg).expandDims(0);
    refImg = applyGaussianBlur(refImg, 5, 2);
    refImg = tf.image.resizeBilinear(refImg, [height, width]);
    refImg = refImg.sub(0.0438).div(0.1850);

    let genImg = tf.tidy(() => {
        let genImg = sampleEulerAncestralWithILVR(globalObj.unet, 
                        tf.randomNormal([1, height, width, 1]),
                        refImg,
                        callback,
                        steps,
                        guideRatio);
        // let genImg = lowPassFilter(refImg, 4);
        // let genImg = refImg;
        genImg = genImg.squeeze(0).squeeze(-1);
        genImg = genImg.mul(0.1850).add(0.0438);
        genImg = tf.clipByValue(genImg, 0, 1);
        return genImg;
    });
    return genImg;
}

onmessage = function(evt) {
	// tf.engine().startScope();
	if (evt.data.action === "load weights"){
        fetch(evt.data.path, {mode: 'no-cors'}).then(r => r.text()).then(jsonstr => {
            postMessage({action: evt.data.action, data: jsonstr, message: "The weights is loaded.", status: 0});
        }).catch(err => {
            postMessage({action: evt.data.action, message: "Fails to load the weights. The error message is: " + String(err), status: 1});
        });
    } else if (evt.data.action === "set weights"){
        let minConfig = {
			base_channels: 32,
			out_channels: 1,
			channels_mult: [1, 1, 2],
			num_blocks: [2, 2, 2],
			use_attentions: [false, false, false, false]
		};
        globalObj.unet = createUNet(minConfig);

        let json = JSON.parse(evt.data.data);
        let weights = json.map(e => tf.tensor(e[1], e[0]));
        globalObj.unet.setWeights(weights);
		postMessage({action: evt.data.action, message: "The weights is set.", status: 0});

    } else if (evt.data.action === "set backend"){
        tf.setBackend(evt.data.backend);
        postMessage({action: evt.data.action, 
                     message: `The current backend is ${tf.getBackend()}.`, 
                     status: Number.parseInt(evt.data.backend != tf.getBackend())});
    } else if (evt.data.action === "generate image"){
        let callback = i => {
            postMessage({action: 'progress', 
                         status: 0,
                         currentStep: i});
        }
        let genImg;
        if (evt.data.mode === 'img2img'){
            genImg = generateImageImg2Img(evt.data.refImg, evt.data.height, evt.data.width, evt.data.steps, 
                                          evt.data.guideRatio, callback);
        } else {
            genImg = generateImageUncond(evt.data.height, evt.data.width, evt.data.steps, callback);
        }
        
        postMessage({action: evt.data.action, 
                     mode: evt.data.mode,
                     message: `The image with ${genImg.shape[0]}X${genImg.shape[1]} pixels is generated.`, 
                     status: 0,
                     image: genImg.arraySync()});
        genImg.dispose();
    }

	// tf.engine().endScope();
}

