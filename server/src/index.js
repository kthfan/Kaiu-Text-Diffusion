

if(document.readyState === "complete" || document.readyState === "interactive") {
	setTimeout(main, 1);
}else{
    document.addEventListener("DOMContentLoaded", main);
}

let globalObj = {diffusionMode: 'uncond'};


async function main(){
	Promise.all([
		loadJS("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js")
	])
	.then(() => loadJS('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js'))
	.then(() => loadJS("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tfjs-vis.umd.min.js"))
	.then(() => loadJS("modules.js"))
	.then(() => loadJS("sampling.js"))
	.then(() => {
		// try{
		// 	tf.setBackend('wasm');
		// } catch(err){
		// 	tf.setBackend('cpu');
		// }
		// tf.setBackend('wasm');
		tf.setBackend('cpu');
		// tf.setBackend('webgl');
		
		globalObj.unetPrepared = false;
		globalObj.isGenerating = false;
		globalObj.worker = new Worker("worker.js");
		reloadModelBnElem.click();

		globalObj.worker.onmessage = evt => {
			if (evt.data.action === "load weights"){
				if (evt.data.status === 0){
					try{
						localStorage.setItem('kaiu-diffusion-model-weights', evt.data.data);
					} catch (err){
						console.warn(err);
					}
					globalObj.worker.postMessage({
						action: 'set weights',
						data: evt.data.data
					});
				} else alert(evt.data.message);
			} else if(evt.data.action === "set weights"){
				if (evt.data.status === 0){
					globalObj.worker.postMessage({
						action: 'set backend',
						backend: radioCpuElem.checked ? 'wasm' : 'webgl'
					});
					globalObj.unetPrepared = true;
					generateBnElem.disabled = false;
				} else alert(evt.data.message);
			} else if (evt.data.action === "set backend") {
				if (evt.data.status !== 0){
					if (evt.data.backend === "wasm"){ // if wasm is invalid, back to cpu
						globalObj.worker.postMessage({
							action: 'set backend',
							backend: radioCpuElem.checked ? 'cpu' : 'webgl'
						});
					} else {
						alert(evt.data.message);
					}
				}
			} else if (evt.data.action === "generate image"){
				globalObj.isGenerating = false;
				if (evt.data.status === 0){
					advancedImageShow(evt.data.image);
				} else alert(evt.data.message);
			} else if (evt.data.action === "progress"){
				progressbarElem.value = (evt.data.currentStep + 1);
			}
		}
	});
}