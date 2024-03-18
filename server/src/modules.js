
class GroupNorm  extends tf.layers.Layer {
    constructor(config){
        super(config);
        this.numGroups = config.numGroups;
        this.eps = config.eps ?? 1e-5;
        this.affine = config.affine ?? true;
    }

    build(inputShape){
        let channels = inputShape[inputShape.length - 1];
        this.subchannels = channels / this.numGroups;
        if (this.affine){
            this.weight = this.addWeight('weight', [channels], 'float32', tf.initializers.ones());
            this.bias = this.addWeight('bias', [channels], 'float32', tf.initializers.zeros());
        } else {
            this.weight = tf.ones([channels]);
            this.bias = tf.zeros([channels]);
        }
    }

    call(x){
        let [batchSize, height, width, channels] = x.shape;
        x = tf.reshape(x , [batchSize, height, width, this.numGroups, this.subchannels]);
        let mean = tf.mean(x, [1, 2, 4], true);
        let variance = tf.mean(x.sub(mean).pow(2), [1, 2, 4], true);
        let std = variance.add(this.eps).sqrt()
        x = x.sub(mean).div(std);
        x = tf.reshape(x, [batchSize, height, width, channels]);
        x = x.mul(this.weight.read()).add(this.bias.read());
        return x;
    }

    static get className() {
        return 'GroupNorm';
    }
}

class SmartGroupNorm extends GroupNorm{
    constructor(config){
        super({numGroups: null, ...config});
    }

    build(inputShape){
        let channels = inputShape[inputShape.length - 1];
        this.numGroups = 1;
        while (this.numGroups < 32 && channels % (2 * this.numGroups) == 0){
            this.numGroups *= 2;
        }
        super.build(inputShape);
    }

    static get className() {
        return 'SmartGroupNorm';
    }
}

var Identity = () => Object({apply: (x, ...args) => x});
var Conv2d = config => tf.layers.conv2d({...{padding: "same", useBias: false, kernelInitializer: 'heNormal'}, ...config});
var Activation = config => tf.layers.activation({...{activation: 'swish'}, ...config});
var Linear = tf.layers.dense;
var Pooling = config => tf.layers.averagePooling2d({...{poolSize: [2, 2], strides: [2, 2]}, ...config});
var Upsampling = config => tf.layers.upSampling2d({...{size: [2, 2], interpolation: 'nearest'}, ...config});
var Normalization = config => new SmartGroupNorm(config);

class Layer extends tf.layers.Layer {
    /** Solving the problem that weights of custom layer won't be initialized. */
    constructor(config){
        super(config);
        this._attrsSetOrdered = [];
        let proxy =  new Proxy(this, {
            set: function (target, key, value) {
                target[key] = value;
                if(value instanceof tf.layers.Layer || value instanceof Array){
                    target._attrsSetOrdered.push(value);
                }
                return true;
            }
        }); 
        proxy._layers = [];
        this._attrsSetOrdered.pop();
        return proxy;
    }

    _checkAndSetLayer(value){
        if (value instanceof tf.layers.Layer){
            this._layers.push(value);
        } else if (value instanceof Array){
            for(let elem of value){
                this._checkAndSetLayer(elem);
            }
        }
    }

    _initializeWeights(){
        for(let layer of this._layers){
            if('weights' in layer){
                for(let weight of layer.weights){
                    this._addedWeightNames.push(weight.name);
                    if (weight.trainable)
                        this._trainableWeights.push(weight);
                    else
                        this._nonTrainableWeights.push(weight);
                }
            }
        }
    }

    apply(inputs, kwargs){
        let built = this.built;
        let result = super.apply(inputs, kwargs);
        if(!built){
            for(let attr of this._attrsSetOrdered)
                this._checkAndSetLayer(attr);
            this._initializeWeights();
        }
        return result;
    }
}

function timestepEmbedding(timesteps, dim, max_period=10000){
    let half = dim / 2;
    let freqs = tf.exp(tf.range(0, half).div(half).mul(-Math.log(max_period)));
    let args = timesteps.expandDims(1).mul(freqs.expandDims(0));
    let embedding = tf.concat([tf.cos(args), tf.sin(args)], -1);
    if(dim % 2)
        embedding = tf.concat([embedding, tf.zeros([embedding.shape[0], 1])], -1);
    return embedding;
}

class Sequential extends Layer{
    constructor(...layers){
        super({});
        this.layers = layers;
    }
    
    call(x){
        for(let layer of this.layers){
            x = layer.apply(x);
        }
        return x;
    }

    static get className() {
        return 'Sequential';
    }
}
class BasicResBlock extends Layer{
    constructor(config){
        super(config);
        this.out_channels = config.out_channels;
        this.emb_channels = config.emb_channels;
        this.is_up = config.is_up ?? false;
        this.is_down = config.is_down ?? false;
        this.dropout = config.dropout ?? 0;
    }

    build(inputShapes){
        let [x_shape, t_shape] = inputShapes;
        let in_channels = x_shape[x_shape.length - 1];
        this.shortcut_layers = new Sequential(
            this.is_up ? Upsampling() : this.is_down ? Pooling() : Identity(),
            this.out_channels != in_channels ? Conv2d({filters: this.out_channels, kernelSize: 1}) : Identity(),
        )
        
        this.in_residual_layers = new Sequential(
            Normalization(),
            Activation(),
            this.is_up ? Upsampling() : this.is_down ? Pooling() : Identity(),
            Conv2d({filters: this.out_channels, kernelSize: 3}),
        )
        
        this.out_residual_layers = new Sequential(
            Normalization(),
            Activation(),
            Conv2d({filters: this.out_channels, kernelSize: 3, kernelInitializer: 'zeros'}),
        )
        
        if(this.emb_channels){
            this.emb_layers = new Sequential(
                Activation(),
                Linear({units: this.out_channels}),
            )
        }  
    }

    call(inputs){
        let [x, t] = inputs;
        let x0 = this.shortcut_layers.apply(x);
        x = this.in_residual_layers.apply(x);
        if (this.emb_channels){
            t = this.emb_layers.apply(t);
            t = t.expandDims(1).expandDims(1);
            x = x.add(t);
        }
        x = this.out_residual_layers.apply(x);
        return x0.add(x);
    }

    static get className() {
        return 'BasicResBlock';
    }
}
let ResBlock = BasicResBlock;

function qkv_attention(qkv, n_heads=8){
    let [b, h, w, ch3] = qkv.shape;
    let ch = ch3 / 3;
    let scale = 1 / (ch / n_heads)**0.25;
    qkv = qkv.transpose([0, 3, 1, 2]);
    qkv = tf.reshape(qkv, [b * n_heads, ch3 / n_heads, h * w]);
    let [q, k, v] = tf.split(qkv, 3, 1);
    let m = q.mul(scale).transpose([0, 2, 1]).matMul(k.mul(scale));
    m = tf.softmax(m, -1);
    let a = v.matMul(m.transpose([0, 2, 1]));
    a = a.transpose([0, 2, 1]);
    a = tf.reshape(a, [b, h, w, ch]);
    return a;
}

class AttentionBlock extends Layer{
    constructor(config){
        super(config);
        this.num_heads = config.num_heads ?? 8;
    }

    build(inputShape){
        let channels = inputShape[inputShape.length - 1];
        this.norm = Normalization();
        this.qkv = Conv2d({filters: channels * 3, kernelSize: 1});
        this.proj_out = Conv2d({filters: channels, kernelSize: 1, kernelInitializer: 'zeros'});
    }   

    call(x){
        let qkv = this.qkv.apply(this.norm.apply(x));
        let h = qkv_attention(qkv, this.num_heads);
        h = this.proj_out.apply(h);
        return x.add(h);
    }

    static get className() {
        return 'AttentionBlock';
    }
}

class UNetBlock extends Layer{
    constructor(config){
        super(config);
        this.resblock = new ResBlock(config);
        this.attblock = !config.is_up && !config.is_down && config.use_att ? new AttentionBlock(config) : Identity();
    }

    build(inputShapes){}

    call(inputs){
        let [x, t] = inputs;
        return this.attblock.apply(this.resblock.apply([x, t]));
    }

    static get className() {
        return 'UNetBlock';
    }
}

class UNet extends Layer {
    constructor(config){
        super(config);
        this.base_channels = config.base_channels;
        this.out_channels = config.out_channels;
        this.channels_mult = config.channels_mult;
        this.num_blocks = config.num_blocks;
        this.use_attentions = config.use_attentions;
        this.emb_channels = 4 * this.base_channels;
    }

    build(inputShape){
        this.emb_layers = new Sequential(
            Linear({units: this.emb_channels}),
            Activation(),
            Linear({units: this.emb_channels}),
        );
        
        let _in_chs = [];
        this.in_layers = new Sequential(
            Conv2d({filters: this.base_channels, kernelSize: 3})
        );
        let _ch = this.base_channels;
        _in_chs.push(_ch);
        
        this.in_blocks = [];
        
        for(let i=0; i<this.channels_mult.length; i++){
            let [mul, nb, use_att] = [this.channels_mult[i], this.num_blocks[i], this.use_attentions[i]];
            for(let j=0; j<nb; j++){
                this.in_blocks.push(
                    new UNetBlock({out_channels: this.base_channels * mul, emb_channels: this.emb_channels,
                               is_down: i != this.num_blocks.length - 1 && j == nb - 1,
                               use_att: use_att})
                );
                _ch = this.base_channels * mul;
                _in_chs.push(_ch);
            }
        }

        this.mid_blocks = [
            new UNetBlock({out_channels: _ch, emb_channels: this.emb_channels, use_att: this.use_attentions[-1]}),
            new ResBlock({out_channels: _ch, emb_channels: this.emb_channels})
        ];
        
        this.out_blocks = [];
        for(let i=this.channels_mult.length - 1; i >= 0; i--){
            let [mul, nb, use_att] = [this.channels_mult[i], this.num_blocks[i], this.use_attentions[i]];
            for(let j=0; j < nb; j++){
                this.out_blocks.push(
                    new UNetBlock({out_channels: this.base_channels * mul, emb_channels: this.emb_channels, 
                               is_up: i != this.channels_mult.length - 1 && j == 0,
                               use_att: use_att})
                );
                _ch = this.base_channels * mul;
            }
        }

        this.out_layers = new Sequential(
            Normalization(),
            Activation(),
            Conv2d({filters: this.out_channels, kernelSize: 3, kernelInitializer: 'zeros'})
        );
    }

    call(inputs){
        let [x, t] = inputs;
        let t_emb = timestepEmbedding(t, this.base_channels);
        t_emb = this.emb_layers.apply(t_emb);
        let latents = [];
        
        x = this.in_layers.apply(x);
        latents.push(x);
        
        for(let block of this.in_blocks){
            x = block.apply([x, t_emb]);
            latents.push(x);
        }
        
        for(let block of this.mid_blocks){
            x = block.apply([x, t_emb]);
        }

        for(let block of this.out_blocks){
            let h = latents.pop();
            x = tf.concat([x, h], -1);
            x = block.apply([x, t_emb]);
        }
        
        x = tf.concat([x, latents.pop()], -1);
        x = this.out_layers.apply(x);
        
        return x
    }

    static get className() {
        return 'UNet';
    }
}
        

