

class DrawableCanvas{
    constructor(canvas){
        this.canvas = canvas;
        this.ctx = this.canvas.getContext("2d");
        this._lineColor = 'black';
        this._lineWidth = 15;
        this._currPosition = undefined;
        this._prevPosition = undefined;
        this._drawFlag = false;
        this._lambdaFuncs = [
            (evt => this._start(evt)), 
            (evt => this._moving(evt)), 
            (evt => this._end(evt))
        ];
    }

    set lineColor(val){
        this._lineColor = val;
    }
    get lineColor(){
        return this._lineColor;
    }
    set lineWidth(val){
        this._lineWidth = val;
    }
    get lineWidth(){
        return this._lineWidth;
    }

    enable(){
        this.canvas.addEventListener("mousedown", this._lambdaFuncs[0], false);
        this.canvas.addEventListener("mousemove", this._lambdaFuncs[1], false);
        this.canvas.addEventListener("mouseup", this._lambdaFuncs[1], false);
        document.documentElement.addEventListener("mouseup", this._lambdaFuncs[2], false);
    }

    disable(){
        this.canvas.removeEventListener("mousedown", this._lambdaFuncs[0], false);
        this.canvas.removeEventListener("mousemove", this._lambdaFuncs[1], false);
        this.canvas.removeEventListener("mouseup", this._lambdaFuncs[1], false);
        document.documentElement.removeEventListener("mouseup", this._lambdaFuncs[2], false);
    }

    _start(evt){
        this._drawFlag = true;
        this._currPosition = this._getCurrentPosition(evt);
    }

    _end(evt){
        this._drawFlag = false;
        this._prevPosition = undefined;
        this._currPosition = undefined;
    }

    _moving(evt){
        if(this._drawFlag){
            this._updatePosition(evt);
            this._draw(
                this._prevPosition[0], 
                this._prevPosition[1], 
                this._currPosition[0],
                this._currPosition[1]
            );
        }
    }

    _getCurrentPosition(evt){
        let dx = 0;
        let dy = 0;
        let elem = this.canvas.offsetParent;
        let marginLeft = Number.parseFloat(this.canvas.style.marginLeft);
        let marginTop = Number.parseFloat(this.canvas.style.marginTop);
        if (!Number.isNaN(marginLeft)){
            dx += marginLeft;
        }
        if (!Number.isNaN(marginTop)){
            dy += marginTop;
        }
        while(elem){
            dx += elem.offsetLeft - elem.scrollLeft;
            dy += elem.offsetTop - elem.scrollTop;
            elem = elem.offsetParent;
        }
        dx -= window.scrollX;
        dy -= window.scrollY;
        let x = evt.clientX - dx;
        let y = evt.clientY - dy;
        return [x, y];
    }

    _updatePosition(evt){
        this._prevPosition = this._currPosition;
        this._currPosition = this._getCurrentPosition(evt);
        if (this._prevPosition == undefined){ // check first update
            this._prevPosition = this._currPosition;
        }
    }

    _draw(x1, y1, x2, y2){
        this.ctx.beginPath();
        this.ctx.lineCap = "round";
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = this._lineColor;
        this.ctx.lineWidth = this._lineWidth;
        this.ctx.stroke();
        this.ctx.closePath();
    }
}