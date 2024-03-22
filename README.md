# Kaiu Text Diffusion 
The diffusion model generates chinese character. 

### Demo
See [Here](https://kthfan.github.io/text-generator).

### Usage

```bash
cd train/

# Train the model.
python train.py

# Convert weights to json format.
python pt2json.py
cd ..
cp train/results/model.json server/src/model.json

# Run the server.
cd server
python run.py 8888
```


:
