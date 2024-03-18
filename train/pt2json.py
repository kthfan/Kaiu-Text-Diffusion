
import argparse
import torch
import numpy as np

def weight_pt2tf(weight):
    weight = weight.detach().cpu().numpy()
    if len(weight.shape) == 4:
        weight = np.transpose(weight, (2, 3, 1, 0))
    elif len(weight.shape) == 2:
        weight = np.transpose(weight, (1, 0))
    return weight

def to_weight_json_format(params):
    shapes = [[str(n) for n in param.shape] for param in params]
    shapes = [','.join(shape) for shape in shapes]
    shapes = [f'[{shape}]' for shape in shapes]
    params = [param.ravel().astype(np.float16).round(4).astype(str).tolist() for param in params]
    params = [','.join(param) for param in params]
    params = [f'[{param}]' for param in params]
    params = [f'[{shape},{param}]' for shape, param in zip(shapes, params)]
    params = ','.join(params)
    params = f'[{params}]'
    return params

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pt-path', type=str, default='results/model.pt')
    parser.add_argument('--json-path', type=str, default='results/model.json')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    params = torch.load(args.pt_path)
    params = list(params.values())
    # shapes = [param.shape for param in params]
    params = [weight_pt2tf(param) for param in params]
    json_txt = to_weight_json_format(params)
    with open(args.json_path, 'w') as file:
        file.write(json_txt)
    

if __name__ == "__main__":
    main()
