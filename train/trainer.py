
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.cuda import amp
from torchmetrics import MeanMetric
from utils import ModelEMA

class DDPMTrainer:
    def __init__(self, model, optimizer, criterion, lr_scheduler=None, use_cuda=None, use_amp=True,
                 num_timesteps=1000, use_ema=True):
        ### set models ###
        self.model = model
        self.use_ema = use_ema
        
        ##################

        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self._initialize_cuda(use_cuda, use_amp)
        self.grad_scaler = amp.GradScaler(enabled=self.use_amp)
        if self.use_ema:
            self.ema_model = ModelEMA(self.model)
            if self.use_cuda:
                self.ema_model.cuda()
            

        ### define metrics ###
        self.metrics = {
            'loss': MeanMetric(),
        }
        ######################
        
        ########
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0015 ** 0.5, 0.0155 ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        self.betas = self.betas[:, None, None, None]
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones_like(self.alphas_cumprod[:1]), self.alphas_cumprod[:-1]], dim=0)
        self.betas, self.alphas_cumprod, self.alphas_cumprod_prev = self._convert_cuda_data([self.betas, 
                                                                                             self.alphas_cumprod, 
                                                                                             self.alphas_cumprod_prev])
        if self.use_amp:
            self.betas = self.betas.half()
            self.alphas_cumprod = self.alphas_cumprod.half()
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.half()

    def fit(self, train_loader, val_loader=None, epochs=1, verbose=True):
        history = {name: [] for name in self.metrics.keys()}
        if val_loader is not None:  # return history for validation data
            history = {**history,
                       **{'val_' + name: [] for name in self.metrics.keys()}}

        for epoch in range(1, epochs + 1):
            ### model setting ###
            if self.use_cuda:
                self.model.cuda()
            self.model.train()
            ######################

            # train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            if verbose:
                train_iter = tqdm(train_iter, total=len(train_loader))
                train_iter.set_description(f'Epoch {epoch}/{epochs}')
            self._reset_metrics()

            for step, data in enumerate(train_iter):
                self.train_step(data)
                if verbose:  # show metrics on train data
                    train_iter.set_postfix(self._compute_metrics())
            self._update_history(history)

            if val_loader is not None:
                self.evaluate(val_loader, verbose=verbose)
                self._update_history(history, prefix='val_')
        return history

    def evaluate(self, val_loader, verbose=True):
        ### model setting ###
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()
        #####################

        val_iter = iter(val_loader)
        if verbose:
            val_iter = tqdm(val_iter, total=len(val_loader))
            val_iter.set_description(f'Eval')

        self._reset_metrics()

        for step, data in enumerate(val_iter):
            self.test_step(data)
            if verbose:
                val_iter.set_postfix(self._compute_metrics())  # show metrics on val data
        return self._compute_metrics()

    def predict(self, data_loader, verbose=True):
        ### model setting ###
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()
        #####################

        data_iter = iter(data_loader)
        if verbose:
            data_iter = tqdm(data_iter, total=len(data_loader))

        results = []
        for step, data in enumerate(data_iter):
            outputs = self.predict_step(data)
            results.append(outputs)

        if isinstance(outputs, (tuple, list)):  # multi outputs
            results = list(*zip(results))
            results = [torch.cat(tensor, dim=0) for tensor in results]
        else:
            results = torch.cat(results, dim=0)
        return results

    def _q_sample(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        return self.alphas_cumprod[t]**0.5 * x + (1 - self.alphas_cumprod[t])**0.5 * noise
    
    def train_step(self, x):
        ### config input data ###
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long()
        x, t = self._convert_cuda_data((x, t))
        #########################

        ### optimizer.zero_grad() ###
        self.optimizer.zero_grad()
        #############################
        with self.autocast():
            ### forward pass ###
            noise = torch.randn_like(x)
            xt = self._q_sample(x, t, noise=noise)
            noise_hat = self.model(xt, t=t)
            
            loss = self.criterion(noise_hat, noise)
            ####################

        ### update model ###
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.use_ema:
            self.ema_model.update_parameters(self.model)
        #####################

        ### update metrics ###
        loss = self._convert_not_training_data(loss)
        self.metrics['loss'].update(loss)
        #######################
        return

    def test_step(self, x):
        ### config input data ###
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long()
        x, t = self._convert_cuda_data((x, t))
        #########################

        with self.autocast():
            with torch.no_grad():
                ### forward pass ###
                noise = torch.randn_like(x)
                xt = self._q_sample(x, t, noise=noise)
                noise_hat = self.model(xt, t)

                loss = self.criterion(noise_hat, noise)
                ####################

        ### update metrics ###
        loss = self._convert_not_training_data(loss)
        self.metrics['loss'].update(loss)
        #######################
        return

    def _predict_start_from_noise(self, xt, t, noise):
        return (1. / self.alphas_cumprod[t])**0.5 * xt - (1. / self.alphas_cumprod[t] - 1)**0.5 * noise
    
    def _q_posterior(self, x0, xt, t):
        coef1 = self.betas[t] * (self.alphas_cumprod_prev[t])**0.5 / (1. - self.alphas_cumprod[t])
        coef2 = (1. - self.alphas_cumprod_prev[t]) * (1 - self.betas[t])**0.5 / (1. - self.alphas_cumprod[t])
        mu = coef1 * x0 + coef2 * xt
        var = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        logvar = torch.log(torch.clip(var, 1e-20))
        return mu, var, logvar
    
    def _p_mean_variance(self, xt, t):
        noise_hat = self.ema_model(xt, t=t) if self.use_ema else self.model(xt, t=t)
        x0_hat = self._predict_start_from_noise(xt, t, noise_hat)
        mu, var, logvar = self._q_posterior(x0_hat, xt, t)
        return mu, var ,logvar
    
    @torch.no_grad()
    def _p_sample(self, xt, t):
        mu, _, logvar = self._p_mean_variance(xt, t)
        noise = torch.randn_like(xt)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
        return mu + nonzero_mask * (0.5 * logvar).exp() * noise
    
    @torch.no_grad()
    def sampling(self, shape=None, xt=None, T=None):
        if T is None:
            T = self.num_timesteps
        if xt is None:
            xt = torch.randn(*shape)
        if shape is None:
            shape = xt.shape
        xt = self._convert_cuda_data(xt)
        for i in tqdm(reversed(range(0, T)), desc='Sampling t', total=T):
            xt = self._p_sample(xt, torch.full((shape[0],), i, device=xt.device, dtype=torch.long)).detach()
        
        return xt.cpu()
    
    def predict_step(self, data):
        ### config input data ###
        shape = data
        #########################

        with self.autocast():
            with torch.no_grad():
                ### forward pass ###
                x0 = self.sampling(shape=shape)
                ####################

        return x0

    def _initialize_cuda(self, use_cuda=None, use_amp=True):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.use_amp = use_amp and self.use_cuda
        self.autocast = amp.autocast if self.use_amp else nullcontext

    def _convert_not_training_data(self, data, detach=True, cpu=True, numpy=False):
        if isinstance(data, (tuple, list)):
            return [self._convert_not_training_data(e) for e in data]
        elif isinstance(data, dict):
            return {k: self._convert_not_training_data(v) for k, v in data.items()}
        else:
            if detach:
                data = data.detach()
            if cpu:
                data = data.cpu()
            if numpy:
                data = data.numpy()
            return data

    def _convert_cuda_data(self, data):
        if not self.use_cuda:
            return data
        if isinstance(data, (tuple, list)):
            return [self._convert_cuda_data(e) for e in data]
        elif isinstance(data, dict):
            return {k: self._convert_cuda_data(v) for k, v in data.item()}
        else:
            return data.cuda()

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def _update_history(self, history, prefix=''):
        metrics = self._compute_metrics()
        for name, value in metrics.items():
            history[prefix + name].append(value)