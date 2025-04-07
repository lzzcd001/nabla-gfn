from typing import Optional, Tuple, Union

import math
import torch

try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers import DPMSolverSinglestepScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def get_alpha_prod_t(self, timestep, sample):
    # 2. compute alphas, betas
    # self.alphas_cumprod  torch.Size([1000])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    return alpha_prod_t


def inference_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 1.0,
    generator = None,
    prev_sample: Optional[torch.FloatTensor] = None,
    prev_timestep: int = None,
    reverse_grad: bool = False,
    retain_graph: bool = False,
    allow_2nd: bool = False,
    strength: float = 1.0,
    step_index: int = 0
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        sample: x_t
        prev_sample: x_{t-1} (closer to clean image)

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

    """
    # assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if isinstance(self, DPMSolverSinglestepScheduler):

        if isinstance(step_index, int):
            step_index = torch.tensor((step_index, )).expand(sample.size(0))

        random_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )

        if self.config.prediction_type == "epsilon":
            sigma = self.sigmas.gather(0, step_index.cpu())
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            sigma_t = _left_broadcast(sigma_t, sample.shape).to(sample.device)
            alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "sample":
            x0_pred = model_output
        elif self.config.prediction_type == "v_prediction":
            sigma = self.sigmas.gather(0, step_index.cpu())
            sigma_t = _left_broadcast(sigma_t, sample.shape).to(sample.device)
            alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction` for the DPMSolverSinglestepScheduler."
            )

        sigma_t = self.sigmas.gather(0, step_index.cpu() + 1)  # torch scalar
        sigma_s = self.sigmas.gather(0, step_index.cpu())  # torch scalar
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s

        sigma_t = _left_broadcast(sigma_t, sample.shape).to(sample.device)
        sigma_s = _left_broadcast(sigma_s, sample.shape).to(sample.device)
        alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
        h = _left_broadcast(h, sample.shape).to(sample.device)
        prev_sample_mean = (
            (sigma_t / sigma_s * torch.exp(-h)) * sample
            + (alpha_t * (1 - torch.exp(-2.0 * h))) * x0_pred
        )
        std_dev_t = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        prev_sample = prev_sample_mean + std_dev_t * random_noise

        if reverse_grad:
            target = ((prev_sample.detach() - prev_sample_mean) ** 2).sum()
            grad = torch.autograd.grad(
                outputs=target,                 # The value whose gradient we want
                inputs=sample,                  # The intermediate node we want the gradient with respect to
                retain_graph=retain_graph,      # Retain graph for further gradient computations
                create_graph=allow_2nd          # If higher-order gradients are needed
            )[0]
            score = -grad / (2 * (std_dev_t ** 2)) * strength
        else:
            score = -(prev_sample.detach() - prev_sample_mean) / (std_dev_t ** 2) * strength

        return prev_sample.type(sample.dtype), score

    elif isinstance(self, DDIMScheduler):
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_{t-1}"

        # 1. get previous step value (=t-1)
        if prev_timestep is None:
            prev_timestep = (
                timestep - self.config.num_train_timesteps // self.num_inference_steps
            )
        # to prevent OOB on gather
        prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

        # 2. compute alphas, betas
        # self.alphas_cumprod  torch.Size([1000])
        alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
        alpha_prod_t_prev = torch.where(
            prev_timestep.cpu() >= 0,
            self.alphas_cumprod.gather(0, prev_timestep.cpu()),
            self.final_alpha_cumprod,
        )
        alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
            sample.device
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (
                beta_prod_t ** 0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (
                beta_prod_t ** 0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        # cifar ddpm: self.config.thresholding = False, self.config.clip_sample_range = 1.0
        # SD: self.config.thresholding = False, self.config.clip_sample = False
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = _get_variance(self, timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)  # eta is 1.0
        std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (
            0.5
        ) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample_mean = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if prev_sample is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise


        if reverse_grad:
            target = ((prev_sample.detach() - prev_sample_mean) ** 2).sum()
            grad = torch.autograd.grad(
                outputs=target,                # The value whose gradient we want
                inputs=sample,                 # The intermediate node we want the gradient with respect to
                retain_graph=retain_graph,     # Retain graph for further gradient computations
                create_graph=allow_2nd         # If higher-order gradients are needed
            )[0]
            score = -grad / (2 * (std_dev_t ** 2)) * strength
        else:
            score = -(prev_sample.detach() - prev_sample_mean) / (std_dev_t ** 2) * strength

        return prev_sample.type(sample.dtype), score
    else:
        raise NotImplementedError


def predict_clean(
        self, 
        model_output, 
        sample: torch.FloatTensor, 
        timestep: int, 
        no_jacobian: bool = False,
        strength: float = 1.0
    ):
    with torch.no_grad():
        alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
        alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t = alpha_prod_t.to(sample.dtype)
        beta_prod_t = 1 - alpha_prod_t

        beta_prod_t[timestep == 0] = 0
        alpha_prod_t[timestep == 0] = 1

    if self.config.prediction_type == "epsilon":
        pred_clean_sample = (
            sample - beta_prod_t ** (0.5) * model_output * strength
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_clean_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_clean_sample = (alpha_prod_t ** 0.5) * sample - (
            beta_prod_t ** 0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    return pred_clean_sample