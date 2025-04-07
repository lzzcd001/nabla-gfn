from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "hpd_photo_painting" # for HPSv2
    config.experiment.reward_fn = "hpscore"
    config.experiment.prompt_fn_kwargs = {}

    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 16

    config.model.unet_reg_scale = 1e2
    config.model.reward_scale = 1e4


    return config