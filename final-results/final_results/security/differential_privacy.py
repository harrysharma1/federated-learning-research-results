from opacus import PrivacyEngine
import torch

def apply_differential_privacy(model, optimizer, training_loader, noise_multiplier=1.0, max_graient_norm=1.0):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        training_loader (_type_): _description_
        noise_multiplier (float, optional): _description_. Defaults to 1.0.
        max_graient_norm (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    privacy_engine = PrivacyEngine()
    model, optimizer, training_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=training_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_graient_norm
    )
    return model, optimizer, training_loader, privacy_engine