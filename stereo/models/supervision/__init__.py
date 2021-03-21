from . import true_value

TRUE_VALUE_GENERATORS = {
    'MultiScaleTrueValues': true_value.MultiScaleTrueValues,
    'OriScaleTrueValues': true_value.OriScaleTrueValues
}

from . import basic_losses

LOSS_COMPUTERS = {
    'MultiScaleLoss': basic_losses.MultiScaleLoss,
    'OriScaleLoss': basic_losses.OriScaleLoss
}