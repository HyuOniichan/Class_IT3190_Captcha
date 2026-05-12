from .loader import load_inference_model

from predictors.char_predictor import (
    CharacterPredictor
)

from predictors.captcha_predictor import (
    CaptchaPredictor
)

def build_captcha_predictor():
    model, device = load_inference_model()

    char_predictor = CharacterPredictor(
        model=model,
        device=device
    )

    predictor = CaptchaPredictor(
        char_predictor=char_predictor
    )

    return predictor