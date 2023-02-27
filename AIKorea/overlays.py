import numpy as np

def overlay_transparent(background, foreground, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]
    try:

        if x >= background_width or y >= background_height:
            return background

        h, w = foreground.shape[0], foreground.shape[1]

        if x + w > background_width:
            w = background_width - x
            foreground = foreground[:, :w]

        if y + h > background_height:
            h = background_height - y
            foreground = foreground[:h]

        if foreground.shape[2] < 4:
            foreground = np.concatenate(
                [
                    foreground,
                    np.ones((foreground.shape[0], foreground.shape[1], 1), dtype=foreground.dtype) * 255
                ],
                axis=2,
            )

        overlay_image = foreground[..., :3]
        mask = foreground[..., 3:] / 255.0

        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image
    except:
        pass

    return background