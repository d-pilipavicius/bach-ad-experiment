import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

def output_to_marked_pil(output) -> Image:
  img = output.image.squeeze(0).detach().cpu().numpy()
  img = np.transpose(img, (1, 2, 0))

  img = (img - img.min()) / (img.max() - img.min() + 1e-8)
  img = (img * 255).astype(np.uint8)

  pil_img = Image.fromarray(img).convert("RGBA")
  mask = output.pred_mask.squeeze(0).detach().cpu().numpy().astype(bool)
  edges = binary_dilation(mask, iterations=1) ^ mask

  overlay = Image.new("RGBA", pil_img.size)
  draw = ImageDraw.Draw(overlay)

  h, w = mask.shape

  for y in range(h):
    for x in range(w):
      if edges[y, x]:
        draw.point((x, y), fill=(255, 0, 0, 255))

  img_with_border = Image.alpha_composite(pil_img, overlay)

  return img_with_border