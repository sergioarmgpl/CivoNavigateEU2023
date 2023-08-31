import time

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def classify():
  labels = read_label_file("test_data/inat_bird_labels.txt")
  interpreter = make_interpreter("test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite")
  interpreter.allocate_tensors()

  # Model must be uint8 quantized
  if common.input_details(interpreter, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')

  size = common.input_size(interpreter)
  image = Image.open("test_data/parrot.jpg").convert('RGB').resize(size, Image.ANTIALIAS)

  params = common.input_details(interpreter, 'quantization_parameters')
  scale = params['scales']
  zero_point = params['zero_points']
  mean = 128.0
  std = 128.0
  if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
    common.set_input(interpreter, image)
  else:
    # Input data requires preprocessing
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter, normalized_input.astype(np.uint8))

  # Run inference
  logs = ""
  logs = "----INFERENCE TIME----"
  logs += "Note: The first inference on Edge TPU is slow because it includes\n" +\
          "loading the model into Edge TPU memory."
  for _ in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreter, 1, 0.0)
    logs += ('%.1fms' % (inference_time * 1000))

  logs += '-------RESULTS--------'
  for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
  print(logs)
  return logs
