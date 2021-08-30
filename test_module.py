import os
import shutil
import time
import cv2
import torch
from torch.autograd import Variable
import utils
import dataset
import models.crnn as crnn
import params

model_path = 'expr/netCRNN_999.pth'
image_dir = '/home/ai/projects/Irancel/recognition/train'
image_res = '/home/ai/projects/Irancel/recognition/train_problematic'
os.makedirs(image_res, exist_ok=True)
device = 'cpu'

nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
converter = utils.strLabelConverter(params.alphabet)
transformer = dataset.CV2ResizeNormalize((params.imgW, params.imgH))

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = transformer(image)
    image = image.to(device)
    image = image.view(1, *image.size())
    tic = time.time()
    with torch.no_grad():
        preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    toc = time.time()
    # print(toc - tic)
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[-1]
    if image_name != sim_pred:
        print(f'{image_name == sim_pred} %-20s => %-20s   |   {image_name}' % (raw_pred, sim_pred))
        shutil.copy(image_path, os.path.join(image_res, f'{image_name}---{sim_pred}.jpg'))
