import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
import utils
import dataset
import models.crnn as net
import params

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""


def data_loader():
    train_dataset = dataset.CRNNDataset(root=params.TRAIN_ROOT, transform=params.transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.batchSize,
                                               shuffle=True,
                                               num_workers=int(params.workers),
                                               )

    val_dataset = dataset.CRNNDataset(root=params.VAL_ROOT,
                                      transform=dataset.ResizeNormalize((params.imgW, params.imgH)))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=True,
                                             batch_size=params.batchSize,
                                             num_workers=int(params.workers))

    return train_loader, val_loader


train_loader, val_loader = data_loader()

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))

    return crnn


crnn = net_init()
print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""
# image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
# text = torch.LongTensor(params.batchSize * 5)
# length = torch.LongTensor(params.batchSize)

# if params.cuda and torch.cuda.is_available():
#     criterion = criterion.cuda()
#     image = image.cuda()
#     text = text.cuda()
#
#     crnn = crnn.cuda()
#     if params.multi_gpu:
#         crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

# image = Variable(image)
# text = Variable(text)
# length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

criterion = CTCLoss(zero_infinity=True)


# -----------------------------------------------

def val(net, criterion):
    print('Start val')
    net.eval()
    n_correct = 0
    loss_avg = utils.averager()  # The blobal loss_avg is used by train
    for i, data in enumerate(val_loader):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        image = cpu_images.to(device)
        text, length = converter.encode(cpu_texts)
        text = text.to(device)
        length = length.to(device)
        with torch.no_grad():
            preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        # cpu_texts_decode = []
        # for i in cpu_texts:
        #     cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        cpu_texts_decode = cpu_texts
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / len(val_loader.dataset)  # float(len(val_loader) * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train(crnn, criterion, optimizer, data):
    crnn.train()
    cpu_images, cpu_texts = data
    cpu_images = cpu_images.to(device)
    batch_size = cpu_images.size(0)
    text, length = converter.encode(cpu_texts)
    text = text.to(device)
    length = length.to(device)
    image = cpu_images
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = torch.LongTensor([preds.size(0)] * batch_size)
    cost = criterion(preds, text, preds_size, length) / batch_size
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crnn = crnn.to(device)
    j = 0
    for epoch in range(params.nepoch):
        for i, data in enumerate(train_loader):
            j += 1
            cost = train(crnn, criterion, optimizer, data)
            loss_avg.add(cost)
            if j % params.displayInterval == 0:
                print('[%d/%d][%d] Loss: %f' %
                      (epoch, params.nepoch, j, loss_avg.val()))
                loss_avg.reset()

            if j % params.valInterval == 0:
                val(crnn, criterion)
                save_path = f'{params.expr_dir}/{params.NAME}_{epoch}.pth'
                torch.save(crnn.state_dict(), save_path)
                print(f'model saved in {save_path}')

    torch.save(crnn.state_dict(), f'{params.expr_dir}/{params.NAME}_{epoch}.pth')
