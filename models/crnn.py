import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_utils import BlocksTorch


class CRNN(nn.Module):

    def __init__(self, img_h, n_channels, n_classes, n_hidden, lstm_input=64, return_cls=False):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'
        self.return_cls = return_cls

        block_0 = BlocksTorch.conv_norm_act(n_channels, 64, pooling='max')
        block_1 = BlocksTorch.conv_norm_act(64, 128, pooling='max')
        block_2 = BlocksTorch.conv_norm_act(128, 256)
        block_3 = BlocksTorch.conv_norm_act(256, 256, pooling='max', pooling_s=(2, 1), pooling_k=(2, 1))
        block_4 = BlocksTorch.conv_norm_act(256, 512, norm="bn")
        block_5 = BlocksTorch.conv_norm_act(512, 512, pooling='max', norm="bn", pooling_s=(2, 1), pooling_k=(2, 1))
        block_6 = BlocksTorch.conv_norm_act(512, 512, k=2, p=0)
        self.cnn = nn.Sequential(block_0, block_1, block_2, block_3, block_4, block_5, block_6)
        self.rnn = nn.Sequential(
            nn.Linear(512 * (img_h // 16 - 1), lstm_input),
            nn.LSTM(lstm_input, n_hidden, bidirectional=True),
            nn.LSTM(2 * n_hidden, n_hidden, bidirectional=True)
        )
        self.classifier = nn.Linear(2 * n_hidden, n_classes)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # rnn features
        output = self.rnn(conv)
        cls = self.classifier(output)
        # add log_softmax to converge output
        output = F.log_softmax(cls, dim=2)
        if self.return_cls:
            return output, F.softmax(cls, 2)

        return output


if __name__ == '__main__':
    m = CRNN(32, 1, 100, 256)
    print(m)
    # sample = torch.randn((1, 1, 32, 100))
    # m(sample)
    # summary(m, input_size=(1, 32, 100), batch_size=1, device='cpu')
