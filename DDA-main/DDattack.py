from Modeling.DerainDataset import *
from Modeling.utils import *
from Modeling.network import *
import time
from option import *
from attack_tools import gen_pgd_confs
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import timm
import torchvision
from torch.autograd import Variable


os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="max_split_size_mb:256"

classifier = torchvision.models.resnet50(pretrained=True).eval().cuda()

def generate_x_adv(x, y, pgd_conf,model, device):

    delta = torch.zeros(x.shape).to(x.device)
    loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    # 初始化参数
    eps = pgd_conf['eps']
    alpha = pgd_conf['alpha']
    iter = pgd_conf['iter']

    x_norain, _ = model(x)
    mask = x - x_norain
    mask1 = abs(mask)
    mask1[mask1 > 0.01] = 1
    mask1[mask1 <= 0.01] = 0
    x_norain = torch.clamp(x_norain, 0., 1.)
    x_norain.requires_grad_()
    d = delta

    for pgd_iter_id in range(iter):

        with torch.enable_grad():
            # 损失
            loss = loss_fn(classifier(x_norain + d), y)
            loss.backward()
            grad_sign = x_norain.grad.data.sign()
            # 扰动强度
            delta += grad_sign * alpha
            delta = torch.clamp(delta, -eps, eps)  # 噪声裁剪
            d = mask1 * delta

        torch.cuda.empty_cache()

    print("Done")
    # 图像裁剪


    x_adv = x_norain + d
    x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.detach()


def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [Id for Id in range(torch.cuda.device_count())]

    os.makedirs(opt.output_path, exist_ok=True)
    pgd_conf = gen_pgd_confs(eps=100, alpha=1, iter=20, input_range=(0, 1))

    # Build model
    print('Loading model ...\n')
    model = SAPNet(recurrent_iter=opt.recurrent_iter,
                   use_dilation=opt.use_dilation).to(device)
    # print_network(model)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_latest.pth'), map_location=device))
    model.eval()

    time_test = 0
    count = 0
    num = 0

    imglist=os.listdir(opt.data_path)
    imglist=sorted(imglist)
    for img_name in imglist:
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            x = cv2.imread(img_path)
            x = cv2.resize(x, (224, 224))
            b, g, r = cv2.split(x)
            x = cv2.merge([r, g, b])

            x = normalize(np.float32(x))
            x = np.expand_dims(x.transpose(2, 0, 1), 0)
            x = Variable(torch.Tensor(x)).to(device)

            with torch.no_grad(): #
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()

                y, _ = model(x)
                y = torch.clamp(y, 0., 1.)
                x_pred = classifier(y).argmax(1)  # original prediction
                out = generate_x_adv(x, x_pred, pgd_conf, model, 0)


                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if torch.cuda.is_available():
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.output_path, img_name), save_out)

            count += 1
            with open('imagenet_classes.txt', errors='ignore') as f:
                labels = [line.strip() for line in f.readlines()]
                label1 = torch.argmax(classifier(x), 1)
                label2 = torch.argmax(classifier(out), 1)
                print("the rainy current predict class name : %s" % labels[label1[0]])
                print("the DDA current predict class name : %s" % labels[label2[0]])

                src1 = cv2.imread(os.path.join('/datasets/test/nips-17', img_name))
                image1 = cv2.resize(src1, (224, 224))
                image1 = np.float32(image1) / 255.0
                image1[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
                image1[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
                image1 = image1.transpose((2, 0, 1))
                input_x1 = torch.from_numpy(image1).unsqueeze(0)
                pred1 = classifier(input_x1.cuda())
                pred_index1 = torch.argmax(pred1, 1).cpu().detach().numpy()
                print("the GT current predict class name: %s" % labels[pred_index1[0]])


                if labels[label2[0]] != labels[pred_index1[0]]:
                    num += 1

    print('the success rate of attacks: ', num / count * 100, '%')

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    test()