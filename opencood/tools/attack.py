# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
import copy


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(np.array([splits[0], splits[1], splits[2]]))

    return data_list


dataset_root = '/home/fulan/Datasets/lfw/train_C'


def load_image(img_path):
    image = cv2.imread(os.path.join(dataset_root, img_path))
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))  # fliplr means flip left-right
    # image = image.transpose((2, 0, 1))
    # image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def adapt_mask_gaussblur(original, mask_org):
    max_iterations = 150
    tv_beta = 3
    learning_rate = 0.05
    l1_coeff = 2.0
    tv_coeff = 0.1

    # mask_ = F.upsample(mask_org.unsqueeze(0), (128, 128), mode='bilinear').detach().cpu()
    mask_ = mask_org.cuda().float()
    mask_.requires_grad_()
    optimizer = torch.optim.Adam([mask_], lr=learning_rate)
    blurred = cv2.GaussianBlur(original.detach().cpu().numpy(), (11, 11), 5).transpose(2, 0, 1)

    blurred = torch.from_numpy(blurred).cuda()
    original = original.permute(2, 0, 1)

    def tv_norm(input, tv_beta):
        img = input
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
        return row_grad + col_grad

    for i in range(max_iterations):
        mask = mask_
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        mask = mask.unsqueeze(0).repeat(3, 1, 1)
        # Use the mask to perturbated the input image.
        # perturbated_input = original.mul(1 - mask) + \
        #                     blurred.mul(mask)

        # outputs = torch.nn.Softmax()(model._model(perturbated_input.unsqueeze(0)))

        loss = l1_coeff * torch.mean(torch.abs(mask)) + \
               tv_coeff * tv_norm(mask, tv_beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if self.disp:
        #     vis = visdom.Visdom(env='Adversarial Example Showing')
        #     vis.images(mask.data.clamp_(0, 1), win='mask_')

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    mask.data.clamp_(0, 1)
    mask[mask >= 0.5] = 1.
    mask[mask < 0.5] = 0.

    return mask


def get_features(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))

            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


def gen_random_perspective_transform(params):
    """ generate a batch of 3x3 homography matrices by composing rotation, translation, shear, and projection matrices,
    where each samples components from a uniform(-1,1) * multiplicative_factor
    """

    batch_size = params.batch_size

    # debugging
    if params.dict.get('identity_transform_only'):
        return torch.eye(3).repeat(batch_size, 1, 1).to(params.device)

    I = torch.eye(3).repeat(batch_size, 1, 1)
    uniform = torch.distributions.Uniform(-1, 1)
    factor = 0.25
    c = copy.deepcopy

    # rotation component
    a = math.pi / 6 * uniform.sample((batch_size,))
    R = c(I)
    R[:, 0, 0] = torch.cos(a)
    R[:, 0, 1] = - torch.sin(a)
    R[:, 1, 0] = torch.sin(a)
    R[:, 1, 1] = torch.cos(a)
    R.cuda()

    # translation component
    tx = factor * uniform.sample((batch_size,))
    ty = factor * uniform.sample((batch_size,))
    T = c(I)
    T[:, 0, 2] = tx
    T[:, 1, 2] = ty
    T.cuda()

    # shear component
    sx = factor * uniform.sample((batch_size,))
    sy = factor * uniform.sample((batch_size,))
    A = c(I)
    A[:, 0, 1] = sx
    A[:, 1, 0] = sy
    A.cuda()

    # projective component
    px = uniform.sample((batch_size,))
    py = uniform.sample((batch_size,))
    P = c(I)
    P[:, 2, 0] = px
    P[:, 2, 1] = py
    P.cuda()

    # compose the homography
    H = R @ T @ P @ A

    return H


import matplotlib.pyplot as plt


def show_tensor_image(tensor):
    x = tensor.detach().cpu().numpy()
    plt.imshow(x)


def initialize(model, std=0.1):
    for p in model.parameters():
        p.data.normal_(0, std)

    # init last linear layer of the transformer at 0
    model.transformer.net[-1].weight.data.zero_()
    model.transformer.net[-1].bias.data.copy_(torch.eye(3).flatten()[:model.transformer.net[-1].out_features])


def lfw_test_attack(model, identity_list, compair_list, batch_size):
    s = time.time()

    images = None
    features = None
    cnt = 0

    from spatial_transformer import utils as stn_utils
    from spatial_transformer import model as stn_model
    json_path = os.path.join('./spatial_transformer/experiments/base_stn_model/params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = stn_utils.Params(json_path)

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(11052018)
    if params.device.type == 'cuda': torch.cuda.manual_seed(11052018)

    P_init = gen_random_perspective_transform(params)
    P_init = P_init.to(params.device)
    stn = stn_model.STN(getattr(stn_model, params.stn_module), params, P_init).to(params.device)
    # stn_utils.load_checkpoint(restore_file, STN)
    for param in stn.parameters():
        param.requires_grad = True

    for i, img_path in enumerate(identity_list):
        input_image = load_image(img_path[0])
        target_image = load_image(img_path[1])
        label = img_path[2]

        if input_image.shape[0] % batch_size == 0:
            cnt += 1

            data0 = torch.from_numpy(input_image).permute(2, 0, 1)
            data0 = data0.to(torch.device("cuda"))

            data1 = torch.from_numpy(target_image).permute(2, 0, 1)
            data1 = data1.to(torch.device("cuda"))

            shape_index = np.random.randint(1, 131)
            mask_path = os.path.join('./shape_silouette', os.listdir('./shape_silouette')[shape_index - 1])
            inten = 0.5
            inten = torch.tensor(inten).float().cuda()
            size = 0.5
            size = torch.tensor(size).float().cuda()
            theta = torch.tensor([
                [1., 0., 0.],
                [0., 1., 0.]
            ], dtype=torch.float).cuda().unsqueeze(0)

            #
            mask = Image.open(mask_path)
            # mask resize to the same shape of original
            mask = Image.Image.resize(mask, (128, 128))
            import matplotlib.pyplot as plt
            plt.imshow(np.asarray(mask))
            mask = torch.from_numpy(np.asarray(mask)).cuda().float().unsqueeze(0).unsqueeze(0)
            step_size = 0.1

            num_iter = 40
            for itr in range(num_iter):
                inten.requires_grad_()
                theta.requires_grad_()
                mask.requires_grad_()

                occl = inten * data0
                flow = F.affine_grid(theta, mask.size(), align_corners=True)
                mask_stn = F.grid_sample(input=mask, grid=(flow), mode='bilinear', align_corners=True)
                mask_stn = mask_stn.clamp(0, 1)
                # show_tensor_image(mask_stn.squeeze(0).squeeze(0))

                mask_stn = mask_stn.squeeze(0).repeat((3, 1, 1))
                adv = occl * mask_stn + (1 - mask_stn) * data0

                # show_tensor_image((data0.permute(1, 2, 0) + 1) / 2)
                # show_tensor_image((occl.permute(1, 2, 0) + 1) / 2)
                # show_tensor_image((adv.permute(1, 2, 0) + 1) / 2)

                adv = adv.clamp(-1, 1)

                # recoginition loss

                adv = adv.squeeze(0)
                input = adv[2, ...] * 0.2989 + adv[1, ...] * 0.5870 + adv[0, ...] * 0.1140
                feature0 = model(input.unsqueeze(0).unsqueeze(0))

                input1 = data1[2, ...] * 0.2989 + data1[1, ...] * 0.5870 + data1[0, ...] * 0.1140
                feature1 = model(input1.unsqueeze(0).unsqueeze(0))
                loss = 1 - torch.cosine_similarity(feature0.squeeze(0), feature1.squeeze(0), dim=0)
                # loss = torch.sqrt(torch.sum(torch.square(feature0 - feature1)))
                loss.backward()
                grad_inten = inten.grad
                grad_theta = theta.grad
                grad_mask = mask.grad

                inten = inten + step_size * grad_inten.sign()
                inten = inten.clamp(0.2, 0.8)
                theta = theta + step_size * grad_theta.sign()
                mask = mask + step_size * grad_mask.sign()
                mask = mask.clamp(0, 1)

                inten.detach_()
                theta.detach_()
                mask.detach_()
                occl = inten * data0
                flow = F.affine_grid(theta, mask.size(), align_corners=True)
                mask_stn = F.grid_sample(input=mask, grid=(flow), mode='bilinear', align_corners=True)
                mask_stn = mask_stn.clamp(0, 1)
                mask_stn = mask_stn.squeeze(0).repeat((3, 1, 1))
                adv = occl * mask_stn + (1 - mask_stn) * data0
                adv = adv.clamp(-1, 1)
                adv = adv.squeeze(0)
                input = adv[2, ...] * 0.2989 + adv[1, ...] * 0.5870 + adv[0, ...] * 0.1140
                feature0 = model(input.unsqueeze(0).unsqueeze(0))
                sim = cosin_metric(np.squeeze(feature0.detach().cpu().numpy()),
                                   np.squeeze(feature1.detach().cpu().numpy()))
                y_score = np.asarray(sim)
                strict = False
                if int(label) == 1:
                    strict = True
                if (float(y_score) >= 0.20) != strict:
                    print(inten)
                    if not os.path.exists('./attack/' + img_path[0][:img_path[0].rfind('/')]):
                        os.makedirs('./attack/' + img_path[0][:img_path[0].rfind('/')])
                    img = (adv.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255
                    cv2.imwrite('./attack/' + img_path[0], img)
                    cv2.imwrite('./attack/' + img_path[0].replace('.jpg', '') + '_mask.jpg',
                                mask_stn.detach().cpu().numpy()[0, ...] * 255)
                    cv2.imwrite('./attack/' + img_path[0].replace('.jpg', '') + '_occl.jpg',
                                (occl.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255)
                    break


from PIL import Image


def read_path_arr_generate_shadow_list(filename):
    with open(filename) as f:
        lines = f.readlines()

    name_list = []
    mask_list = []
    for line in lines:
        line = line.strip('\n')
        name_list.append(line)
        mask_list.append(line.replace('lfw-align-128', 'lfw-align-mask'))
    name_list = np.array(name_list)
    mask_list = np.array(mask_list)

    sil_path = '/home/fulan/face_recognition/portrait-shadow-manipulation/silhouette'
    files = os.listdir(sil_path)
    sil_list = []
    for file in files:
        sil_list.append(os.path.join(sil_path, file))
    sil_list = np.array(sil_list)
    len_img = name_list.shape[0]
    len_sil = sil_list.shape[0]
    if len_sil < len_img:
        cnt = len_img // len_sil + 1
        sil_list = np.tile(sil_list, cnt)
        sil_list = sil_list[:len_img]

    bbx = np.array([0, 0, 127, 127])[np.newaxis, :]
    bbx = np.tile(bbx, [len_img, 1])

    with open('./example.txt', 'w') as ff:
        for i in range(name_list.shape[0]):
            line = name_list[i] + ',' + mask_list[i] + ',' + sil_list[i] + ',0,0,128,128' + '\n'
            ff.write(line)

    # generate mask for each face image
    cnt = 0
    for line in lines:
        mask = np.ones((128, 128, 3), dtype='uint8') * 255
        img = Image.fromarray(mask)
        str = line.replace('lfw-align-128', 'lfw-align-mask').strip('\n')
        os_path = str[:str.rfind('/')]
        if not os.path.exists(os_path):
            os.makedirs(os_path)
        img.save(str)
        cnt += 1
    print(cnt)


if __name__ == '__main__':
    # read_path_arr_generate_shadow_list('./path_arr.txt')

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)

    # save the image list
    # path_arr = np.array(img_paths)
    # np.savetxt('./path_arr.txt', path_arr, fmt='%s')

    model.eval()
    lfw_test_attack(model, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
