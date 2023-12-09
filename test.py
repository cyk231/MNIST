import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from CNN import CNN

model = CNN()
model.load_state_dict(torch.load('cnn.pth'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
unloader = transforms.ToPILImage()

for k in range(10):
    infile = 'test'+str(k)+'.png'

    img = Image.open(infile)
    img = img.resize((28, 28))
    img = img.convert('L')
    img_array = np.array(img)


    # 像素反转
    for i in range(28):
        for j in range(28):
            img_array[i, j] = 255 - img_array[i, j]
    # print(img_array)
    img = Image.fromarray(img_array)
    # img.show()
    img = transform(img)
    img = torch.unsqueeze(img, 0)

    output = model(img)
    pred = torch.argmax(output, dim=1)

    image = torch.squeeze(img, 0)
    image = unloader(image)

    plt.subplot(5, 2, k + 1)
    plt.tight_layout()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title("Number: {}, Prediction: {}".format(k, pred.item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
