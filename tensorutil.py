from PIL import Image
import torchvision.transforms as T

#Utility code for tensors

#Single frame sigma transformation described in HLC
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84, 84), interpolation=Image.CUBIC),
                    #T.Grayscale(),
                    T.ToTensor()])


def sigma(observation):
    return resize(observation)


