import os
import torch
from PIL import Image
import open_clip

def test():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
    # text = tokenizer(["a diagram", "a dog", "a cat"])

    tx1 = tokenizer([input("Text 1: "),input("Text 2: ")])
    # tx2 = tokenizer([input("Text 2: ")])

    with torch.no_grad(), torch.cuda.amp.autocast():
        # image_features = model.encode_image(image)
        text_features1 = model.encode_text(tx1)
        # text_features2 = model.encode_text(tx2)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("finish!")
    import ipdb;ipdb.set_trace()
    # print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

if __name__=="__main__":
    test()