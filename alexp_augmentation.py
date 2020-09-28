#%%
import tensorflow as tf
import matplotlib.pyplot as plt
from src.applications.resnet_custom import make_model

#%%
image_path = '/home/ngari/Dev/projectzeamays/data/NLB_stuff_resized/images_handheld/DSC00034.JPG'
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img)
plt.imshow(img)

#%%
img0 = tf.image.adjust_brightness(img, -0.3)
plt.imshow(img0)

#%%
img1 = tf.image.adjust_contrast(img, 0.5)
plt.imshow(img1)

#%%
img2 = tf.image.adjust_hue(img, 0.1)
plt.imshow(img2)

# %%
img3 = tf.image.adjust_jpeg_quality(img2, 10)
plt.imshow(img3)

# %%
img4 = tf.image.adjust_saturation(img, 1)
plt.imshow(img4)

# %%
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
model = make_model((256,256,3), None, optimizer, loss, extra_layers=True)
model.summary()

# %%
model.fit()