### Latent Space

Latent means "hidden" or "concealed". In the context of neural networks, a latent space often means a feature space, and a latent vector is just a compressed, feature-level representation of an image!

For example, when you created a simple autoencoder, the outputs that connected the encoder and decoder portion of a network made up a compressed representation that could also be referred to as a latent vector.

You can read more about latent space in [this blog post] as well as an interesting property of this space: recall that we can mathematically operate on vectors in vector space and with latent vectors, we can perform a kind of feature-level transformation on an image!

```
This manipulation of latent space has even been used to create an interactive GAN, iGAN for interactive image generation! I recommend reading the paper, linked in the Github readme.
```

* https://github.com/junyanz/iGAN/blob/master/README.md
