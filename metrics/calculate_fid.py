import numpy as np
import torch
from scipy import linalg
import torch.nn.functional as F
import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, classifier, batches_to_take=1000):
    classifier.eval()
    model.eval()
    device = next(classifier.parameters()).device
    assert dataloader.drop_last
    batch_size = dataloader.batch_size
    n_imgs = batch_size * batches_to_take
    real_ft_all = np.zeros((n_imgs, 2048))
    fake_ft_all = np.zeros((n_imgs, 2048))
    
    idx = 0
    dataloader_iter = iter(dataloader)
    for _ in tqdm.tqdm(range(batches_to_take)):
        image = next(dataloader_iter)
        image = image.to(device)
        real = image
        fake = model.sample(len(image))
        # Upsample the images, since the inception_v3 takes 299x299x3 images.
        real = F.interpolate(real, 299)
        fake = F.interpolate(fake, 299)
        real_features = classifier(real).detach().cpu().numpy()
        fake_features = classifier(fake).detach().cpu().numpy()
        real_ft_all[idx:idx+batch_size, :] = real_features
        fake_ft_all[idx:idx+batch_size, :] = fake_features
        idx += batch_size
        if idx >= n_imgs:
            break

    real_mu = np.mean(real_ft_all, axis=0)
    real_sigma = np.cov(real_ft_all, rowvar=False)
    fake_mu = np.mean(fake_ft_all, axis=0)
    fake_sigma = np.cov(fake_ft_all, rowvar=False)
    return real_mu, real_sigma, fake_mu, fake_sigma


@torch.no_grad()
def calculate_fid(dataloader, model, classifier):
    print('Calculating FID...')
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
