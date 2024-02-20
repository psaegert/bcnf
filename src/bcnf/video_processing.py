import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

# TODO: Take care of time and timeframes, so far the entire video is converted


def gmm_approximation(frames: np.ndarray,
                      gif_name: str = 'video_gif',
                      save_gif: bool = True,
                      ratio: tuple = (16, 9)) -> np.ndarray:
    gmm = GaussianMixture(n_components=1, covariance_type='spherical', random_state=42)

    heatmap = []

    for i in range(len(frames)):
        if np.sum(frames[i]) != 0:  # if frame is not black
            gmm.fit(np.argwhere(frames[i] != 0))

            # sample from gaussian
            sample = gmm.sample(5000)

            # plot in 2d histogram
            hist = np.histogram2d(sample[0][:, 0], sample[0][:, 1], bins=(ratio[1] * 10, ratio[0] * 10), range=((0, ratio[1] * 10), (0, ratio[0] * 10)))[0]

            # normalize
            hist = hist / np.sum(hist)

            # save frame
            heatmap.append(hist)

        else:  # if black, no fit
            heatmap.append(np.zeros((ratio[1] * 10, ratio[0] * 10)))

    if save_gif:
        # animation with heatmaps
        ims = []
        fig, ax = plt.subplots()
        for frame in heatmap:
            im = ax.imshow(frame, cmap='hot', animated=True)

            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=33.3, blit=True, repeat_delay=1000)
        ani.save(f'gifs/{gif_name}_gmm.gif', writer='imagemagick')

    return heatmap


def process_video(video_path: str,
                  gif_name: str = 'video_gif',
                  save_gif: bool = False,
                  ggm_approximation: bool = True,
                  ratio: tuple = (16, 9)) -> np.ndarray:
    # read video
    cap = cv2.VideoCapture(f'{video_path}')

    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # read video frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames = np.array(frames[:-1])  # somehow tha last frame often makes some problems...

    # calculate time average as background
    time_average = np.mean(frames, axis=0)

    # calculate the euclidian norm between each pixel of frame and time average in each frame
    frame_diff = []
    for i in range(len(frames)):
        frame_diff.append(np.sum((frames[i] - time_average)**2, axis=2)**(1 / 2))

    # resize frames to simulated camera resolution
    factor = int(width / (ratio[0] * 10))

    frames_resized = []
    for frame in frame_diff:
        blocks = frame.reshape(frame.shape[0] // factor, factor, frame.shape[1] // factor, factor)
        block_averages = np.mean(blocks, axis=(1, 3))
        frames_resized.append(block_averages.reshape(ratio[1] * 10, ratio[0] * 10))

    frames = np.array(frames_resized)

    # thresholding to remove small noise
    assert type(frames) is np.ndarray  # does this keep flake8 happy?
    frames[frames < 100] = 0  # arbitrary threshold, feel free to play around with it

    # calculate average value per frame, and either black out or normalize the frame
    sums = np.sum(frames, axis=(1, 2))

    for i in range(len(sums)):
        if sums[i] < 1500:  # second threshold to check wether object is in frame, again: arbitrary
            frames[i] = np.zeros((ratio[1] * 10, ratio[0] * 10))
        else:
            frames[i] = frames[i] / sums[i]

    if save_gif and not ggm_approximation:
        # these values are only for better visibility in the gif
        vmin = np.min(frames)
        vmax = np.max(frames) / 5

        # create gif
        ims = []
        fig, ax = plt.subplots()
        for frame in frames:
            im = ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, animated=True)

            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=33.3, blit=True, repeat_delay=1000)
        ani.save(f'gifs/{gif_name}.gif', writer='imagemagick')

    if ggm_approximation:
        return gmm_approximation(frames, gif_name, save_gif, ratio)

    else:
        return frames
