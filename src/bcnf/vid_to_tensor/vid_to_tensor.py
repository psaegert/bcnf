import torch
import torchvision.io as io


def video_to_tensor(video_path: str,
                    greyscale: bool = False,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load a video from a file and convert it to a tensor that is greyscale

    Parameters
    ----------
    video_path : str
        The path to the video file
    greyscale : bool
        Whether to convert the video to greyscale
    dtype : torch.dtype
        The datatype to convert the video to

    Returns
    -------
    video : torch.Tensor
        The video as a tensor
    """
    video, audio, info = io.read_video(video_path, pts_unit='sec')

    if not greyscale:
        video = video.to(dtype)
        video = torch.mean(video, dim=3, keepdim=False)

    return video


def two_camera_videos_to_tensor(video_path1: str,
                                video_path2: str,
                                greyscale: bool = False,
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load two videos from files and convert them to a tensor that is greyscale

    Parameters
    ----------
    video_path1 : str
        The path to the first video file
    video_path2 : str
        The path to the second video file
    greyscale : bool
        Whether to convert the videos to greyscale
    dtype : torch.dtype
        The datatype to convert the videos to

    Returns
    -------
    video : torch.Tensor
        The combined videos as a tensor
    """
    video1 = video_to_tensor(video_path1, greyscale, dtype)
    video2 = video_to_tensor(video_path2, greyscale, dtype)

    # Match the number of frames in the two videos to the minimum number of frames
    min_frames = min(video1.shape[0], video2.shape[0])
    video1 = video1[:min_frames]
    video2 = video2[:min_frames]

    # Now combine the two videos into a single tensor, where the first dimension are the frames and the second dimension are the two videos
    video1 = video1.unsqueeze(1)
    video2 = video2.unsqueeze(1)

    video = torch.cat((video1, video2), dim=1)

    return video
