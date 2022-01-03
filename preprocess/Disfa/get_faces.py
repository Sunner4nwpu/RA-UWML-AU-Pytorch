
import os
import json
import argparse
import platform


video_format = 'avi,mp4'


def process_video(video_dir, processed_video_dir, img_size, img_scale):
    sys = platform.system()
    if sys == 'Linux':
        executable = '/mnt/chf/software/OpenFace-master/build/bin/FeatureExtraction'
    elif sys == 'Windows':
        raise ValueError('Platform Error ...')
    else:
        raise ValueError('Platform Error ...')

    for video_i in sorted(os.listdir(video_dir)):
        if video_i.split('.')[-1] in video_format:
            print('process video: {}'.format(video_i))

            video_in = video_dir + '/' + video_i

            command = '{executable} -f {video_in} -out_dir {video_out} -simscale {img_scale} -simalign -nomask -simsize {img_size} -of {video_in_name} -2Dfp'.format(
                executable=executable, video_in=video_in, video_out=processed_video_dir, img_size=img_size, video_in_name=video_i, img_scale=img_scale
            )
            # -hogalign
            os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gene_AU_Database')
    parser.add_argument('--video_dir', type=str, default='./Videos')
    parser.add_argument('--processed_video_dir', type=str, default='./Videos_112_0.55')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--img_scale', type=float, default=0.55)
    opts = parser.parse_args()

    if os.path.isdir(opts.processed_video_dir) is False:
        os.mkdir(opts.processed_video_dir)

    process_video(opts.video_dir, opts.processed_video_dir, opts.img_size, opts.img_scale)

    print('finish ...')
